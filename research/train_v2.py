from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from research.data import build_dataset_from_csv, set_seed
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.model_v2 import DeepStructureAwareGATCoxModelV2


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current environment.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(model, loader, device, graph_aux_weight: float, node_aux_weight: float):
    model.eval()
    all_time, all_event, all_risk = [], [], []
    losses = []
    graph_aux_losses = []
    node_aux_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, compute_contrastive=False)
            main_loss = cox_ph_loss(output["risk"], batch.time, batch.event)
            graph_aux = output["graph_aux_loss"]
            node_aux = output["node_aux_loss"]
            total_loss = main_loss + graph_aux_weight * graph_aux + node_aux_weight * node_aux
            losses.append(float(total_loss.item()))
            graph_aux_losses.append(float(graph_aux.item()))
            node_aux_losses.append(float(node_aux.item()))
            all_time.extend(batch.time.cpu().numpy().tolist())
            all_event.extend(batch.event.cpu().numpy().tolist())
            all_risk.extend(output["risk"].cpu().numpy().tolist())
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "graph_aux_loss": sum(graph_aux_losses) / max(len(graph_aux_losses), 1),
        "node_aux_loss": sum(node_aux_losses) / max(len(node_aux_losses), 1),
        "c_index": concordance_index(all_time, all_event, all_risk),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])

    dataset = build_dataset_from_csv(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
        node_feature_columns=config["model"]["node_feature_columns"],
        clinical_columns=config["model"]["clinical_columns"],
        metabolite_columns=config["model"]["metabolite_columns"],
        seed=config["seed"],
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
    )

    train_loader = DataLoader(dataset.train_set, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.val_set, batch_size=config["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=config["train"]["batch_size"], shuffle=False)

    device = resolve_device(args.device)
    model = DeepStructureAwareGATCoxModelV2(
        node_feature_dim=dataset.node_feature_dim,
        clinical_dim=dataset.clinical_dim,
        metabolite_dim=dataset.metabolite_dim,
        hidden_dim=config["train"]["hidden_dim"],
        heads=config["train"]["heads"],
        dropout=config["train"]["dropout"],
        edge_hidden_dim=config["train"].get("edge_hidden_dim", 24),
        num_layers=config["train"].get("num_layers", 4),
        layer_attn_heads=config["train"].get("layer_attn_heads", 4),
        contrastive_temperature=config["train"].get("contrastive_temperature", 0.2),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=config["train"]["epochs"],
        warmup_epochs=config["train"].get("warmup_epochs", 10),
    )

    graph_aux_weight = float(config["train"].get("graph_aux_weight", 0.08))
    node_aux_weight = float(config["train"].get("node_aux_weight", 0.05))
    contrastive_weight = float(config["train"].get("contrastive_weight", 0.03))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))

    best_val = float("-inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        epoch_main_losses = []
        epoch_graph_aux = []
        epoch_node_aux = []
        epoch_contrastive = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch, compute_contrastive=True)
            main_loss = cox_ph_loss(output["risk"], batch.time, batch.event)
            graph_aux = output["graph_aux_loss"]
            node_aux = output["node_aux_loss"]
            contrastive = output["contrastive_loss"]

            loss = (
                main_loss
                + graph_aux_weight * graph_aux
                + node_aux_weight * node_aux
                + contrastive_weight * contrastive
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_main_losses.append(float(main_loss.item()))
            epoch_graph_aux.append(float(graph_aux.item()))
            epoch_node_aux.append(float(node_aux.item()))
            epoch_contrastive.append(float(contrastive.item()))

        scheduler.step()

        train_main_loss = sum(epoch_main_losses) / max(len(epoch_main_losses), 1)
        train_graph_aux = sum(epoch_graph_aux) / max(len(epoch_graph_aux), 1)
        train_node_aux = sum(epoch_node_aux) / max(len(epoch_node_aux), 1)
        train_contrastive = sum(epoch_contrastive) / max(len(epoch_contrastive), 1)

        val_metrics = evaluate(model, val_loader, device, graph_aux_weight, node_aux_weight)
        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_main_loss": train_main_loss,
                "train_graph_aux_loss": train_graph_aux,
                "train_node_aux_loss": train_node_aux,
                "train_contrastive_loss": train_contrastive,
                **val_metrics,
            }
        )

        if val_metrics["c_index"] > best_val + config["train"]["min_delta"]:
            best_val = val_metrics["c_index"]
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= config["train"]["early_stop_patience"]:
                break

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, graph_aux_weight, node_aux_weight)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "device": str(device),
                "best_val_c_index": best_val,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()