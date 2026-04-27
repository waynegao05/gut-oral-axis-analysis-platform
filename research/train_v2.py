from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from research.data import build_dataset_from_csv, set_seed
from research.losses import (
    build_time_bin_edges,
    combined_survival_loss,
    discrete_time_nll_loss,
    pairwise_ranking_loss,
)
from research.metrics import concordance_index
from research.model_v2 import DeepStructureAwareGATCoxModelV2


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current environment.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA auto-selection failed. Use --device cpu only for explicit CPU debugging.")


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_survival_losses(
    output,
    batch,
    survival_head_type: str,
    time_bin_edges: torch.Tensor | None,
    ranking_weight: float,
    ranking_margin: float,
):
    if survival_head_type == "discrete_time":
        if time_bin_edges is None:
            raise ValueError("time_bin_edges must be provided for the discrete_time survival head.")
        discrete_time_loss = discrete_time_nll_loss(
            time_logits=output["time_logits"],
            time=batch.time,
            event=batch.event,
            time_bin_edges=time_bin_edges.to(batch.time.device),
        )
        ranking_loss = pairwise_ranking_loss(
            risk=output["risk"],
            time=batch.time,
            event=batch.event,
            margin=ranking_margin,
        )
        zero = torch.zeros((), device=batch.time.device, dtype=batch.time.dtype)
        return {
            "total": discrete_time_loss + ranking_weight * ranking_loss,
            "cox": zero,
            "ranking": ranking_loss,
            "discrete_time": discrete_time_loss,
        }

    cox_losses = combined_survival_loss(
        risk=output["risk"],
        time=batch.time,
        event=batch.event,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
    )
    return {
        "total": cox_losses["total"],
        "cox": cox_losses["cox"],
        "ranking": cox_losses["ranking"],
        "discrete_time": torch.zeros((), device=batch.time.device, dtype=batch.time.dtype),
    }


def evaluate(
    model,
    loader,
    device,
    survival_head_type: str,
    time_bin_edges: torch.Tensor | None,
    graph_aux_weight: float,
    node_aux_weight: float,
    ranking_weight: float,
    ranking_margin: float,
):
    model.eval()
    all_time, all_event, all_risk = [], [], []
    losses = []
    cox_losses = []
    ranking_losses = []
    discrete_time_losses = []
    graph_aux_losses = []
    node_aux_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, compute_contrastive=False)
            survival_losses = compute_survival_losses(
                output=output,
                batch=batch,
                survival_head_type=survival_head_type,
                time_bin_edges=time_bin_edges,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
            )
            graph_aux = output["graph_aux_loss"]
            node_aux = output["node_aux_loss"]
            total_loss = survival_losses["total"] + graph_aux_weight * graph_aux + node_aux_weight * node_aux
            losses.append(float(total_loss.item()))
            cox_losses.append(float(survival_losses["cox"].item()))
            ranking_losses.append(float(survival_losses["ranking"].item()))
            discrete_time_losses.append(float(survival_losses["discrete_time"].item()))
            graph_aux_losses.append(float(graph_aux.item()))
            node_aux_losses.append(float(node_aux.item()))
            all_time.extend(batch.time.cpu().numpy().tolist())
            all_event.extend(batch.event.cpu().numpy().tolist())
            all_risk.extend(output["risk"].cpu().numpy().tolist())
            del batch, output, survival_losses, graph_aux, node_aux, total_loss
    return {
        "head_type": survival_head_type,
        "loss": sum(losses) / max(len(losses), 1),
        "cox_loss": sum(cox_losses) / max(len(cox_losses), 1),
        "ranking_loss": sum(ranking_losses) / max(len(ranking_losses), 1),
        "discrete_time_loss": sum(discrete_time_losses) / max(len(discrete_time_losses), 1),
        "graph_aux_loss": sum(graph_aux_losses) / max(len(graph_aux_losses), 1),
        "node_aux_loss": sum(node_aux_losses) / max(len(node_aux_losses), 1),
        "c_index": concordance_index(all_time, all_event, all_risk),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--split-seed", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])
    split_seed = args.split_seed
    if split_seed is None:
        split_seed = config["train"].get("split_seed")
    graph_preprocess = config.get("graph_preprocess", {})
    survival_head_type = str(config["train"].get("survival_head_type", "cox"))
    num_time_bins = int(config["train"].get("num_time_bins", 12))

    dataset = build_dataset_from_csv(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
        node_feature_columns=config["model"]["node_feature_columns"],
        clinical_columns=config["model"]["clinical_columns"],
        metabolite_columns=config["model"]["metabolite_columns"],
        seed=config["seed"],
        split_seed=split_seed,
        keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
        min_edge_weight=graph_preprocess.get("min_edge_weight"),
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
    )
    if survival_head_type == "discrete_time":
        train_times = torch.tensor([float(item.time.item()) for item in dataset.train_set], dtype=torch.float32)
        time_bin_edges = build_time_bin_edges(train_times, num_bins=num_time_bins)
    else:
        time_bin_edges = None

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
        survival_head_type=survival_head_type,
        num_time_bins=num_time_bins,
        use_layer_attention=bool(config["train"].get("use_layer_attention", False)),
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
    ranking_weight = float(config["train"].get("ranking_weight", 0.0))
    ranking_margin = float(config["train"].get("ranking_margin", 0.0))
    ranking_warmup_epochs = int(config["train"].get("ranking_warmup_epochs", 0))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))

    best_val = float("-inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        epoch_survival_losses = []
        epoch_cox_losses = []
        epoch_ranking_losses = []
        epoch_discrete_time_losses = []
        epoch_graph_aux = []
        epoch_node_aux = []
        epoch_contrastive = []

        if ranking_weight > 0.0 and ranking_warmup_epochs > 0:
            ranking_scale = min(1.0, float(epoch) / float(ranking_warmup_epochs))
        else:
            ranking_scale = 1.0
        effective_ranking_weight = ranking_weight * ranking_scale

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(batch, compute_contrastive=contrastive_weight > 0.0)
            survival_losses = compute_survival_losses(
                output=output,
                batch=batch,
                survival_head_type=survival_head_type,
                time_bin_edges=time_bin_edges,
                ranking_weight=effective_ranking_weight,
                ranking_margin=ranking_margin,
            )
            graph_aux = output["graph_aux_loss"]
            node_aux = output["node_aux_loss"]
            contrastive = output["contrastive_loss"]

            loss = (
                survival_losses["total"]
                + graph_aux_weight * graph_aux
                + node_aux_weight * node_aux
                + contrastive_weight * contrastive
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_survival_losses.append(float(survival_losses["total"].item()))
            epoch_cox_losses.append(float(survival_losses["cox"].item()))
            epoch_ranking_losses.append(float(survival_losses["ranking"].item()))
            epoch_discrete_time_losses.append(float(survival_losses["discrete_time"].item()))
            epoch_graph_aux.append(float(graph_aux.item()))
            epoch_node_aux.append(float(node_aux.item()))
            epoch_contrastive.append(float(contrastive.item()))
            del batch, output, survival_losses, graph_aux, node_aux, contrastive, loss

        scheduler.step()
        gc.collect()

        train_survival_loss = sum(epoch_survival_losses) / max(len(epoch_survival_losses), 1)
        train_cox_loss = sum(epoch_cox_losses) / max(len(epoch_cox_losses), 1)
        train_ranking_loss = sum(epoch_ranking_losses) / max(len(epoch_ranking_losses), 1)
        train_discrete_time_loss = sum(epoch_discrete_time_losses) / max(len(epoch_discrete_time_losses), 1)
        train_graph_aux = sum(epoch_graph_aux) / max(len(epoch_graph_aux), 1)
        train_node_aux = sum(epoch_node_aux) / max(len(epoch_node_aux), 1)
        train_contrastive = sum(epoch_contrastive) / max(len(epoch_contrastive), 1)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            survival_head_type,
            time_bin_edges,
            graph_aux_weight,
            node_aux_weight,
            ranking_weight,
            ranking_margin,
        )
        history.append(
            {
                "epoch": epoch,
                "head_type": survival_head_type,
                "lr": optimizer.param_groups[0]["lr"],
                "train_survival_loss": train_survival_loss,
                "train_cox_loss": train_cox_loss,
                "train_ranking_loss": train_ranking_loss,
                "train_discrete_time_loss": train_discrete_time_loss,
                "train_graph_aux_loss": train_graph_aux,
                "train_node_aux_loss": train_node_aux,
                "train_contrastive_loss": train_contrastive,
                "effective_ranking_weight": effective_ranking_weight,
                **val_metrics,
            }
        )

        if val_metrics["c_index"] > best_val + config["train"]["min_delta"]:
            best_val = val_metrics["c_index"]
            best_state = copy.deepcopy(model.state_dict())
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
    if time_bin_edges is not None:
        (output_dir / "time_bins.json").write_text(
            json.dumps(time_bin_edges.cpu().tolist(), indent=2),
            encoding="utf-8",
        )
    (output_dir / "task_definition.json").write_text(
        json.dumps(dataset.task_definition, indent=2),
        encoding="utf-8",
    )
    (output_dir / "data_summary.json").write_text(
        json.dumps(dataset.data_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(dataset.split_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        survival_head_type,
        time_bin_edges,
        graph_aux_weight,
        node_aux_weight,
        ranking_weight,
        ranking_margin,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "device": str(device),
                "split_seed": split_seed,
                "head_type": survival_head_type,
                "best_val_c_index": best_val,
                "task_name": dataset.task_definition["task_name"],
                "dataset_version": dataset.data_summary["dataset_origin"]["dataset_version"],
                "split_strategy": dataset.split_summary["split_strategy"],
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
