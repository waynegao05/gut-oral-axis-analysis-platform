from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from research.data_balanced_scaled import build_dataset_from_csv, set_seed
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.model import GATCoxModel


def summarize_split(name, dataset_split):
    size = len(dataset_split)
    events = sum(int(float(data.event.item())) for data in dataset_split) if size > 0 else 0
    censored = size - events
    times = [float(data.time.item()) for data in dataset_split] if size > 0 else []
    return {
        "name": name,
        "size": size,
        "events": events,
        "censored": censored,
        "min_time": min(times) if times else None,
        "max_time": max(times) if times else None,
    }


def evaluate(model, loader, device):
    model.eval()
    all_time, all_event, all_risk = [], [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = cox_ph_loss(output["risk"], batch.time, batch.event)
            losses.append(float(loss.item()))
            all_time.extend(batch.time.cpu().numpy().tolist())
            all_event.extend(batch.event.cpu().numpy().tolist())
            all_risk.extend(output["risk"].cpu().numpy().tolist())

    if len(losses) == 0:
        raise RuntimeError("Evaluation split is empty. Please check split construction.")

    c_index = concordance_index(all_time, all_event, all_risk)
    return {
        "loss": sum(losses) / len(losses),
        "c_index": c_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_balanced_scaled_mid.yaml")
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

    split_summary = {
        "train": summarize_split("train", dataset.train_set),
        "val": summarize_split("val", dataset.val_set),
        "test": summarize_split("test", dataset.test_set),
    }
    print(json.dumps({"split_summary": split_summary}, indent=2))

    train_loader = DataLoader(dataset.train_set, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.val_set, batch_size=config["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=config["train"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATCoxModel(
        node_feature_dim=dataset.node_feature_dim,
        clinical_dim=dataset.clinical_dim,
        metabolite_dim=dataset.metabolite_dim,
        hidden_dim=config["train"]["hidden_dim"],
        heads=config["train"]["heads"],
        dropout=config["train"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    best_val = float("-inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = cox_ph_loss(output["risk"], batch.time, batch.event)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

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

    test_metrics = evaluate(model, test_loader, device)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    print(json.dumps({"best_val_c_index": best_val, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
