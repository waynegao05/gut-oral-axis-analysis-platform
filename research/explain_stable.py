from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from research.data_stable import build_dataset_from_csv
from research.model import GATCoxModel


def compute_node_importance(model, batch):
    batch.x.requires_grad_(True)
    output = model(batch)
    risk_sum = output["risk"].sum()
    risk_sum.backward()
    gradients = batch.x.grad.detach()
    importance = gradients.abs().sum(dim=1)
    return importance.cpu().numpy().tolist(), output["risk"].detach().cpu().numpy().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_stable_eval.yaml")
    parser.add_argument("--checkpoint", default="outputs/research_stable_eval/best_model.pt")
    parser.add_argument("--output", default="outputs/research_stable_eval/explainability.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
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
    split_map = {"train": dataset.train_set, "val": dataset.val_set, "test": dataset.test_set}
    loader = DataLoader(split_map[args.split], batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATCoxModel(
        node_feature_dim=dataset.node_feature_dim,
        clinical_dim=dataset.clinical_dim,
        metabolite_dim=dataset.metabolite_dim,
        hidden_dim=config["train"]["hidden_dim"],
        heads=config["train"]["heads"],
        dropout=config["train"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    outputs = []
    for batch in loader:
        batch = batch.to(device)
        node_importance, risk_values = compute_node_importance(model, batch)
        outputs.append(
            {
                "sample_id": batch.sample_id[0],
                "predicted_risk": float(risk_values[0]),
                "node_importance": node_importance,
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
