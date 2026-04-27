from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from research.data import build_dataset_from_csv
from research.metrics import concordance_index
from research.model_v2 import DeepStructureAwareGATCoxModelV2
from research.train_v2 import resolve_device


def load_checkpoints(checkpoint_glob: str) -> list[Path]:
    checkpoints = sorted(Path().glob(checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {checkpoint_glob}")
    return checkpoints


def build_loader(config: dict, split_seed: int | None, split: str) -> tuple[DataLoader, object]:
    graph_preprocess = config.get("graph_preprocess", {})
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
    split_map = {
        "train": dataset.train_set,
        "val": dataset.val_set,
        "test": dataset.test_set,
    }
    loader = DataLoader(split_map[split], batch_size=config["train"]["batch_size"], shuffle=False)
    return loader, dataset


def build_model(config: dict, dataset, device: torch.device) -> DeepStructureAwareGATCoxModelV2:
    return DeepStructureAwareGATCoxModelV2(
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
        survival_head_type=config["train"].get("survival_head_type", "cox"),
        num_time_bins=int(config["train"].get("num_time_bins", 12)),
        use_layer_attention=bool(config["train"].get("use_layer_attention", False)),
    ).to(device)


def evaluate_ensemble(
    config_path: str,
    checkpoint_glob: str,
    split: str,
    device_arg: str,
    split_seed: int | None,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    if split_seed is None:
        split_seed = config["train"].get("split_seed")

    loader, dataset = build_loader(config, split_seed=split_seed, split=split)
    device = resolve_device(device_arg)

    all_sample_ids: list[str] = []
    all_time: list[float] = []
    all_event: list[float] = []
    checkpoint_predictions: dict[str, list[float]] = {}

    for checkpoint_path in checkpoints:
        model = build_model(config, dataset, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        risks: list[float] = []
        sample_ids: list[str] = []
        time_values: list[float] = []
        event_values: list[float] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch, compute_contrastive=False)
                risks.extend(output["risk"].cpu().numpy().tolist())
                sample_ids.extend(list(batch.sample_id))
                time_values.extend(batch.time.cpu().numpy().tolist())
                event_values.extend(batch.event.cpu().numpy().tolist())

        if not all_sample_ids:
            all_sample_ids = sample_ids
            all_time = time_values
            all_event = event_values
        elif sample_ids != all_sample_ids:
            raise RuntimeError("Checkpoint predictions are not aligned. Use a fixed split seed for ensemble evaluation.")

        checkpoint_predictions[str(checkpoint_path)] = risks

    risk_matrix = torch.tensor(list(checkpoint_predictions.values()), dtype=torch.float32)
    ensemble_risk = risk_matrix.mean(dim=0).tolist()
    member_c_indices = {
        checkpoint: concordance_index(all_time, all_event, risks)
        for checkpoint, risks in checkpoint_predictions.items()
    }

    result = {
        "config_path": config_path,
        "split": split,
        "split_seed": split_seed,
        "num_models": len(checkpoints),
        "checkpoints": [str(path) for path in checkpoints],
        "member_c_indices": member_c_indices,
        "mean_member_c_index": float(sum(member_c_indices.values()) / max(len(member_c_indices), 1)),
        "ensemble_c_index": concordance_index(all_time, all_event, ensemble_risk),
        "predictions": [
            {
                "sample_id": sample_id,
                "time": float(time_value),
                "event": float(event_value),
                "ensemble_risk": float(risk_value),
            }
            for sample_id, time_value, event_value, risk_value in zip(
                all_sample_ids,
                all_time,
                all_event,
                ensemble_risk,
            )
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--output", default="outputs/current_mainline_v2/ensemble_summary.json")
    args = parser.parse_args()

    result = evaluate_ensemble(
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        split=args.split,
        device_arg=args.device,
        split_seed=args.split_seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
