from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from research.ensemble_v2 import build_loader, build_model, load_checkpoints
from research.metrics import concordance_index
from research.train_v2 import resolve_device


DISAGREEMENT_FEATURE_NAMES = [
    "risk_std_all",
    "risk_range_all",
    "risk_std_topk",
    "risk_range_topk",
    "abs_topk_minus_raw_mean",
    "max_abs_member_minus_raw_mean",
]


def export_structured_gnn_features(
    *,
    config_path: str,
    checkpoint_glob: str,
    split_seed: int = 42,
    splits: Sequence[str] = ("train", "val", "test"),
    device_arg: str = "cuda",
    top_k: int = 3,
    output_npz_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    device = resolve_device(device_arg)
    split_predictions = {
        split: _predict_structured_split(
            config=config,
            checkpoints=checkpoints,
            split_seed=split_seed,
            split=split,
            device=device,
        )
        for split in splits
    }
    if "val" not in split_predictions:
        raise ValueError("The val split is required to select validation top-k GNN members.")
    val_prediction = split_predictions["val"]
    member_val_c_indices = [
        concordance_index(val_prediction["time"], val_prediction["event"], val_prediction["risk_matrix"][index])
        for index in range(len(checkpoints))
    ]
    top_indices = _select_topk_by_validation(member_val_c_indices, top_k=top_k)
    val_means, val_stds = _member_reference_scaler(val_prediction["risk_matrix"])

    arrays: dict[str, np.ndarray] = {}
    split_summaries: dict[str, Any] = {}
    for split, prediction in split_predictions.items():
        risk_matrix = prediction["risk_matrix"]
        standardized_risk_matrix = _apply_member_reference_scaler(risk_matrix, val_means, val_stds)
        raw_mean_risk = risk_matrix.mean(axis=0)
        standardized_topk_risk = standardized_risk_matrix[top_indices].mean(axis=0)
        arrays[f"{split}_sample_ids"] = np.asarray(prediction["sample_ids"], dtype="U64")
        arrays[f"{split}_time"] = prediction["time"].astype(float)
        arrays[f"{split}_event"] = prediction["event"].astype(float)
        arrays[f"{split}_risk_matrix"] = risk_matrix.astype(np.float32)
        arrays[f"{split}_standardized_risk_matrix"] = standardized_risk_matrix.astype(np.float32)
        arrays[f"{split}_raw_mean_risk"] = raw_mean_risk.astype(np.float32)
        arrays[f"{split}_standardized_topk_risk"] = standardized_topk_risk.astype(np.float32)
        arrays[f"{split}_risk_disagreement"] = _risk_disagreement_matrix(risk_matrix, top_indices).astype(np.float32)
        graph_summary = _summarize_member_array(prediction["graph_embedding"])
        latent_summary = _summarize_member_array(prediction["latent"])
        arrays[f"{split}_graph_embedding_mean"] = graph_summary["mean"].astype(np.float32)
        arrays[f"{split}_graph_embedding_std"] = graph_summary["std"].astype(np.float32)
        arrays[f"{split}_latent_mean"] = latent_summary["mean"].astype(np.float32)
        arrays[f"{split}_latent_std"] = latent_summary["std"].astype(np.float32)
        arrays[f"{split}_graph_target_mean"] = prediction["graph_target"].mean(axis=0).astype(np.float32)
        arrays[f"{split}_graph_cluster_target_mean"] = prediction["graph_cluster_target"].mean(axis=0).astype(np.float32)
        split_summaries[split] = {
            "num_samples": int(len(prediction["sample_ids"])),
            "event_rate": float(np.mean(prediction["event"])),
            "raw_mean_c_index": concordance_index(prediction["time"], prediction["event"], raw_mean_risk),
            "standardized_topk_c_index": concordance_index(
                prediction["time"],
                prediction["event"],
                standardized_topk_risk,
            ),
            "risk_disagreement_feature_means": {
                name: float(value)
                for name, value in zip(DISAGREEMENT_FEATURE_NAMES, arrays[f"{split}_risk_disagreement"].mean(axis=0))
            },
        }

    metadata = {
        "config_path": config_path,
        "checkpoint_glob": checkpoint_glob,
        "checkpoints": [str(path) for path in checkpoints],
        "split_seed": int(split_seed),
        "device": str(device),
        "splits": list(splits),
        "num_models": len(checkpoints),
        "top_k": int(top_k),
        "top_indices": [int(index) for index in top_indices.tolist()],
        "member_val_c_indices": [float(value) for value in member_val_c_indices],
        "risk_reference_scaler": {
            "member_val_means": val_means.astype(float).tolist(),
            "member_val_stds": val_stds.astype(float).tolist(),
        },
        "disagreement_feature_names": DISAGREEMENT_FEATURE_NAMES,
        "ctm_structured_input_groups": {
            "risk_members": len(checkpoints),
            "risk_disagreement_features": DISAGREEMENT_FEATURE_NAMES,
            "graph_embedding_mean_dim": int(arrays[f"{next(iter(split_predictions))}_graph_embedding_mean"].shape[1]),
            "graph_embedding_std_dim": int(arrays[f"{next(iter(split_predictions))}_graph_embedding_std"].shape[1]),
            "latent_mean_dim": int(arrays[f"{next(iter(split_predictions))}_latent_mean"].shape[1]),
            "latent_std_dim": int(arrays[f"{next(iter(split_predictions))}_latent_std"].shape[1]),
            "topology_target_dim": int(arrays[f"{next(iter(split_predictions))}_graph_target_mean"].shape[1]),
            "cluster_target_dim": int(arrays[f"{next(iter(split_predictions))}_graph_cluster_target_mean"].shape[1]),
        },
        "split_summaries": split_summaries,
        "interpretation": (
            "Structured features expose GNN member risks, member disagreement, graph embeddings, latent fusion "
            "embeddings, and topology targets. These are intended as token groups for a constrained CTM residual "
            "rather than as a replacement for OOF evaluation."
        ),
    }
    if output_npz_path is not None:
        path = Path(output_npz_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    if output_json_path is not None:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _predict_structured_split(
    *,
    config: dict[str, Any],
    checkpoints: Sequence[Path],
    split_seed: int,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    loader, dataset = build_loader(config, split_seed=split_seed, split=split)
    sample_ids: list[str] | None = None
    time_values: np.ndarray | None = None
    event_values: np.ndarray | None = None
    risk_rows = []
    graph_embedding_rows = []
    latent_rows = []
    graph_target_rows = []
    graph_cluster_target_rows = []

    for checkpoint in checkpoints:
        model = build_model(config, dataset, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        current_ids: list[str] = []
        current_time: list[float] = []
        current_event: list[float] = []
        risks = []
        graph_embeddings = []
        latents = []
        graph_targets = []
        graph_cluster_targets = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch, compute_contrastive=False)
                current_ids.extend([str(sample_id) for sample_id in batch.sample_id])
                current_time.extend(batch.time.detach().cpu().numpy().astype(float).tolist())
                current_event.extend(batch.event.detach().cpu().numpy().astype(float).tolist())
                risks.append(output["risk"].detach().cpu().numpy())
                graph_embeddings.append(output["graph_embedding"].detach().cpu().numpy())
                latents.append(output["latent"].detach().cpu().numpy())
                graph_targets.append(output["graph_target"].detach().cpu().numpy())
                graph_cluster_targets.append(output["graph_cluster_target"].detach().cpu().numpy())
        if sample_ids is None:
            sample_ids = current_ids
            time_values = np.asarray(current_time, dtype=float)
            event_values = np.asarray(current_event, dtype=float)
        elif current_ids != sample_ids:
            raise RuntimeError(f"Checkpoint predictions are not aligned for split {split}.")
        risk_rows.append(np.concatenate(risks, axis=0))
        graph_embedding_rows.append(np.concatenate(graph_embeddings, axis=0))
        latent_rows.append(np.concatenate(latents, axis=0))
        graph_target_rows.append(np.concatenate(graph_targets, axis=0))
        graph_cluster_target_rows.append(np.concatenate(graph_cluster_targets, axis=0))

    if sample_ids is None or time_values is None or event_values is None:
        raise RuntimeError(f"No predictions produced for split {split}.")
    return {
        "sample_ids": sample_ids,
        "time": time_values,
        "event": event_values,
        "risk_matrix": np.vstack(risk_rows).astype(float),
        "graph_embedding": np.stack(graph_embedding_rows, axis=0).astype(float),
        "latent": np.stack(latent_rows, axis=0).astype(float),
        "graph_target": np.stack(graph_target_rows, axis=0).astype(float),
        "graph_cluster_target": np.stack(graph_cluster_target_rows, axis=0).astype(float),
    }


def _select_topk_by_validation(member_val_c_indices: Sequence[float], top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    values = np.asarray(member_val_c_indices, dtype=float)
    return np.argsort(values)[-min(top_k, len(values)) :][::-1]


def _member_reference_scaler(val_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(val_matrix, dtype=float)
    means = matrix.mean(axis=1)
    stds = np.maximum(matrix.std(axis=1), 1e-6)
    return means, stds


def _apply_member_reference_scaler(matrix: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=float) - means[:, None]) / stds[:, None]


def _summarize_member_array(values: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError("Expected member array with shape [num_members, num_samples, dim].")
    return {
        "mean": array.mean(axis=0),
        "std": array.std(axis=0),
    }


def _risk_disagreement_matrix(risk_matrix: np.ndarray, top_indices: Sequence[int]) -> np.ndarray:
    matrix = np.asarray(risk_matrix, dtype=float)
    raw_mean = matrix.mean(axis=0)
    top_matrix = matrix[np.asarray(top_indices, dtype=int)]
    top_mean = top_matrix.mean(axis=0)
    return np.vstack(
        [
            matrix.std(axis=0),
            matrix.max(axis=0) - matrix.min(axis=0),
            top_matrix.std(axis=0),
            top_matrix.max(axis=0) - top_matrix.min(axis=0),
            np.abs(top_mean - raw_mean),
            np.max(np.abs(matrix - raw_mean[None, :]), axis=0),
        ]
    ).T


def _parse_splits(value: str) -> list[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"train", "val", "test"}
    unknown = sorted(set(splits).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported split(s): {unknown}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument(
        "--checkpoint-glob",
        default="outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-npz",
        default="outputs/current_mainline_v2/structured_features_v2/structured_gnn_features_v2.npz",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/current_mainline_v2/structured_features_v2/structured_gnn_features_v2_summary.json",
    )
    args = parser.parse_args()
    result = export_structured_gnn_features(
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        split_seed=args.split_seed,
        splits=_parse_splits(args.splits),
        device_arg=args.device,
        top_k=args.top_k,
        output_npz_path=args.output_npz,
        output_json_path=args.output_json,
    )
    print(json.dumps(result["split_summaries"], indent=2))


if __name__ == "__main__":
    main()
