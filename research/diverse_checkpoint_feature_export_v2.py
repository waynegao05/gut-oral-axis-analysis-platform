from __future__ import annotations

import argparse
import glob
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from research.ensemble_v2 import build_loader, build_model
from research.metrics import concordance_index
from research.structured_feature_export_v2 import (
    DISAGREEMENT_FEATURE_NAMES,
    _apply_member_reference_scaler,
    _member_reference_scaler,
    _risk_disagreement_matrix,
    _select_topk_by_validation,
    _summarize_member_array,
)
from research.train_v2 import resolve_device


DEFAULT_BASELINE_CHECKPOINT_GLOB = "outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt"
DEFAULT_MAIN_MODEL_FEATURE_NPZ = (
    "outputs/current_mainline_v2/main_model_ctm_residual_v2/main_model_structured_features.npz"
)
DEFAULT_OUTPUT_DIR = "outputs/current_mainline_v2/diverse_main_model_features_v2"


@dataclass(frozen=True)
class CheckpointSpec:
    member_name: str
    checkpoint_path: str
    config_path: str
    source: str
    variant: str
    seed: int | None


def export_diverse_main_model_structured_features(
    *,
    diversity_summary_paths: Sequence[str | Path],
    baseline_checkpoint_glob: str = DEFAULT_BASELINE_CHECKPOINT_GLOB,
    base_config_path: str = "research_config_v2.yaml",
    main_model_feature_npz: str | Path = DEFAULT_MAIN_MODEL_FEATURE_NPZ,
    split_seed: int = 42,
    top_k: int = 3,
    device_arg: str = "cuda",
    include_baseline_checkpoints: bool = True,
    include_variants: Sequence[str] | None = None,
    output_npz_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    specs = collect_checkpoint_specs(
        diversity_summary_paths=diversity_summary_paths,
        baseline_checkpoint_glob=baseline_checkpoint_glob,
        base_config_path=base_config_path,
        include_baseline_checkpoints=include_baseline_checkpoints,
        include_variants=include_variants,
    )
    device = resolve_device(device_arg)
    predictions = {
        split: _predict_diverse_split(specs=specs, split_seed=split_seed, split=split, device=device)
        for split in ("train", "val", "test")
    }
    main_npz = np.load(main_model_feature_npz)
    main_risk_by_split = _load_aligned_main_model_risk(main_npz, predictions)
    val_prediction = predictions["val"]
    member_val_c_indices = [
        concordance_index(val_prediction["time"], val_prediction["event"], val_prediction["risk_matrix"][index])
        for index in range(len(specs))
    ]
    extended_member_val_c_indices = member_val_c_indices + [
        concordance_index(
            val_prediction["time"],
            val_prediction["event"],
            main_risk_by_split["val"],
        )
    ]
    top_indices = _select_topk_by_validation(extended_member_val_c_indices, top_k=top_k)
    member_val_means, member_val_stds = _member_reference_scaler(val_prediction["risk_matrix"])
    main_val_mean = float(np.mean(main_risk_by_split["val"]))
    main_val_std = float(max(np.std(main_risk_by_split["val"]), 1e-6))

    arrays: dict[str, np.ndarray] = {}
    split_summaries: dict[str, Any] = {}
    for split, prediction in predictions.items():
        raw_matrix = np.asarray(prediction["risk_matrix"], dtype=float)
        standardized_matrix = _apply_member_reference_scaler(raw_matrix, member_val_means, member_val_stds)
        extended_raw, extended_standardized = _extend_with_main_model_risk(
            raw_matrix=raw_matrix,
            standardized_matrix=standardized_matrix,
            main_risk=main_risk_by_split[split],
            main_val_mean=main_val_mean,
            main_val_std=main_val_std,
        )
        graph_summary = _summarize_member_array(prediction["graph_embedding"])
        latent_summary = _summarize_member_array(prediction["latent"])
        arrays[f"{split}_sample_ids"] = np.asarray(prediction["sample_ids"], dtype="U64")
        arrays[f"{split}_time"] = prediction["time"].astype(float)
        arrays[f"{split}_event"] = prediction["event"].astype(float)
        arrays[f"{split}_risk_matrix"] = extended_raw.astype(np.float32)
        arrays[f"{split}_standardized_risk_matrix"] = extended_standardized.astype(np.float32)
        arrays[f"{split}_raw_mean_risk"] = extended_raw.mean(axis=0).astype(np.float32)
        arrays[f"{split}_standardized_topk_risk"] = main_risk_by_split[split].astype(np.float32)
        arrays[f"{split}_risk_disagreement"] = _risk_disagreement_matrix(extended_raw, top_indices).astype(np.float32)
        arrays[f"{split}_graph_embedding_mean"] = graph_summary["mean"].astype(np.float32)
        arrays[f"{split}_graph_embedding_std"] = graph_summary["std"].astype(np.float32)
        arrays[f"{split}_latent_mean"] = latent_summary["mean"].astype(np.float32)
        arrays[f"{split}_latent_std"] = latent_summary["std"].astype(np.float32)
        arrays[f"{split}_graph_target_mean"] = prediction["graph_target"].mean(axis=0).astype(np.float32)
        arrays[f"{split}_graph_cluster_target_mean"] = prediction["graph_cluster_target"].mean(axis=0).astype(np.float32)
        split_summaries[split] = {
            "num_samples": int(len(prediction["sample_ids"])),
            "main_model_c_index": concordance_index(
                prediction["time"],
                prediction["event"],
                main_risk_by_split[split],
            ),
            "diverse_raw_mean_c_index": concordance_index(
                prediction["time"],
                prediction["event"],
                raw_matrix.mean(axis=0),
            ),
            "extended_raw_mean_c_index": concordance_index(
                prediction["time"],
                prediction["event"],
                extended_raw.mean(axis=0),
            ),
            "standardized_topk_field": "main_model_risk",
        }

    member_names = [spec.member_name for spec in specs] + ["main_model_risk"]
    metadata = {
        "diversity_summary_paths": [str(Path(path)) for path in diversity_summary_paths],
        "baseline_checkpoint_glob": baseline_checkpoint_glob,
        "base_config_path": base_config_path,
        "main_model_feature_npz": str(main_model_feature_npz),
        "split_seed": int(split_seed),
        "device": str(device),
        "top_k": int(top_k),
        "include_variants": [str(value) for value in include_variants] if include_variants is not None else None,
        "num_gnn_members": len(specs),
        "num_risk_members_with_main": len(member_names),
        "member_names": member_names,
        "checkpoint_specs": [asdict(spec) for spec in specs],
        "member_val_c_indices": [float(value) for value in member_val_c_indices],
        "extended_member_val_c_indices": [float(value) for value in extended_member_val_c_indices],
        "top_indices": [int(index) for index in top_indices.tolist()],
        "risk_reference_scaler": {
            "member_val_means": member_val_means.astype(float).tolist(),
            "member_val_stds": member_val_stds.astype(float).tolist(),
            "main_val_mean": main_val_mean,
            "main_val_std": main_val_std,
        },
        "disagreement_feature_names": DISAGREEMENT_FEATURE_NAMES,
        "split_summaries": split_summaries,
        "interpretation": (
            "Diverse checkpoint features use each checkpoint's own config_snapshot.yaml during prediction. "
            "This is required for topology-preprocessed variants such as top-k or min-weight graphs. "
            "The current research/ main model risk is injected as the fallback baseline."
        ),
    }
    if output_npz_path is not None:
        output_npz = Path(output_npz_path)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, **arrays)
    if output_json_path is not None:
        output_json = Path(output_json_path)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def collect_checkpoint_specs(
    *,
    diversity_summary_paths: Sequence[str | Path],
    baseline_checkpoint_glob: str,
    base_config_path: str,
    include_baseline_checkpoints: bool,
    include_variants: Sequence[str] | None = None,
) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    allowed_variants = {str(value) for value in include_variants} if include_variants is not None else None
    if include_baseline_checkpoints:
        for checkpoint in _glob_paths(baseline_checkpoint_glob):
            config_path = checkpoint.parent / "config_snapshot.yaml"
            if not config_path.exists():
                config_path = Path(base_config_path)
            specs.append(
                CheckpointSpec(
                    member_name=f"baseline:{checkpoint.parent.name}",
                    checkpoint_path=str(checkpoint.as_posix()),
                    config_path=str(config_path.as_posix()),
                    source="baseline",
                    variant="baseline",
                    seed=_extract_seed(checkpoint.parent.name),
                )
            )
    for summary_path in diversity_summary_paths:
        if str(summary_path).strip() == "":
            continue
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for row in summary.get("runs", []):
            variant = str(row.get("variant", "unknown"))
            if allowed_variants is not None and variant not in allowed_variants:
                continue
            output_dir = Path(str(row["output_dir"]))
            checkpoint = output_dir / "best_model.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint for diversity row: {checkpoint}")
            config_path = output_dir / "config_snapshot.yaml"
            if not config_path.exists():
                config_path = Path(str(row["config_path"]))
            seed = int(row["seed"]) if row.get("seed") is not None else None
            specs.append(
                CheckpointSpec(
                    member_name=f"{variant}:seed{seed}",
                    checkpoint_path=str(checkpoint.as_posix()),
                    config_path=str(config_path.as_posix()),
                    source=str(summary_path),
                    variant=variant,
                    seed=seed,
                )
            )
    return _deduplicate_specs(specs)


def _glob_paths(pattern: str) -> list[Path]:
    paths = [Path(path) for path in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")
    return sorted(paths)


def _deduplicate_specs(specs: Sequence[CheckpointSpec]) -> list[CheckpointSpec]:
    seen: set[str] = set()
    unique: list[CheckpointSpec] = []
    for spec in specs:
        key = str(Path(spec.checkpoint_path).resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _predict_diverse_split(
    *,
    specs: Sequence[CheckpointSpec],
    split_seed: int,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    reference_ids: list[str] | None = None
    reference_time: np.ndarray | None = None
    reference_event: np.ndarray | None = None
    risk_rows: list[np.ndarray] = []
    graph_embedding_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    graph_target_rows: list[np.ndarray] = []
    graph_cluster_target_rows: list[np.ndarray] = []

    for spec in specs:
        prediction = _predict_single_member_split(spec=spec, split_seed=split_seed, split=split, device=device)
        if reference_ids is None:
            reference_ids = prediction["sample_ids"]
            reference_time = prediction["time"]
            reference_event = prediction["event"]
        elif prediction["sample_ids"] != reference_ids:
            raise RuntimeError(f"Sample order mismatch for {spec.member_name} on split {split}.")
        risk_rows.append(prediction["risk"])
        graph_embedding_rows.append(prediction["graph_embedding"])
        latent_rows.append(prediction["latent"])
        graph_target_rows.append(prediction["graph_target"])
        graph_cluster_target_rows.append(prediction["graph_cluster_target"])
    if reference_ids is None or reference_time is None or reference_event is None:
        raise RuntimeError(f"No predictions produced for split {split}.")
    return {
        "sample_ids": reference_ids,
        "time": reference_time,
        "event": reference_event,
        "risk_matrix": np.vstack(risk_rows).astype(float),
        "graph_embedding": np.stack(graph_embedding_rows, axis=0).astype(float),
        "latent": np.stack(latent_rows, axis=0).astype(float),
        "graph_target": np.stack(graph_target_rows, axis=0).astype(float),
        "graph_cluster_target": np.stack(graph_cluster_target_rows, axis=0).astype(float),
    }


def _predict_single_member_split(
    *,
    spec: CheckpointSpec,
    split_seed: int,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(spec.config_path).read_text(encoding="utf-8"))
    loader, dataset = build_loader(config, split_seed=split_seed, split=split)
    model = build_model(config, dataset, device)
    model.load_state_dict(torch.load(spec.checkpoint_path, map_location=device))
    model.eval()
    sample_ids: list[str] = []
    time_values: list[float] = []
    event_values: list[float] = []
    risks: list[np.ndarray] = []
    graph_embeddings: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    graph_targets: list[np.ndarray] = []
    graph_cluster_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, compute_contrastive=False)
            sample_ids.extend([str(sample_id) for sample_id in batch.sample_id])
            time_values.extend(batch.time.detach().cpu().numpy().astype(float).tolist())
            event_values.extend(batch.event.detach().cpu().numpy().astype(float).tolist())
            risks.append(output["risk"].detach().cpu().numpy())
            graph_embeddings.append(output["graph_embedding"].detach().cpu().numpy())
            latents.append(output["latent"].detach().cpu().numpy())
            graph_targets.append(output["graph_target"].detach().cpu().numpy())
            graph_cluster_targets.append(output["graph_cluster_target"].detach().cpu().numpy())
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "sample_ids": sample_ids,
        "time": np.asarray(time_values, dtype=float),
        "event": np.asarray(event_values, dtype=float),
        "risk": np.concatenate(risks, axis=0).astype(float),
        "graph_embedding": np.concatenate(graph_embeddings, axis=0).astype(float),
        "latent": np.concatenate(latents, axis=0).astype(float),
        "graph_target": np.concatenate(graph_targets, axis=0).astype(float),
        "graph_cluster_target": np.concatenate(graph_cluster_targets, axis=0).astype(float),
    }


def _load_aligned_main_model_risk(
    main_npz: np.lib.npyio.NpzFile,
    predictions: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    risks: dict[str, np.ndarray] = {}
    for split, prediction in predictions.items():
        main_ids = [str(value) for value in main_npz[f"{split}_sample_ids"].tolist()]
        if main_ids != prediction["sample_ids"]:
            raise RuntimeError(f"Main model feature sample order mismatch for split {split}.")
        risks[split] = np.asarray(main_npz[f"{split}_standardized_topk_risk"], dtype=float)
    return risks


def _extend_with_main_model_risk(
    *,
    raw_matrix: np.ndarray,
    standardized_matrix: np.ndarray,
    main_risk: np.ndarray,
    main_val_mean: float,
    main_val_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    main_risk = np.asarray(main_risk, dtype=float)
    main_standardized = (main_risk - float(main_val_mean)) / float(max(main_val_std, 1e-6))
    return (
        np.vstack([np.asarray(raw_matrix, dtype=float), main_risk[None, :]]),
        np.vstack([np.asarray(standardized_matrix, dtype=float), main_standardized[None, :]]),
    )


def _extract_seed(name: str) -> int | None:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else None


def _parse_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_optional_strings(value: str) -> list[str] | None:
    parsed = _parse_paths(value)
    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diversity-summaries", required=True)
    parser.add_argument("--baseline-checkpoint-glob", default=DEFAULT_BASELINE_CHECKPOINT_GLOB)
    parser.add_argument("--base-config", default="research_config_v2.yaml")
    parser.add_argument("--main-model-feature-npz", default=DEFAULT_MAIN_MODEL_FEATURE_NPZ)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-variants", default="")
    parser.add_argument("--no-baseline-checkpoints", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = export_diverse_main_model_structured_features(
        diversity_summary_paths=_parse_paths(args.diversity_summaries),
        baseline_checkpoint_glob=args.baseline_checkpoint_glob,
        base_config_path=args.base_config,
        main_model_feature_npz=args.main_model_feature_npz,
        split_seed=args.split_seed,
        top_k=args.top_k,
        device_arg=args.device,
        include_baseline_checkpoints=not args.no_baseline_checkpoints,
        include_variants=_parse_optional_strings(args.include_variants),
        output_npz_path=output_dir / "diverse_main_model_structured_features.npz",
        output_json_path=output_dir / "diverse_main_model_structured_features_summary.json",
    )
    compact = {
        "num_gnn_members": metadata["num_gnn_members"],
        "num_risk_members_with_main": metadata["num_risk_members_with_main"],
        "top_indices": metadata["top_indices"],
        "split_summaries": metadata["split_summaries"],
    }
    (output_dir / "diverse_main_model_structured_features_compact_summary.json").write_text(
        json.dumps(compact, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
