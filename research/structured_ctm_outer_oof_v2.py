from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from ctm_fusion_experiment.models.baseline_concat import BaselineConcatModel
from ctm_fusion_experiment.train import _evaluate_fusion, resolve_device, set_seed
from ctm_fusion_experiment.train_baseline_v9_oof import CandidateSpec
from ctm_fusion_experiment.utils.data_loader import FoldSplit, FusionArraySet, prepare_fusion_arrays
from ctm_fusion_experiment.utils.metrics import concordance_index, summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json
from research.data import build_sample_table, load_research_tables
from research.structured_ctm_oof_v2 import run_structured_ctm_oof_experiment
from research.structured_feature_export_v2 import DISAGREEMENT_FEATURE_NAMES, _risk_disagreement_matrix


@dataclass(frozen=True)
class FoldFeatureExport:
    npz_path: Path
    metadata_path: Path
    references: dict[str, float]


def run_outer_structured_ctm_oof_experiment(
    *,
    config_paths: Sequence[str | Path],
    output_dir: str | Path,
    inner_folds: int = 5,
    ctm_seeds: Sequence[int] = (42,),
    max_deltas: Sequence[float] = (0.75,),
    distillation_weights: Sequence[float] = (0.1,),
    delta_l2_weights: Sequence[float] = (0.08,),
    disagreement_l2_weights: Sequence[float] = (0.25,),
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    epochs: int = 100,
    patience: int = 14,
    device_arg: str = "cuda",
    max_seed_runs: int | None = None,
    max_folds: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    device = resolve_device(device_arg)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    selected_config_paths = list(config_paths)
    if max_seed_runs is not None:
        selected_config_paths = selected_config_paths[: int(max_seed_runs)]

    for config_path in selected_config_paths:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        source_dir = Path(config["paths"]["output_dir"])
        seed = int(config["seed"])
        sample_table = _load_sample_table(config)
        fold_dirs = sorted(source_dir.glob("fold_*"))
        if max_folds is not None:
            fold_dirs = fold_dirs[: int(max_folds)]
        for fold_dir in fold_dirs:
            fold = int(fold_dir.name.split("_")[-1])
            run_dir = output_path / f"source_seed_{seed}" / f"fold_{fold:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            feature_export = export_outer_fold_structured_features(
                config=config,
                source_fold_dir=fold_dir,
                sample_table=sample_table,
                output_npz_path=run_dir / "structured_features.npz",
                output_json_path=run_dir / "structured_features_summary.json",
                device=device,
                force=force,
            )
            summary_path = run_dir / "structured_ctm_oof_summary.json"
            if force or not summary_path.exists():
                result = run_structured_ctm_oof_experiment(
                    feature_npz_path=feature_export.npz_path,
                    inner_folds=inner_folds,
                    seeds=ctm_seeds,
                    max_deltas=max_deltas,
                    distillation_weights=distillation_weights,
                    delta_l2_weights=delta_l2_weights,
                    disagreement_l2_weights=disagreement_l2_weights,
                    alpha_grid=alpha_grid,
                    epochs=epochs,
                    patience=patience,
                    min_oof_delta=0.0,
                    min_val_delta=-0.00025,
                    min_high_disagreement_val_delta=0.0,
                    output_path=summary_path,
                    device_arg=device_arg,
                )
            else:
                result = json.loads(summary_path.read_text(encoding="utf-8"))
            selected = result["selected"]
            rows.append(
                {
                    "source_seed": seed,
                    "fold": fold,
                    "baseline_test_c_index": float(result["references"]["test_baseline_c_index"]),
                    "selected_test_c_index": float(selected["test_c_index"]),
                    "test_delta": float(selected["test_delta_vs_baseline"]),
                    "selected_candidate": selected["candidate_name"],
                    "selected_alpha": float(selected["alpha"]),
                    "oof_delta": float(selected["oof_delta"]),
                    "val_delta": float(selected["val_delta"]),
                    "high_disagreement_val_delta": float(selected["high_disagreement_val_delta"]),
                    "net_corrected_pairs": float(selected["pair_change"]["net_corrected_pairs"]),
                    "feature_npz": str(feature_export.npz_path),
                    "fold_summary": str(summary_path),
                }
            )

    if not rows:
        raise ValueError("No outer fold rows were produced.")

    baseline_scores = [row["baseline_test_c_index"] for row in rows]
    selected_scores = [row["selected_test_c_index"] for row in rows]
    result = {
        "num_outer_fold_runs": len(rows),
        "config_paths": [str(path) for path in selected_config_paths],
        "inner_folds": int(inner_folds),
        "ctm_seeds": [int(seed) for seed in ctm_seeds],
        "max_deltas": [float(value) for value in max_deltas],
        "selected_paired_comparison": summarize_paired_folds(baseline_scores, selected_scores),
        "folds": rows,
        "interpretation": (
            "Outer structured CTM v2 reuses saved fold-local graph embeddings and saved baseline head weights. "
            "For each outer fold it reconstructs train/val/test structured features, keeps the baseline v9 selected "
            "risk as the untouched reference, and lets a CTM residual pass only if train inner-OOF and validation "
            "safety gates allow it."
        ),
        "caveat": (
            "The baseline member risks are generated from saved fold-level heads rather than fully refit inside every "
            "CTM inner split. The outer test folds remain held out, but this is cheaper than a complete nested "
            "baseline-head retraining pipeline."
        ),
    }
    write_json(output_path / "structured_ctm_outer_oof_v2_summary.json", result)
    write_csv(output_path / "structured_ctm_outer_oof_v2_folds.csv", rows)
    return result


def export_outer_fold_structured_features(
    *,
    config: dict[str, Any],
    source_fold_dir: str | Path,
    sample_table: Any,
    output_npz_path: str | Path,
    output_json_path: str | Path,
    device: torch.device,
    force: bool = False,
) -> FoldFeatureExport:
    npz_path = Path(output_npz_path)
    json_path = Path(output_json_path)
    if not force and npz_path.exists() and json_path.exists():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        return FoldFeatureExport(
            npz_path=npz_path,
            metadata_path=json_path,
            references=metadata["references"],
        )

    fold_dir = Path(source_fold_dir)
    split = _load_fold_split(fold_dir / "split.json")
    graph_seeds = [int(seed) for seed in config["baseline_v5"]["graph_seeds"]]
    baseline_seeds = [int(seed) for seed in config["baseline_v5"]["baseline_seeds"]]
    selection = json.loads((fold_dir / "baseline_v9_oof_selection.json").read_text(encoding="utf-8"))
    specs = _candidate_specs(graph_seeds, baseline_seeds)
    arrays_by_graph_seed = {
        graph_seed: _load_arrays_for_graph_seed(config, sample_table, split, fold_dir, graph_seed)
        for graph_seed in graph_seeds
    }
    candidate_predictions = []
    for spec in specs:
        arrays = arrays_by_graph_seed[int(spec.graph_seed)]
        model_path = _model_path_for_spec(fold_dir, spec)
        model = _load_baseline_model(model_path, arrays, config, device)
        candidate_predictions.append(
            {
                "spec": spec,
                "train": _predict_risk(model, arrays.train, config, device),
                "val": _predict_risk(model, arrays.val, config, device),
                "test": _predict_risk(model, arrays.test, config, device),
            }
        )

    arrays: dict[str, np.ndarray] = {}
    split_summaries: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        reference_arrays = arrays_by_graph_seed[graph_seeds[0]]
        split_arrays = getattr(reference_arrays, split_name)
        risk_matrix = np.vstack([np.asarray(prediction[split_name], dtype=float) for prediction in candidate_predictions])
        standardized_risk_matrix = _standardize_with_selection_stats(
            risk_matrix,
            means=np.asarray(selection["risk_means"], dtype=float),
            stds=np.asarray(selection["risk_stds"], dtype=float),
        )
        selected_risk = np.asarray(selection["weights"], dtype=float) @ standardized_risk_matrix
        graph_member_array = np.stack(
            [
                getattr(arrays_by_graph_seed[int(prediction["spec"].graph_seed)], split_name).graph
                for prediction in candidate_predictions
            ],
            axis=0,
        )
        clinical_metabolite = np.hstack([split_arrays.clinical, split_arrays.metabolite]).astype(np.float32)
        member_val_c_indices = _member_val_c_indices(candidate_predictions, arrays_by_graph_seed[graph_seeds[0]].val)
        top_indices = _select_top_indices(member_val_c_indices, top_k=min(3, len(candidate_predictions)))

        arrays[f"{split_name}_sample_ids"] = np.asarray(split_arrays.sample_ids, dtype="U64")
        arrays[f"{split_name}_time"] = np.asarray(split_arrays.time, dtype=np.float32)
        arrays[f"{split_name}_event"] = np.asarray(split_arrays.event, dtype=np.float32)
        arrays[f"{split_name}_risk_matrix"] = risk_matrix.astype(np.float32)
        arrays[f"{split_name}_standardized_risk_matrix"] = standardized_risk_matrix.astype(np.float32)
        arrays[f"{split_name}_raw_mean_risk"] = risk_matrix.mean(axis=0).astype(np.float32)
        arrays[f"{split_name}_standardized_topk_risk"] = selected_risk.astype(np.float32)
        arrays[f"{split_name}_risk_disagreement"] = _risk_disagreement_matrix(risk_matrix, top_indices).astype(np.float32)
        arrays[f"{split_name}_graph_embedding_mean"] = graph_member_array.mean(axis=0).astype(np.float32)
        arrays[f"{split_name}_graph_embedding_std"] = graph_member_array.std(axis=0).astype(np.float32)
        arrays[f"{split_name}_latent_mean"] = clinical_metabolite
        arrays[f"{split_name}_latent_std"] = np.zeros_like(clinical_metabolite, dtype=np.float32)
        arrays[f"{split_name}_graph_target_mean"] = np.zeros((len(split_arrays.sample_ids), 1), dtype=np.float32)
        arrays[f"{split_name}_graph_cluster_target_mean"] = np.zeros((len(split_arrays.sample_ids), 1), dtype=np.float32)
        split_summaries[split_name] = {
            "num_samples": int(len(split_arrays.sample_ids)),
            "event_rate": float(np.mean(split_arrays.event)),
            "baseline_selected_c_index": concordance_index(split_arrays.time, split_arrays.event, selected_risk),
            "raw_mean_c_index": concordance_index(split_arrays.time, split_arrays.event, risk_matrix.mean(axis=0)),
            "risk_disagreement_feature_means": {
                name: float(value)
                for name, value in zip(
                    DISAGREEMENT_FEATURE_NAMES,
                    arrays[f"{split_name}_risk_disagreement"].mean(axis=0),
                )
            },
        }

    metadata = {
        "source_fold_dir": str(fold_dir),
        "fold": int(split.fold),
        "candidate_names": [spec.name for spec in specs],
        "selection_candidate_name": selection["candidate_name"],
        "selection_weights": [float(value) for value in selection["weights"]],
        "baseline_risk_field": "baseline_v9_selected_risk_stored_as_standardized_topk_risk",
        "risk_reference_scaler": {
            "risk_means": [float(value) for value in selection["risk_means"]],
            "risk_stds": [float(value) for value in selection["risk_stds"]],
        },
        "references": {
            "train_baseline_c_index": split_summaries["train"]["baseline_selected_c_index"],
            "val_baseline_c_index": split_summaries["val"]["baseline_selected_c_index"],
            "test_baseline_c_index": split_summaries["test"]["baseline_selected_c_index"],
        },
        "split_summaries": split_summaries,
        "ctm_structured_input_groups": {
            "risk_members": len(specs),
            "risk_disagreement_features": DISAGREEMENT_FEATURE_NAMES,
            "graph_embedding_mean_dim": int(arrays["train_graph_embedding_mean"].shape[1]),
            "graph_embedding_std_dim": int(arrays["train_graph_embedding_std"].shape[1]),
            "latent_mean_dim": int(arrays["train_latent_mean"].shape[1]),
            "latent_std_dim": int(arrays["train_latent_std"].shape[1]),
            "topology_target_dim": int(arrays["train_graph_target_mean"].shape[1]),
            "cluster_target_dim": int(arrays["train_graph_cluster_target_mean"].shape[1]),
        },
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return FoldFeatureExport(npz_path=npz_path, metadata_path=json_path, references=metadata["references"])


def _load_sample_table(config: dict[str, Any]):
    _, clinical_df, metabolite_df, label_df, _ = load_research_tables(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
    )
    return build_sample_table(clinical_df, metabolite_df, label_df).reset_index(drop=True)


def _load_fold_split(path: Path) -> FoldSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FoldSplit(
        fold=int(payload["fold"]),
        train_ids=tuple(str(value) for value in payload["train_ids"]),
        val_ids=tuple(str(value) for value in payload["val_ids"]),
        test_ids=tuple(str(value) for value in payload["test_ids"]),
    )


def _candidate_specs(graph_seeds: Sequence[int], baseline_seeds: Sequence[int]) -> list[CandidateSpec]:
    if not graph_seeds:
        raise ValueError("At least one graph seed is required.")
    specs = [CandidateSpec("reference", int(graph_seeds[0]), 0, is_reference=True)]
    for graph_seed in graph_seeds:
        for baseline_seed in baseline_seeds:
            specs.append(CandidateSpec(f"g{int(graph_seed)}_b{int(baseline_seed)}", int(graph_seed), int(baseline_seed)))
    return specs


def _load_arrays_for_graph_seed(
    config: dict[str, Any],
    sample_table: Any,
    split: FoldSplit,
    fold_dir: Path,
    graph_seed: int,
):
    embeddings_path = fold_dir / f"graph_seed_{int(graph_seed)}" / "frozen_graph_embeddings.npz"
    embeddings = _load_embedding_dict(embeddings_path)
    return prepare_fusion_arrays(
        sample_table=sample_table,
        split=split,
        graph_embeddings=embeddings,
        clinical_columns=config["model"]["clinical_columns"],
        metabolite_columns=config["model"]["metabolite_columns"],
    )


def _load_embedding_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        str(sample_id): np.asarray(embedding, dtype=np.float32)
        for sample_id, embedding in zip(data["sample_ids"].tolist(), data["graph_embeddings"])
    }


def _model_path_for_spec(fold_dir: Path, spec: CandidateSpec) -> Path:
    graph_dir = fold_dir / f"graph_seed_{int(spec.graph_seed)}"
    if spec.is_reference:
        return graph_dir / "reference_baseline.pt"
    return graph_dir / f"baseline_seed_{int(spec.baseline_seed)}" / "baseline_v9.pt"


def _load_baseline_model(
    path: Path,
    arrays: Any,
    config: dict[str, Any],
    device: torch.device,
) -> BaselineConcatModel:
    settings = config["fusion"]
    model = BaselineConcatModel(
        graph_dim=arrays.train.graph.shape[1],
        clinical_dim=arrays.train.clinical.shape[1],
        metabolite_dim=arrays.train.metabolite.shape[1],
        hidden_dim=int(settings["baseline_hidden_dim"]),
        dropout=float(settings["dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _predict_risk(
    model: torch.nn.Module,
    arrays: FusionArraySet,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    _, details = _evaluate_fusion(
        model,
        arrays,
        "baseline",
        int(config["fusion"]["batch_size"]),
        device,
    )
    return np.asarray(details["risk"], dtype=float)


def _standardize_with_selection_stats(
    risk_matrix: np.ndarray,
    *,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    stds = np.maximum(np.asarray(stds, dtype=float), 1e-6)
    return (np.asarray(risk_matrix, dtype=float) - np.asarray(means, dtype=float)[:, None]) / stds[:, None]


def _member_val_c_indices(candidate_predictions: Sequence[dict[str, Any]], val_arrays: FusionArraySet) -> list[float]:
    return [
        concordance_index(val_arrays.time, val_arrays.event, np.asarray(prediction["val"], dtype=float))
        for prediction in candidate_predictions
    ]


def _select_top_indices(values: Sequence[float], top_k: int) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    return np.argsort(scores)[-min(int(top_k), len(scores)) :][::-1]


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default=(
            "ctm_fusion_experiment/configs/baseline_v10_oof_replication_seed7.yaml,"
            "ctm_fusion_experiment/configs/baseline_v10_oof_replication_seed21.yaml,"
            "ctm_fusion_experiment/configs/baseline_v10_oof_replication_seed123.yaml,"
            "ctm_fusion_experiment/configs/baseline_v10_oof_replication_seed2026.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/current_mainline_v2/structured_ctm_outer_oof_v2",
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--ctm-seeds", default="42")
    parser.add_argument("--max-deltas", default="0.75")
    parser.add_argument("--distillation-weights", default="0.1")
    parser.add_argument("--delta-l2-weights", default="0.08")
    parser.add_argument("--disagreement-l2-weights", default="0.25")
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--max-seed-runs", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    result = run_outer_structured_ctm_oof_experiment(
        config_paths=[part.strip() for part in args.configs.split(",") if part.strip()],
        output_dir=args.output_dir,
        inner_folds=args.inner_folds,
        ctm_seeds=_parse_ints(args.ctm_seeds),
        max_deltas=_parse_floats(args.max_deltas),
        distillation_weights=_parse_floats(args.distillation_weights),
        delta_l2_weights=_parse_floats(args.delta_l2_weights),
        disagreement_l2_weights=_parse_floats(args.disagreement_l2_weights),
        alpha_grid=_parse_floats(args.alpha_grid),
        epochs=args.epochs,
        patience=args.patience,
        device_arg=args.device,
        max_seed_runs=args.max_seed_runs,
        max_folds=args.max_folds,
        force=args.force,
    )
    print(json.dumps(result["selected_paired_comparison"], indent=2))


if __name__ == "__main__":
    main()
