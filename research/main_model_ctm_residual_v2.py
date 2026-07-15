from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from research.ensemble_stack_v2 import _apply_weights, _predict_split
from research.ensemble_v2 import load_checkpoints
from research.expert_stack_v2 import (
    FeatureSplit,
    _append_gnn_risk_features,
    _load_feature_splits,
    _standardize_feature_splits,
)
from research.metrics import concordance_index
from research.risk_adapter_v2 import (
    _build_standardized_disagreement_splits,
    _fit_plain_adapter,
    _standardize_gnn_members_by_validation,
)
from research.structured_ctm_oof_v2 import run_structured_ctm_oof_experiment
from research.structured_feature_export_v2 import _risk_disagreement_matrix
from research.train_v2 import resolve_device


DEFAULT_CONFIG = "research_config_v2.yaml"
DEFAULT_CHECKPOINT_GLOB = "outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt"
DEFAULT_STRUCTURED_FEATURES = (
    "outputs/current_mainline_v2/structured_features_v2/structured_gnn_features_v2_all_splits.npz"
)
DEFAULT_OUTPUT_DIR = "outputs/current_mainline_v2/main_model_ctm_residual_v2"


def run_main_model_ctm_residual_experiment(
    *,
    config_path: str = DEFAULT_CONFIG,
    checkpoint_glob: str = DEFAULT_CHECKPOINT_GLOB,
    structured_feature_npz: str = DEFAULT_STRUCTURED_FEATURES,
    split_seed: int = 42,
    device_arg: str = "cuda",
    adapter_device_arg: str = "cpu",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ctm_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    max_deltas: Sequence[float] = (0.10, 0.20, 0.35, 0.50),
    inner_folds: int = 5,
    epochs: int = 100,
    patience: int = 14,
    min_val_delta: float = 0.0,
    force_export: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_npz = output_path / "main_model_structured_features.npz"
    feature_json = output_path / "main_model_structured_features_summary.json"
    if force_export or not feature_npz.exists() or not feature_json.exists():
        export = export_main_model_structured_features(
            config_path=config_path,
            checkpoint_glob=checkpoint_glob,
            structured_feature_npz=structured_feature_npz,
            split_seed=split_seed,
            device_arg=device_arg,
            adapter_device_arg=adapter_device_arg,
            output_npz_path=feature_npz,
            output_json_path=feature_json,
        )
    else:
        export = json.loads(feature_json.read_text(encoding="utf-8"))

    ctm_summary_path = output_path / "main_model_ctm_residual_v2_summary.json"
    ctm_result = run_structured_ctm_oof_experiment(
        feature_npz_path=feature_npz,
        inner_folds=inner_folds,
        seeds=ctm_seeds,
        max_deltas=max_deltas,
        distillation_weights=(0.1,),
        delta_l2_weights=(0.08,),
        disagreement_l2_weights=(0.25,),
        alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        epochs=epochs,
        patience=patience,
        min_oof_delta=0.0,
        min_val_delta=min_val_delta,
        min_high_disagreement_val_delta=0.0,
        output_path=ctm_summary_path,
        device_arg=device_arg,
    )
    compact = {
        "main_model": export["main_model"],
        "ctm_selected": {
            key: ctm_result["selected"][key]
            for key in (
                "candidate_name",
                "alpha",
                "oof_c_index",
                "oof_delta",
                "val_c_index",
                "val_delta",
                "high_disagreement_val_delta",
                "test_c_index",
                "test_delta_vs_baseline",
            )
        },
        "pair_change": ctm_result["selected"]["pair_change"],
        "feature_summary_path": str(feature_json),
        "ctm_summary_path": str(ctm_summary_path),
        "interpretation": (
            "This experiment attaches a structured CTM residual after the current research/ main model "
            "risk_adapter_v2 selected risk. The main model risk is the baseline fallback, so alpha=0 exactly "
            "recovers the current main model."
        ),
    }
    compact_path = output_path / "main_model_ctm_residual_v2_compact_summary.json"
    compact_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    return compact


def export_main_model_structured_features(
    *,
    config_path: str,
    checkpoint_glob: str,
    structured_feature_npz: str,
    split_seed: int,
    device_arg: str,
    adapter_device_arg: str,
    output_npz_path: str | Path,
    output_json_path: str | Path,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    device = resolve_device(device_arg)
    adapter_device = resolve_device(adapter_device_arg)
    gnn_train = _predict_split(config, checkpoints, split_seed=split_seed, split="train", device=device)
    gnn_val = _predict_split(config, checkpoints, split_seed=split_seed, split="val", device=device)
    gnn_test = _predict_split(config, checkpoints, split_seed=split_seed, split="test", device=device)
    reference_weights = [1.0 / len(checkpoints) for _ in checkpoints]
    raw_train_risk = _apply_weights(gnn_train.risk_matrix, reference_weights)
    raw_val_risk = _apply_weights(gnn_val.risk_matrix, reference_weights)
    raw_test_risk = _apply_weights(gnn_test.risk_matrix, reference_weights)
    gnn_train_standardized, gnn_val_standardized, gnn_test_standardized, gnn_risk_scaler = (
        _standardize_gnn_members_by_validation(
            train_matrix=gnn_train.risk_matrix,
            val_matrix=gnn_val.risk_matrix,
            test_matrix=gnn_test.risk_matrix,
        )
    )
    member_val_c_indices = [
        concordance_index(gnn_val.time, gnn_val.event, gnn_val.risk_matrix[index])
        for index in range(gnn_val.risk_matrix.shape[0])
    ]
    top3_indices = np.argsort(np.asarray(member_val_c_indices, dtype=float))[-min(3, len(member_val_c_indices)) :][::-1]
    standardized_top3_train_risk = np.mean(gnn_train_standardized[top3_indices], axis=0)
    standardized_top3_val_risk = np.mean(gnn_val_standardized[top3_indices], axis=0)
    standardized_top3_test_risk = np.mean(gnn_test_standardized[top3_indices], axis=0)

    base_feature_names, train_raw, val_raw, test_raw = _load_feature_splits(config, split_seed)
    train_scaled, val_scaled, test_scaled, feature_scaler = _standardize_feature_splits(
        train_raw.features,
        val_raw.features,
        test_raw.features,
    )
    train_features = FeatureSplit(train_raw.sample_ids, train_raw.time, train_raw.event, train_scaled)
    val_features = FeatureSplit(val_raw.sample_ids, val_raw.time, val_raw.event, val_scaled)
    test_features = FeatureSplit(test_raw.sample_ids, test_raw.time, test_raw.event, test_scaled)
    risk_feature_names, train_features, val_features, test_features, risk_feature_scaler = _append_gnn_risk_features(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        gnn_train=gnn_train,
        gnn_val=gnn_val,
        gnn_test=gnn_test,
        reference_train_risk=raw_train_risk,
        reference_val_risk=raw_val_risk,
        reference_test_risk=raw_test_risk,
    )
    _build_standardized_disagreement_splits(
        gnn_train=gnn_train,
        gnn_val=gnn_val,
        gnn_test=gnn_test,
        raw_train_risk=raw_train_risk,
        raw_val_risk=raw_val_risk,
        raw_test_risk=raw_test_risk,
        top_indices=top3_indices,
    )
    adapter = _fit_plain_adapter(
        name="plain_mlp_standardized_top3_seed7_d0p5_dist0p1_l20p08",
        baseline_name="gnn_top3_standardized",
        input_dim=train_features.features.shape[1],
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_baseline_risk=standardized_top3_train_risk,
        val_baseline_risk=standardized_top3_val_risk,
        test_baseline_risk=standardized_top3_test_risk,
        device=adapter_device,
        seed=30007,
        epochs=140,
        patience=20,
        max_delta=0.50,
        distillation_weight=0.10,
        delta_l2_weight=0.08,
    )
    base = np.load(structured_feature_npz)
    arrays = {name: np.asarray(base[name]) for name in base.files}
    main_risk_by_split = {
        "train": np.asarray(adapter.train_risk, dtype=float),
        "val": np.asarray(adapter.val_risk, dtype=float),
        "test": np.asarray(adapter.test_risk, dtype=float),
    }
    _assert_split_order("train", arrays, gnn_train.sample_ids)
    _assert_split_order("val", arrays, gnn_val.sample_ids)
    _assert_split_order("test", arrays, gnn_test.sample_ids)
    main_mean = float(np.mean(main_risk_by_split["val"]))
    main_std = float(max(np.std(main_risk_by_split["val"]), 1e-6))
    extended_member_val_c_indices = member_val_c_indices + [
        concordance_index(gnn_val.time, gnn_val.event, main_risk_by_split["val"])
    ]
    extended_top_indices = np.argsort(np.asarray(extended_member_val_c_indices, dtype=float))[
        -min(3, len(extended_member_val_c_indices)) :
    ][::-1]
    split_summaries: dict[str, Any] = {}
    for split_name, prediction in (("train", gnn_train), ("val", gnn_val), ("test", gnn_test)):
        main_risk = main_risk_by_split[split_name]
        risk_matrix = np.vstack([np.asarray(arrays[f"{split_name}_risk_matrix"], dtype=float), main_risk[None, :]])
        main_standardized = (main_risk - main_mean) / main_std
        standardized_matrix = np.vstack(
            [
                np.asarray(arrays[f"{split_name}_standardized_risk_matrix"], dtype=float),
                main_standardized[None, :],
            ]
        )
        arrays[f"{split_name}_risk_matrix"] = risk_matrix.astype(np.float32)
        arrays[f"{split_name}_standardized_risk_matrix"] = standardized_matrix.astype(np.float32)
        arrays[f"{split_name}_standardized_topk_risk"] = main_risk.astype(np.float32)
        arrays[f"{split_name}_risk_disagreement"] = _risk_disagreement_matrix(risk_matrix, extended_top_indices).astype(np.float32)
        split_summaries[split_name] = {
            "num_samples": int(len(prediction.sample_ids)),
            "main_model_c_index": concordance_index(prediction.time, prediction.event, main_risk),
            "gnn_top3_c_index": concordance_index(
                prediction.time,
                prediction.event,
                np.asarray(base[f"{split_name}_standardized_topk_risk"], dtype=float),
            ),
        }
    output_npz = Path(output_npz_path)
    output_json = Path(output_json_path)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    metadata = {
        "config_path": config_path,
        "checkpoint_glob": checkpoint_glob,
        "split_seed": int(split_seed),
        "device": str(device),
        "adapter_device": str(adapter_device),
        "main_model": {
            "candidate_name": adapter.name,
            "validation_c_index": adapter.val_c_index,
            "test_c_index": adapter.test_c_index,
            "metadata": adapter.metadata,
        },
        "main_model_baseline_field": "standardized_topk_risk",
        "main_model_member_index": int(arrays["train_standardized_risk_matrix"].shape[0] - 1),
        "extended_top_indices_for_disagreement": [int(index) for index in extended_top_indices.tolist()],
        "gnn_risk_scaler": gnn_risk_scaler,
        "feature_names": base_feature_names + risk_feature_names,
        "feature_scaler": feature_scaler,
        "risk_feature_scaler": risk_feature_scaler,
        "main_risk_validation_scaler": {"mean": main_mean, "std": main_std},
        "split_summaries": split_summaries,
        "interpretation": (
            "The selected research/ risk_adapter_v2 main model is stored as the residual baseline. "
            "The CTM input risk members include the original GNN members plus the main-model risk as an extra member."
        ),
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _assert_split_order(split_name: str, arrays: dict[str, np.ndarray], sample_ids: Sequence[str]) -> None:
    stored = [str(value) for value in arrays[f"{split_name}_sample_ids"].tolist()]
    expected = [str(value) for value in sample_ids]
    if stored != expected:
        raise RuntimeError(f"Structured feature sample order does not match main model predictions for {split_name}.")


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-glob", default=DEFAULT_CHECKPOINT_GLOB)
    parser.add_argument("--structured-feature-npz", default=DEFAULT_STRUCTURED_FEATURES)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--adapter-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ctm-seeds", default="7,21,42,123,2026")
    parser.add_argument("--max-deltas", default="0.1,0.2,0.35,0.5")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--min-val-delta", type=float, default=0.0)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()
    result = run_main_model_ctm_residual_experiment(
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        structured_feature_npz=args.structured_feature_npz,
        split_seed=args.split_seed,
        device_arg=args.device,
        adapter_device_arg=args.adapter_device,
        output_dir=args.output_dir,
        ctm_seeds=_parse_ints(args.ctm_seeds),
        max_deltas=_parse_floats(args.max_deltas),
        inner_folds=args.inner_folds,
        epochs=args.epochs,
        patience=args.patience,
        min_val_delta=args.min_val_delta,
        force_export=args.force_export,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
