from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from experiments.temporal_independent_v3.topology_aft_fusion import (
    _align_split,
    _impute_from_train,
    _load_mainline_predictions,
    _make_aft_dmatrix,
    build_topology_fingerprint_dataframe,
)
from research.metrics import concordance_index
from research.survival_roc_v2 import evaluate_prediction_source


SPLIT_SEEDS = (42, 43)
MODEL_SEEDS = (7, 21, 42, 123, 2026)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_eval_metric(eval_text: str, dataset_name: str) -> float:
    match = re.search(
        rf"(?:^|\s){re.escape(dataset_name)}-aft-nloglik:([-+0-9.eE]+)",
        eval_text,
    )
    if not match:
        raise ValueError(f"Could not parse {dataset_name} AFT metric from: {eval_text}")
    return float(match.group(1))


def _checkpoint_iterations(num_rounds: int, max_points: int) -> np.ndarray:
    if num_rounds <= 0:
        raise ValueError("A saved booster must contain at least one tree.")
    count = min(int(num_rounds), max(2, int(max_points)))
    return np.unique(np.rint(np.linspace(1, num_rounds, count)).astype(int))


def _build_split_matrices(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    mainline_predictions_path: Path,
) -> tuple[dict[str, xgb.DMatrix], dict[str, tuple[np.ndarray, np.ndarray]]]:
    mainline = _load_mainline_predictions(mainline_predictions_path)
    aligned = {
        split_name: _align_split(
            frame,
            mainline[f"{split_name}_sample_ids"],
            mainline[f"{split_name}_time"],
            mainline[f"{split_name}_event"],
            split_name,
        )
        for split_name in ("train", "val", "test")
    }
    train_x, val_x, test_x, _ = _impute_from_train(
        aligned["train"],
        aligned["val"],
        aligned["test"],
        feature_columns,
    )
    arrays = {"train": train_x, "val": val_x, "test": test_x}
    labels = {
        split_name: (
            np.asarray(mainline[f"{split_name}_time"], dtype=float),
            np.asarray(mainline[f"{split_name}_event"], dtype=float),
        )
        for split_name in ("train", "val", "test")
    }
    matrices = {
        split_name: _make_aft_dmatrix(
            arrays[split_name],
            labels[split_name][0],
            labels[split_name][1],
            feature_columns,
        )
        for split_name in ("train", "val", "test")
    }
    return matrices, labels


def _replay_model_trajectory(
    *,
    summary: dict[str, Any],
    matrices: dict[str, xgb.DMatrix],
    labels: dict[str, tuple[np.ndarray, np.ndarray]],
    split_seed: int,
    model_seed: int,
    max_points: int,
) -> list[dict[str, Any]]:
    selected = summary["selected_expert"]
    model_path = Path(selected["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Selected model does not exist: {model_path}")

    booster = xgb.Booster()
    booster.load_model(model_path)
    total_iterations = int(booster.num_boosted_rounds())
    expected_iterations = int(selected["best_iteration"]) + 1
    if total_iterations != expected_iterations:
        raise ValueError(
            f"Saved model rounds ({total_iterations}) do not match summary "
            f"({expected_iterations}) for split {split_seed}, seed {model_seed}."
        )

    rows: list[dict[str, Any]] = []
    for iteration in _checkpoint_iterations(total_iterations, max_points):
        partial = booster[: int(iteration)]
        evaluation = partial.eval_set(
            [(matrices["train"], "train"), (matrices["val"], "val")],
            iteration=int(iteration) - 1,
        )
        val_risk = -np.asarray(partial.predict(matrices["val"]), dtype=float)
        val_time, val_event = labels["val"]
        rows.append(
            {
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "selected_expert": str(selected["name"]),
                "iteration": int(iteration),
                "total_iterations": total_iterations,
                "training_progress_pct": 100.0 * float(iteration) / float(total_iterations),
                "train_aft_nloglik": _parse_eval_metric(evaluation, "train"),
                "val_aft_nloglik": _parse_eval_metric(evaluation, "val"),
                "val_c_index": float(concordance_index(val_time, val_event, val_risk)),
            }
        )

    final = rows[-1]
    expected_nloglik = float(selected["metrics"]["val"]["aft_nloglik"])
    expected_c_index = float(selected["metrics"]["val"]["c_index"])
    if not np.isclose(final["val_aft_nloglik"], expected_nloglik, atol=2e-6):
        raise RuntimeError(
            f"Replayed validation AFT nloglik does not match the summary for "
            f"split {split_seed}, seed {model_seed}."
        )
    if not np.isclose(final["val_c_index"], expected_c_index, atol=1e-12):
        raise RuntimeError(
            f"Replayed validation C-index does not match the summary for "
            f"split {split_seed}, seed {model_seed}."
        )
    return rows


def _export_consensus_predictions(
    *,
    input_dir: Path,
    output_dir: Path,
    split_seed: int,
) -> tuple[Path, float, int]:
    prediction_path = (
        input_dir
        / "cross_split_consensus"
        / f"split{split_seed}_consensus_predictions.npz"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(f"Consensus predictions do not exist: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as values:
        required = (
            "selected_alpha",
            "val_selected_risk",
            "test_sample_ids",
            "test_time",
            "test_event",
            "test_selected_risk",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"Consensus prediction archive is missing fields: {missing}")
        alpha = float(values["selected_alpha"])
        validation_threshold = float(np.median(values["val_selected_risk"]))
        frame = pd.DataFrame(
            {
                "split_seed": int(split_seed),
                "sample_id": values["test_sample_ids"].astype(str),
                "time": np.asarray(values["test_time"], dtype=float),
                "event": np.asarray(values["test_event"], dtype=int),
                "selected_risk": np.asarray(values["test_selected_risk"], dtype=float),
                "validation_median_risk_threshold": validation_threshold,
                "selected_alpha": alpha,
            }
        )
    numeric = frame[["time", "event", "selected_risk", "validation_median_risk_threshold"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values found in split {split_seed} consensus predictions.")
    output_path = output_dir / f"split{split_seed}_consensus_test_predictions.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_path, validation_threshold, int(len(frame))


def export_diagnostics(
    *,
    config_path: Path,
    input_dir: Path,
    output_dir: Path,
    horizons: Sequence[float],
    max_curve_points: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame, _, _ = build_topology_fingerprint_dataframe(config)

    trajectory_rows: list[dict[str, Any]] = []
    split_manifest_rows: list[dict[str, Any]] = []
    for split_seed in SPLIT_SEEDS:
        split_manifest = _read_json(input_dir / f"split{split_seed}_seed_sweep_manifest.json")
        first_summary = _read_json(input_dir / f"split{split_seed}_seed7" / "summary.json")
        feature_columns = list(first_summary["feature_metadata"]["feature_columns"])
        matrices, labels = _build_split_matrices(
            frame=frame,
            feature_columns=feature_columns,
            mainline_predictions_path=Path(split_manifest["mainline_predictions_path"]),
        )

        for model_seed in MODEL_SEEDS:
            summary_path = input_dir / f"split{split_seed}_seed{model_seed}" / "summary.json"
            summary = _read_json(summary_path)
            trajectory_rows.extend(
                _replay_model_trajectory(
                    summary=summary,
                    matrices=matrices,
                    labels=labels,
                    split_seed=split_seed,
                    model_seed=model_seed,
                    max_points=max_curve_points,
                )
            )

        prediction_csv, threshold, sample_count = _export_consensus_predictions(
            input_dir=input_dir,
            output_dir=output_dir,
            split_seed=split_seed,
        )
        consensus_prediction_path = (
            input_dir
            / "cross_split_consensus"
            / f"split{split_seed}_consensus_predictions.npz"
        )
        roc_path = output_dir / f"split{split_seed}_consensus_roc.json"
        roc_report = evaluate_prediction_source(
            config_path=str(config_path),
            split_seed=split_seed,
            prediction_path=consensus_prediction_path,
            output_path=roc_path,
            horizons=horizons,
            risk_field="test_selected_risk",
            time_field="test_time",
            event_field="test_event",
        )
        split_manifest_rows.append(
            {
                "split_seed": split_seed,
                "prediction_csv": prediction_csv.as_posix(),
                "roc_json": roc_path.as_posix(),
                "validation_median_risk_threshold": threshold,
                "sample_count": sample_count,
                "mean_horizon_auc": float(
                    np.mean([float(row["auc"]) for row in roc_report["horizons"]])
                ),
            }
        )

    trajectory_frame = pd.DataFrame(trajectory_rows)
    trajectory_path = output_dir / "selected_expert_training_trajectories.csv"
    trajectory_frame.to_csv(trajectory_path, index=False, encoding="utf-8")

    manifest = {
        "protocol": "replay_saved_selected_aft_models_for_r_figure_source",
        "visual_backend": "R-only; this exporter creates non-visual analytical source data",
        "config_path": config_path.as_posix(),
        "input_dir": input_dir.as_posix(),
        "split_seeds": list(SPLIT_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "horizons": [float(value) for value in horizons],
        "max_curve_points_per_model": int(max_curve_points),
        "trajectory_csv": trajectory_path.as_posix(),
        "trajectory_rows": int(len(trajectory_frame)),
        "splits": split_manifest_rows,
        "risk_threshold_policy": "median validation selected risk, applied unchanged to test",
        "selection_uses_test_labels": False,
    }
    manifest_path = output_dir / "figure_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_horizons(value: str) -> list[float]:
    horizons = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not horizons:
        raise ValueError("At least one horizon is required.")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument(
        "--input-dir",
        default="outputs/current_mainline_v2/temporal_independent_v3",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/current_mainline_v2/temporal_independent_v3/figure_diagnostics",
    )
    parser.add_argument("--horizons", default="36,60,84")
    parser.add_argument("--max-curve-points", type=int, default=70)
    args = parser.parse_args()
    manifest = export_diagnostics(
        config_path=Path(args.config),
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        horizons=_parse_horizons(args.horizons),
        max_curve_points=int(args.max_curve_points),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
