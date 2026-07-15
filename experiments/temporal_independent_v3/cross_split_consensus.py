from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.ensemble_stack_v2 import _cohort_cox_loss, _fit_cox_risk_scale
from research.metrics import concordance_index

from .topology_aft_fusion import (
    _pair_correction_diagnostics,
    _risk_correlation,
    _standardize_from_train,
)


def select_consensus_alpha(
    validation_curves: Sequence[Sequence[dict[str, float]]],
    *,
    gain_retention: float = 0.95,
    minimum_c_index_delta: float = 0.0003,
) -> dict[str, Any]:
    if len(validation_curves) < 2:
        raise ValueError("Consensus selection requires at least two validation curves.")
    if gain_retention <= 0.0 or gain_retention > 1.0:
        raise ValueError("gain_retention must be in (0, 1].")
    lengths = {len(curve) for curve in validation_curves}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("Validation curves must be non-empty and use the same alpha grid.")

    alpha_grid = [float(row["alpha"]) for row in validation_curves[0]]
    for curve in validation_curves[1:]:
        if not np.allclose(alpha_grid, [float(row["alpha"]) for row in curve]):
            raise ValueError("Validation curves use different alpha grids.")

    reference_c_indices = [float(curve[0]["validation_c_index"]) for curve in validation_curves]
    peak_gains = [
        max(float(row["validation_c_index"]) for row in curve) - reference
        for curve, reference in zip(validation_curves, reference_c_indices)
    ]
    required_gains = [
        max(float(minimum_c_index_delta), float(gain_retention) * peak_gain)
        for peak_gain in peak_gains
    ]

    feasible_rows = []
    aggregate_rows = []
    for index, alpha in enumerate(alpha_grid):
        split_rows = [curve[index] for curve in validation_curves]
        gains = [
            float(row["validation_c_index"]) - reference
            for row, reference in zip(split_rows, reference_c_indices)
        ]
        row = {
            "alpha": alpha,
            "split_c_index_gains": gains,
            "mean_validation_c_index": float(
                np.mean([float(item["validation_c_index"]) for item in split_rows])
            ),
            "mean_calibrated_validation_cox_loss": float(
                np.mean([float(item["calibrated_validation_cox_loss"]) for item in split_rows])
            ),
            "meets_all_split_retention": bool(
                all(gain >= required for gain, required in zip(gains, required_gains))
            ),
        }
        aggregate_rows.append(row)
        if row["meets_all_split_retention"]:
            feasible_rows.append(row)

    if feasible_rows:
        selected = sorted(
            feasible_rows,
            key=lambda row: (
                row["alpha"],
                row["mean_calibrated_validation_cox_loss"],
                -row["mean_validation_c_index"],
            ),
        )[0]
        selected_expert = bool(selected["alpha"] > 0.0)
    else:
        selected = aggregate_rows[0]
        selected_expert = False
    return {
        "gain_retention": float(gain_retention),
        "minimum_c_index_delta": float(minimum_c_index_delta),
        "reference_validation_c_indices": reference_c_indices,
        "peak_validation_c_index_gains": peak_gains,
        "required_validation_c_index_gains": required_gains,
        "selected": selected,
        "selected_expert": selected_expert,
        "num_feasible_alphas": len(feasible_rows),
        "aggregate_grid": aggregate_rows,
    }


def _load_and_standardize(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    main_values = {
        split_name: np.asarray(arrays[f"{split_name}_mainline_risk"], dtype=float)
        for split_name in ("train", "val", "test")
    }
    expert_values = {
        split_name: np.asarray(arrays[f"{split_name}_expert_ensemble_risk"], dtype=float)
        for split_name in ("train", "val", "test")
    }
    main_arrays, main_scaler = _standardize_from_train(
        main_values["train"], main_values["val"], main_values["test"]
    )
    expert_arrays, expert_scaler = _standardize_from_train(
        expert_values["train"], expert_values["val"], expert_values["test"]
    )
    return {
        "path": str(artifact_path.as_posix()),
        "arrays": arrays,
        "main": dict(zip(("train", "val", "test"), main_arrays)),
        "expert": dict(zip(("train", "val", "test"), expert_arrays)),
        "main_scaler": main_scaler,
        "expert_scaler": expert_scaler,
    }


def _validation_curve(data: dict[str, Any], alpha_grid: np.ndarray) -> list[dict[str, float]]:
    arrays = data["arrays"]
    time = np.asarray(arrays["val_time"], dtype=float)
    event = np.asarray(arrays["val_event"], dtype=float)
    rows = []
    for alpha_value in alpha_grid:
        alpha = float(alpha_value)
        risk = (1.0 - alpha) * data["main"]["val"] + alpha * data["expert"]["val"]
        calibration = _fit_cox_risk_scale(risk, time, event)
        rows.append(
            {
                "alpha": alpha,
                "validation_c_index": float(concordance_index(time, event, risk)),
                "calibrated_validation_cox_loss": float(calibration["calibrated_validation_cox_loss"]),
            }
        )
    return rows


def run_consensus(
    *,
    prediction_paths: Sequence[str | Path],
    split_seeds: Sequence[int],
    output_dir: str | Path,
    gain_retention: float = 0.95,
    minimum_c_index_delta: float = 0.0003,
    maximum_alpha: float = 1.0,
    alpha_step: float = 0.01,
) -> dict[str, Any]:
    if len(prediction_paths) != len(split_seeds):
        raise ValueError("prediction_paths and split_seeds must have the same length.")
    datasets = [_load_and_standardize(path) for path in prediction_paths]
    num_steps = int(round(float(maximum_alpha) / float(alpha_step)))
    alpha_grid = np.linspace(0.0, float(maximum_alpha), num_steps + 1)
    curves = [_validation_curve(data, alpha_grid) for data in datasets]
    selection = select_consensus_alpha(
        curves,
        gain_retention=float(gain_retention),
        minimum_c_index_delta=float(minimum_c_index_delta),
    )
    alpha = float(selection["selected"]["alpha"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for split_seed, data, curve in zip(split_seeds, datasets, curves):
        arrays = data["arrays"]
        selected = {
            split_name: (1.0 - alpha) * data["main"][split_name] + alpha * data["expert"][split_name]
            for split_name in ("train", "val", "test")
        }
        val_time = np.asarray(arrays["val_time"], dtype=float)
        val_event = np.asarray(arrays["val_event"], dtype=float)
        test_time = np.asarray(arrays["test_time"], dtype=float)
        test_event = np.asarray(arrays["test_event"], dtype=float)
        reference_calibration = _fit_cox_risk_scale(data["main"]["val"], val_time, val_event)
        selected_calibration = _fit_cox_risk_scale(selected["val"], val_time, val_event)
        reference_test_c = float(concordance_index(test_time, test_event, data["main"]["test"]))
        selected_test_c = float(concordance_index(test_time, test_event, selected["test"]))
        reference_test_loss = _cohort_cox_loss(
            data["main"]["test"] * float(reference_calibration["scale"]), test_time, test_event
        )
        selected_test_loss = _cohort_cox_loss(
            selected["test"] * float(selected_calibration["scale"]), test_time, test_event
        )
        curve_row = min(curve, key=lambda row: abs(float(row["alpha"]) - alpha))
        split_row = {
            "split_seed": int(split_seed),
            "selected_alpha": alpha,
            "validation_reference_c_index": float(curve[0]["validation_c_index"]),
            "validation_selected_c_index": float(curve_row["validation_c_index"]),
            "validation_c_index_delta": float(
                curve_row["validation_c_index"] - curve[0]["validation_c_index"]
            ),
            "test_reference_c_index": reference_test_c,
            "test_selected_c_index": selected_test_c,
            "test_c_index_delta": selected_test_c - reference_test_c,
            "test_reference_calibrated_cox_loss": reference_test_loss,
            "test_selected_calibrated_cox_loss": selected_test_loss,
            "test_calibrated_cox_loss_delta": selected_test_loss - reference_test_loss,
            "test_main_expert_correlation": _risk_correlation(
                data["main"]["test"], data["expert"]["test"]
            ),
            "test_pair_corrections": _pair_correction_diagnostics(
                test_time, test_event, data["main"]["test"], selected["test"]
            ),
        }
        split_rows.append(split_row)
        np.savez_compressed(
            output_path / f"split{int(split_seed)}_consensus_predictions.npz",
            selected_alpha=np.asarray(alpha, dtype=float),
            train_sample_ids=arrays["train_sample_ids"],
            train_selected_risk=selected["train"],
            val_sample_ids=arrays["val_sample_ids"],
            val_time=val_time,
            val_event=val_event,
            val_selected_risk=selected["val"],
            test_sample_ids=arrays["test_sample_ids"],
            test_time=test_time,
            test_event=test_event,
            test_selected_risk=selected["test"],
        )

    result = {
        "protocol": "validation_only_cross_split_gain_retention_consensus",
        "selection_uses_test_labels": False,
        "prediction_paths": [str(Path(path).as_posix()) for path in prediction_paths],
        "split_seeds": [int(value) for value in split_seeds],
        "selection": selection,
        "splits": split_rows,
        "aggregate": {
            "mean_reference_test_c_index": float(
                np.mean([row["test_reference_c_index"] for row in split_rows])
            ),
            "mean_selected_test_c_index": float(
                np.mean([row["test_selected_c_index"] for row in split_rows])
            ),
            "mean_test_c_index_delta": float(np.mean([row["test_c_index_delta"] for row in split_rows])),
            "mean_test_calibrated_cox_loss_delta": float(
                np.mean([row["test_calibrated_cox_loss_delta"] for row in split_rows])
            ),
            "num_c_index_improved_splits": int(sum(row["test_c_index_delta"] > 0.0 for row in split_rows)),
            "num_cox_loss_improved_splits": int(
                sum(row["test_calibrated_cox_loss_delta"] < 0.0 for row in split_rows)
            ),
        },
        "decision": "accept_consensus" if selection["selected_expert"] else "keep_mainline",
    }
    summary_path = output_path / "cross_split_consensus_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-paths", nargs="+", required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gain-retention", type=float, default=0.95)
    parser.add_argument("--minimum-c-index-delta", type=float, default=0.0003)
    parser.add_argument("--maximum-alpha", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    args = parser.parse_args()
    run_consensus(
        prediction_paths=args.prediction_paths,
        split_seeds=args.split_seeds,
        output_dir=args.output_dir,
        gain_retention=args.gain_retention,
        minimum_c_index_delta=args.minimum_c_index_delta,
        maximum_alpha=args.maximum_alpha,
        alpha_step=args.alpha_step,
    )


if __name__ == "__main__":
    main()
