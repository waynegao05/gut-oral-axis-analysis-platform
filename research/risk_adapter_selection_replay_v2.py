from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.ensemble_stack_v2 import _cohort_cox_loss, _fit_cox_risk_scale
from research.risk_adapter_diagnostics_v2 import pairwise_change_diagnostics


def build_risk_adapter_selection_replay(
    *,
    summary_paths: Sequence[str | Path],
    min_validation_delta: float = 0.0003,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = [
        _replay_one(Path(summary_path), min_validation_delta=float(min_validation_delta))
        for summary_path in summary_paths
    ]
    test_deltas = [float(row["selected_test_delta"]) for row in rows]
    reference_c_indices = [float(row["reference_test_c_index"]) for row in rows]
    selected_c_indices = [float(row["selected_test_c_index"]) for row in rows]
    loss_deltas = [float(row["selected_cohort_cox_loss_delta"]) for row in rows]
    calibrated_loss_deltas = [
        float(row["calibrated_selected_cohort_cox_loss_delta"])
        for row in rows
    ]
    calibrated_reference_losses = [
        float(row["calibrated_reference_cohort_cox_loss"])
        for row in rows
    ]
    calibrated_selected_losses = [
        float(row["calibrated_selected_cohort_cox_loss"])
        for row in rows
    ]
    result = {
        "selection_policy": "max_validation_c_index_with_reference_fallback",
        "min_validation_delta": float(min_validation_delta),
        "num_splits": len(rows),
        "num_adapter_selections": sum(1 for row in rows if row["adapter_selected"]),
        "num_positive_test_deltas": sum(1 for value in test_deltas if value > 0.0),
        "num_nonnegative_test_deltas": sum(1 for value in test_deltas if value >= 0.0),
        "mean_reference_test_c_index": float(np.mean(reference_c_indices)) if reference_c_indices else None,
        "mean_selected_test_c_index": float(np.mean(selected_c_indices)) if selected_c_indices else None,
        "mean_selected_test_delta": float(np.mean(test_deltas)) if test_deltas else None,
        "min_selected_test_delta": min(test_deltas) if test_deltas else None,
        "max_selected_test_delta": max(test_deltas) if test_deltas else None,
        "mean_selected_cohort_cox_loss_delta": float(np.mean(loss_deltas)) if loss_deltas else None,
        "mean_calibrated_selected_cohort_cox_loss_delta": (
            float(np.mean(calibrated_loss_deltas)) if calibrated_loss_deltas else None
        ),
        "mean_calibrated_reference_cohort_cox_loss": (
            float(np.mean(calibrated_reference_losses)) if calibrated_reference_losses else None
        ),
        "mean_calibrated_selected_cohort_cox_loss": (
            float(np.mean(calibrated_selected_losses)) if calibrated_selected_losses else None
        ),
        "num_improved_calibrated_cox_losses": sum(
            1 for value in calibrated_loss_deltas if value < 0.0
        ),
        "total_corrected_pairs": sum(int(row["pair_change"]["corrected_pairs"]) for row in rows),
        "total_harmed_pairs": sum(int(row["pair_change"]["harmed_pairs"]) for row in rows),
        "total_net_corrected_pairs": sum(int(row["pair_change"]["net_corrected_pairs"]) for row in rows),
        "rows": rows,
        "interpretation": (
            "Saved candidate tables are replayed without retraining. A residual is used only when its validation "
            "c-index exceeds the saved ensemble reference by the configured threshold; otherwise the reference "
            "risk is retained."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _replay_one(summary_path: Path, *, min_validation_delta: float) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reference = summary["references"]["selection_reference"]
    original_selected = summary["selected"]
    validation_delta = float(original_selected["validation_c_index"]) - float(reference["validation_c_index"])
    adapter_selected = validation_delta >= float(min_validation_delta)

    validation_predictions = summary["validation_predictions"]
    val_time = np.asarray([row["time"] for row in validation_predictions], dtype=float)
    val_event = np.asarray([row["event"] for row in validation_predictions], dtype=float)
    reference_val_risk = np.asarray(
        [row["selection_reference_risk"] for row in validation_predictions],
        dtype=float,
    )
    candidate_val_risk = np.asarray([row["selected_risk"] for row in validation_predictions], dtype=float)
    selected_val_risk = candidate_val_risk if adapter_selected else reference_val_risk

    test_predictions = summary["test_predictions"]
    sample_ids = [str(row["sample_id"]) for row in test_predictions]
    time = np.asarray([row["time"] for row in test_predictions], dtype=float)
    event = np.asarray([row["event"] for row in test_predictions], dtype=float)
    reference_risk = np.asarray(
        [row["selection_reference_risk"] for row in test_predictions],
        dtype=float,
    )
    candidate_risk = np.asarray([row["selected_risk"] for row in test_predictions], dtype=float)
    selected_risk = candidate_risk if adapter_selected else reference_risk

    reference_test_c_index = float(reference["test_c_index"])
    selected_test_c_index = (
        float(original_selected["test_c_index"])
        if adapter_selected
        else reference_test_c_index
    )
    pair_change, _ = pairwise_change_diagnostics(
        sample_ids=sample_ids,
        time=time,
        event=event,
        baseline_risk=reference_risk,
        selected_risk=selected_risk,
    )
    reference_loss = _cohort_cox_loss(reference_risk, time, event)
    selected_loss = _cohort_cox_loss(selected_risk, time, event)
    reference_calibration = _fit_cox_risk_scale(reference_val_risk, val_time, val_event)
    selected_calibration = _fit_cox_risk_scale(selected_val_risk, val_time, val_event)
    calibrated_reference_loss = _cohort_cox_loss(
        reference_risk * float(reference_calibration["scale"]),
        time,
        event,
    )
    calibrated_selected_loss = _cohort_cox_loss(
        selected_risk * float(selected_calibration["scale"]),
        time,
        event,
    )
    return {
        "split_seed": int(summary["split_seed"]),
        "source_summary_path": str(summary_path.as_posix()),
        "reference_name": str(reference["name"]),
        "reference_validation_c_index": float(reference["validation_c_index"]),
        "reference_test_c_index": reference_test_c_index,
        "original_selected_candidate": str(original_selected["candidate_name"]),
        "original_selected_validation_delta": validation_delta,
        "adapter_selected": bool(adapter_selected),
        "selected_candidate": (
            str(original_selected["candidate_name"])
            if adapter_selected
            else f"{reference['name']}_reference"
        ),
        "selected_test_c_index": selected_test_c_index,
        "selected_test_delta": selected_test_c_index - reference_test_c_index,
        "reference_cohort_cox_loss": reference_loss,
        "selected_cohort_cox_loss": selected_loss,
        "selected_cohort_cox_loss_delta": selected_loss - reference_loss,
        "reference_cox_scale_calibration": reference_calibration,
        "selected_cox_scale_calibration": selected_calibration,
        "calibrated_reference_cohort_cox_loss": calibrated_reference_loss,
        "calibrated_selected_cohort_cox_loss": calibrated_selected_loss,
        "calibrated_selected_cohort_cox_loss_delta": calibrated_selected_loss
        - calibrated_reference_loss,
        "pair_change": pair_change.__dict__,
    }


def _parse_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--min-validation-delta", type=float, default=0.0003)
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/ranking_softmax_residual_v1/selection_replay_summary.json",
    )
    args = parser.parse_args()
    result = build_risk_adapter_selection_replay(
        summary_paths=_parse_paths(args.summaries),
        min_validation_delta=args.min_validation_delta,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
