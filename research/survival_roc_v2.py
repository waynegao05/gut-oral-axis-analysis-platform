from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from research.survival_auc_v2 import (
    _build_dataset,
    _kaplan_meier_censoring_left_limit,
    _load_predictions,
    _parse_float_list,
    _validate_survival_arrays,
    cumulative_dynamic_auc,
)


def cumulative_dynamic_roc(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
) -> dict[str, Any]:
    train_time_values = np.asarray(train_time, dtype=float)
    train_event_values = np.asarray(train_event, dtype=int)
    test_time_values = np.asarray(test_time, dtype=float)
    test_event_values = np.asarray(test_event, dtype=int)
    risk_values = np.asarray(risk, dtype=float)
    _validate_survival_arrays(train_time_values, train_event_values, "training")
    _validate_survival_arrays(test_time_values, test_event_values, "test")
    if test_time_values.shape != risk_values.shape:
        raise ValueError("test_time, test_event, and risk must have identical shapes.")
    if not np.isfinite(risk_values).all():
        raise ValueError("risk contains non-finite values.")

    auc_details = cumulative_dynamic_auc(
        train_time=train_time_values,
        train_event=train_event_values,
        test_time=test_time_values,
        test_event=test_event_values,
        risk=risk_values,
        horizon=float(horizon),
    )
    case_mask = (test_event_values == 1) & (test_time_values <= float(horizon))
    control_mask = test_time_values > float(horizon)
    eligible_mask = case_mask | control_mask
    case_indices = np.flatnonzero(case_mask)

    censoring_survival = _kaplan_meier_censoring_left_limit(
        train_time_values,
        train_event_values,
        test_time_values[case_indices],
    )
    if np.any(censoring_survival <= 0.0):
        raise ValueError("The training censoring survival is zero at one or more case times.")
    weights = np.ones(test_time_values.shape, dtype=float)
    weights[case_indices] = 1.0 / censoring_survival
    labels = case_mask[eligible_mask].astype(int)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        risk_values[eligible_mask],
        sample_weight=weights[eligible_mask],
        drop_intermediate=False,
    )
    curve_auc = float(
        roc_auc_score(
            labels,
            risk_values[eligible_mask],
            sample_weight=weights[eligible_mask],
        )
    )
    pairwise_auc = float(auc_details["auc"])
    if not np.isclose(curve_auc, pairwise_auc, atol=1e-10, rtol=1e-10):
        raise RuntimeError(
            "The weighted ROC AUC does not match the cumulative/dynamic pairwise AUC: "
            f"{curve_auc} != {pairwise_auc}."
        )
    return {
        **auc_details,
        "auc": curve_auc,
        "false_positive_rate": false_positive_rate.tolist(),
        "true_positive_rate": true_positive_rate.tolist(),
        "num_roc_points": int(len(false_positive_rate)),
    }


def evaluate_survival_roc(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    horizons: Sequence[float],
) -> dict[str, Any]:
    return {
        "metric": "cumulative_dynamic_roc_ipcw",
        "horizons": [
            cumulative_dynamic_roc(
                train_time=train_time,
                train_event=train_event,
                test_time=test_time,
                test_event=test_event,
                risk=risk,
                horizon=float(horizon),
            )
            for horizon in horizons
        ],
    }


def evaluate_prediction_source(
    *,
    config_path: str,
    split_seed: int,
    prediction_path: str | Path,
    output_path: str | Path,
    horizons: Sequence[float],
    risk_field: str,
    time_field: str,
    event_field: str,
    row_key: str | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset = _build_dataset(config, split_seed=split_seed)
    train_time = np.asarray([float(item.time.item()) for item in dataset.train_set], dtype=float)
    train_event = np.asarray([float(item.event.item()) for item in dataset.train_set], dtype=float)
    test_time, test_event, risk = _load_predictions(
        Path(prediction_path),
        risk_field=risk_field,
        time_field=time_field,
        event_field=event_field,
        row_key=row_key,
    )
    metrics = evaluate_survival_roc(
        train_time=train_time,
        train_event=train_event,
        test_time=test_time,
        test_event=test_event,
        risk=risk,
        horizons=horizons,
    )
    result = {
        "config_path": str(Path(config_path).as_posix()),
        "split_seed": int(split_seed),
        "prediction_path": str(Path(prediction_path).as_posix()),
        "risk_field": risk_field,
        "time_field": time_field,
        "event_field": event_field,
        "row_key": row_key,
        "sample_count": int(len(risk)),
        **metrics,
        "interpretation": (
            "At each horizon, cumulative cases are observed events by that horizon and dynamic controls "
            "remain event-free beyond it. Cases use training-derived inverse censoring weights; samples "
            "censored before the horizon are excluded."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--risk-field", required=True)
    parser.add_argument("--time-field", required=True)
    parser.add_argument("--event-field", required=True)
    parser.add_argument("--row-key", default=None)
    parser.add_argument("--horizons", default="36,60,84")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_prediction_source(
        config_path=args.config,
        split_seed=args.split_seed,
        prediction_path=args.predictions,
        output_path=args.output,
        horizons=_parse_float_list(args.horizons),
        risk_field=args.risk_field,
        time_field=args.time_field,
        event_field=args.event_field,
        row_key=args.row_key,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
