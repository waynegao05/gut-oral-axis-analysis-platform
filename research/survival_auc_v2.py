from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from research.data import build_dataset_from_csv


def cumulative_dynamic_auc(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
    tied_tolerance: float = 1e-8,
) -> dict[str, float | int]:
    train_time_values = np.asarray(train_time, dtype=float)
    train_event_values = np.asarray(train_event, dtype=int)
    test_time_values = np.asarray(test_time, dtype=float)
    test_event_values = np.asarray(test_event, dtype=int)
    risk_values = np.asarray(risk, dtype=float)
    _validate_survival_arrays(train_time_values, train_event_values, "training")
    _validate_survival_arrays(test_time_values, test_event_values, "test")
    if not (test_time_values.shape == risk_values.shape):
        raise ValueError("test_time, test_event, and risk must have identical shapes.")
    if not np.isfinite(risk_values).all():
        raise ValueError("risk contains non-finite values.")
    if horizon <= 0.0 or horizon >= float(train_time_values.max()):
        raise ValueError("horizon must be positive and below the maximum training follow-up time.")

    case_mask = (test_event_values == 1) & (test_time_values <= float(horizon))
    control_mask = test_time_values > float(horizon)
    case_indices = np.flatnonzero(case_mask)
    control_indices = np.flatnonzero(control_mask)
    if case_indices.size == 0:
        raise ValueError(f"No observed cases are available by horizon {horizon}.")
    if control_indices.size == 0:
        raise ValueError(f"No dynamic controls are available after horizon {horizon}.")

    censoring_survival = _kaplan_meier_censoring_left_limit(
        train_time_values,
        train_event_values,
        test_time_values[case_indices],
    )
    if np.any(censoring_survival <= 0.0):
        raise ValueError("The training censoring survival is zero at one or more case times.")
    case_weights = 1.0 / censoring_survival
    control_risk = risk_values[control_indices]
    weighted_concordance = 0.0
    for case_index, case_weight in zip(case_indices, case_weights):
        differences = risk_values[case_index] - control_risk
        concordant = np.count_nonzero(differences > float(tied_tolerance))
        tied = np.count_nonzero(np.abs(differences) <= float(tied_tolerance))
        weighted_concordance += float(case_weight) * (float(concordant) + 0.5 * float(tied))
    denominator = float(control_indices.size) * float(case_weights.sum())
    auc = weighted_concordance / denominator
    return {
        "horizon": float(horizon),
        "auc": float(auc),
        "num_cases": int(case_indices.size),
        "num_controls": int(control_indices.size),
        "num_excluded_censored": int(len(test_time_values) - case_indices.size - control_indices.size),
        "ipcw_case_weight_mean": float(case_weights.mean()),
        "ipcw_case_weight_max": float(case_weights.max()),
    }


def evaluate_survival_auc(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    horizons: Sequence[float],
) -> dict[str, Any]:
    rows = [
        cumulative_dynamic_auc(
            train_time=train_time,
            train_event=train_event,
            test_time=test_time,
            test_event=test_event,
            risk=risk,
            horizon=float(horizon),
        )
        for horizon in horizons
    ]
    naive_event_auc = roc_auc_score(np.asarray(test_event, dtype=int), np.asarray(risk, dtype=float))
    return {
        "metric": "cumulative_dynamic_auc_ipcw",
        "horizons": rows,
        "arithmetic_mean_horizon_auc": float(np.mean([float(row["auc"]) for row in rows])),
        "naive_event_auc": float(naive_event_auc),
        "naive_event_auc_warning": (
            "This binary AUC ignores follow-up time and censoring and is not the primary survival metric."
        ),
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
    metrics = evaluate_survival_auc(
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
            "Cumulative cases experienced the event by each horizon, dynamic controls remained event-free beyond "
            "that horizon, and training outcomes were used to estimate inverse censoring weights."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _kaplan_meier_censoring_left_limit(
    train_time: np.ndarray,
    train_event: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    time = np.asarray(train_time, dtype=float)
    event = np.asarray(train_event, dtype=int)
    query = np.asarray(query_time, dtype=float)
    unique_times = np.unique(time)
    survival_after = np.ones(unique_times.shape[0], dtype=float)
    survival = 1.0
    for index, current_time in enumerate(unique_times):
        at_risk = int(np.count_nonzero(time >= current_time))
        censoring_events = int(np.count_nonzero((time == current_time) & (event == 0)))
        if at_risk > 0 and censoring_events > 0:
            survival *= 1.0 - (float(censoring_events) / float(at_risk))
        survival_after[index] = survival
    prior_indices = np.searchsorted(unique_times, query, side="left") - 1
    result = np.ones(query.shape, dtype=float)
    valid = prior_indices >= 0
    result[valid] = survival_after[prior_indices[valid]]
    return result


def _validate_survival_arrays(time: np.ndarray, event: np.ndarray, label: str) -> None:
    if time.ndim != 1 or event.ndim != 1 or time.shape != event.shape:
        raise ValueError(f"{label} time and event must be aligned one-dimensional arrays.")
    if time.size == 0:
        raise ValueError(f"{label} survival arrays cannot be empty.")
    if not np.isfinite(time).all() or np.any(time <= 0.0):
        raise ValueError(f"{label} time must contain finite positive values.")
    if not np.isin(event, [0, 1]).all():
        raise ValueError(f"{label} event must contain only 0 and 1.")


def _build_dataset(config: dict[str, Any], *, split_seed: int):
    graph_preprocess = config.get("graph_preprocess", {})
    tabular_preprocess = config.get("tabular_preprocess", {})
    return build_dataset_from_csv(
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
        standardize_tabular=bool(tabular_preprocess.get("standardize", False)),
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
    )


def _load_predictions(
    path: Path,
    *,
    risk_field: str,
    time_field: str,
    event_field: str,
    row_key: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as values:
            missing = [name for name in (risk_field, time_field, event_field) if name not in values]
            if missing:
                raise ValueError(f"Prediction archive is missing fields: {missing}")
            return (
                np.asarray(values[time_field], dtype=float),
                np.asarray(values[event_field], dtype=float),
                np.asarray(values[risk_field], dtype=float),
            )
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if row_key is None else data[row_key]
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON prediction rows must be a non-empty list.")
    return (
        np.asarray([row[time_field] for row in rows], dtype=float),
        np.asarray([row[event_field] for row in rows], dtype=float),
        np.asarray([row[risk_field] for row in rows], dtype=float),
    )


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


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
