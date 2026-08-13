from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.metrics import concordance_index
from research.survival_auc_v2 import (
    _kaplan_meier_censoring_left_limit,
    cumulative_dynamic_auc,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_AUDIT_ROOT = (
    ROOT
    / "outputs"
    / "topology_v7_nested_refit_v1"
    / "audit"
    / "baseline_pooled_cox"
)
DEFAULT_COHORT_ROOT = (
    ROOT
    / "outputs"
    / "topology_v7_nested_refit_v1"
    / "cohorts"
    / "audit_seed20261003"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "topology_v7_compositional_temporal_v1"
    / "baseline"
    / "metric_ceiling_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def uno_c_index(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    tau: float,
    tied_tolerance: float = 1e-8,
) -> float:
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=int)
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if not (
        test_time.shape == test_event.shape == risk.shape
        and test_time.ndim == 1
    ):
        raise ValueError("Test time, event, and risk arrays must be aligned.")
    event_indices = np.flatnonzero(
        (test_event == 1) & (test_time <= float(tau))
    )
    if event_indices.size == 0:
        raise ValueError("Uno C-index has no observed events before tau.")
    censoring_survival = _kaplan_meier_censoring_left_limit(
        np.asarray(train_time, dtype=float),
        np.asarray(train_event, dtype=int),
        test_time[event_indices],
    )
    if np.any(censoring_survival <= 0):
        raise ValueError("Censoring survival is zero before tau.")
    numerator = 0.0
    denominator = 0.0
    for event_index, censor_survival in zip(
        event_indices,
        censoring_survival,
    ):
        later = np.flatnonzero(test_time > test_time[event_index])
        if later.size == 0:
            continue
        weight = 1.0 / float(censor_survival) ** 2
        differences = risk[event_index] - risk[later]
        concordant = float(np.count_nonzero(differences > tied_tolerance))
        tied = float(
            np.count_nonzero(np.abs(differences) <= tied_tolerance)
        )
        numerator += weight * (concordant + 0.5 * tied)
        denominator += weight * float(later.size)
    if denominator <= 0:
        raise ValueError("Uno C-index has no comparable pairs before tau.")
    return float(numerator / denominator)


def fit_breslow_survival(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    train_risk: np.ndarray,
    evaluation_risk: np.ndarray,
    horizons: Sequence[float],
) -> np.ndarray:
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=int)
    train_risk = np.asarray(train_risk, dtype=float)
    evaluation_risk = np.asarray(evaluation_risk, dtype=float)
    horizon_values = np.asarray(horizons, dtype=float)
    if not (
        train_time.shape == train_event.shape == train_risk.shape
        and train_time.ndim == 1
    ):
        raise ValueError("Training time, event, and risk must be aligned.")
    risk_offset = float(np.max(train_risk))
    exp_train_risk = np.exp(np.clip(train_risk - risk_offset, -50.0, 50.0))
    exp_evaluation_risk = np.exp(
        np.clip(evaluation_risk - risk_offset, -50.0, 50.0)
    )
    event_times = np.unique(train_time[train_event == 1])
    increments = np.asarray(
        [
            float(np.count_nonzero((train_time == event_time) & (train_event == 1)))
            / float(exp_train_risk[train_time >= event_time].sum())
            for event_time in event_times
        ],
        dtype=float,
    )
    cumulative_hazard = np.cumsum(increments)
    horizon_indices = np.searchsorted(
        event_times,
        horizon_values,
        side="right",
    ) - 1
    baseline_hazard = np.zeros_like(horizon_values, dtype=float)
    valid = horizon_indices >= 0
    baseline_hazard[valid] = cumulative_hazard[horizon_indices[valid]]
    return np.exp(
        -np.outer(exp_evaluation_risk, baseline_hazard)
    )


def ipcw_brier_score(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    survival_probability: np.ndarray,
    horizon: float,
) -> float:
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=int)
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=int)
    prediction = np.asarray(survival_probability, dtype=float)
    if not (
        test_time.shape == test_event.shape == prediction.shape
        and test_time.ndim == 1
    ):
        raise ValueError("Test arrays and survival predictions must be aligned.")
    event_before = (test_event == 1) & (test_time <= float(horizon))
    still_observed = test_time > float(horizon)
    score = np.zeros_like(test_time, dtype=float)
    if np.any(event_before):
        event_censoring_survival = _kaplan_meier_censoring_left_limit(
            train_time,
            train_event,
            test_time[event_before],
        )
        if np.any(event_censoring_survival <= 0):
            raise ValueError("Censoring survival is zero at an event time.")
        score[event_before] = (
            prediction[event_before] ** 2 / event_censoring_survival
        )
    if np.any(still_observed):
        horizon_censoring_survival = float(
            _kaplan_meier_censoring_left_limit(
                train_time,
                train_event,
                np.asarray([float(horizon)]),
            )[0]
        )
        if horizon_censoring_survival <= 0:
            raise ValueError("Censoring survival is zero at the horizon.")
        score[still_observed] = (
            (1.0 - prediction[still_observed]) ** 2
            / horizon_censoring_survival
        )
    return float(np.mean(score))


def normalized_trapezoid(
    values: Sequence[float],
    horizons: Sequence[float],
) -> float:
    x = np.asarray(horizons, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("At least two aligned horizons are required.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Horizons must be strictly increasing.")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _load_fold_ensemble(
    audit_root: Path,
    *,
    holdout_group: int,
    seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    prediction_rows: list[dict[str, np.ndarray]] = []
    for seed in seeds:
        path = (
            audit_root
            / f"holdout_group{int(holdout_group)}"
            / f"seed{int(seed)}"
            / "predictions.npz"
        )
        with np.load(path, allow_pickle=False) as values:
            prediction_rows.append(
                {key: values[key].copy() for key in values.files}
            )
    reference = prediction_rows[0]
    for row in prediction_rows[1:]:
        for field in ("train_sample_ids", "eval_sample_ids", "train_time", "train_event", "eval_time", "eval_event"):
            if not np.array_equal(reference[field], row[field]):
                raise RuntimeError(
                    f"Fold {holdout_group} prediction field is misaligned: {field}"
                )
    return {
        "train_sample_ids": reference["train_sample_ids"].astype(str),
        "eval_sample_ids": reference["eval_sample_ids"].astype(str),
        "train_time": reference["train_time"].astype(float),
        "train_event": reference["train_event"].astype(int),
        "eval_time": reference["eval_time"].astype(float),
        "eval_event": reference["eval_event"].astype(int),
        "train_risk": np.mean(
            np.stack([row["train_risk"] for row in prediction_rows]),
            axis=0,
        ),
        "eval_risk": np.mean(
            np.stack([row["eval_risk"] for row in prediction_rows]),
            axis=0,
        ),
    }


def _evaluate_risk_source(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    train_risk: np.ndarray,
    eval_time: np.ndarray,
    eval_event: np.ndarray,
    eval_risk: np.ndarray,
    report_horizons: Sequence[float],
    integration_grid: Sequence[float],
    uno_tau: float,
) -> dict[str, Any]:
    all_horizons = sorted(
        set(float(value) for value in [*report_horizons, *integration_grid])
    )
    survival = fit_breslow_survival(
        train_time=train_time,
        train_event=train_event,
        train_risk=train_risk,
        evaluation_risk=eval_risk,
        horizons=all_horizons,
    )
    auc_rows = {
        float(horizon): cumulative_dynamic_auc(
            train_time=train_time,
            train_event=train_event,
            test_time=eval_time,
            test_event=eval_event,
            risk=eval_risk,
            horizon=float(horizon),
        )
        for horizon in all_horizons
    }
    brier_rows = {
        float(horizon): ipcw_brier_score(
            train_time=train_time,
            train_event=train_event,
            test_time=eval_time,
            test_event=eval_event,
            survival_probability=survival[:, index],
            horizon=float(horizon),
        )
        for index, horizon in enumerate(all_horizons)
    }
    integration_auc = [
        float(auc_rows[float(horizon)]["auc"])
        for horizon in integration_grid
    ]
    integration_brier = [
        float(brier_rows[float(horizon)]) for horizon in integration_grid
    ]
    return {
        "harrell_c_index": float(
            concordance_index(eval_time, eval_event, eval_risk)
        ),
        "uno_c_index_tau_96": uno_c_index(
            train_time=train_time,
            train_event=train_event,
            test_time=eval_time,
            test_event=eval_event,
            risk=eval_risk,
            tau=float(uno_tau),
        ),
        "auc_by_horizon": [
            auc_rows[float(horizon)] for horizon in report_horizons
        ],
        "normalized_integrated_auc": normalized_trapezoid(
            integration_auc,
            integration_grid,
        ),
        "brier_by_horizon": [
            {
                "horizon": float(horizon),
                "brier_score": float(brier_rows[float(horizon)]),
            }
            for horizon in report_horizons
        ],
        "normalized_integrated_brier_score": normalized_trapezoid(
            integration_brier,
            integration_grid,
        ),
    }


def build_metric_ceiling_report(
    *,
    plan_path: Path,
    audit_root: Path,
    cohort_root: Path,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    metric_plan = plan["metrics"]
    report_horizons = [
        float(value) for value in metric_plan["report_horizons"]
    ]
    integration_grid = [
        float(value) for value in metric_plan["integration_grid"]
    ]
    seeds = [int(value) for value in plan["model_seeds"]]
    provenance_path = cohort_root / "topology_v7_sample_provenance.csv"
    manifest_path = cohort_root / "topology_v7_manifest.json"
    manifest = _read_json(manifest_path)
    expected_seed = int(plan["diagnostic_baseline"]["dataset_seed"])
    if int(manifest["seed"]) != expected_seed:
        raise RuntimeError("Diagnostic cohort seed does not match the plan.")
    provenance = pd.read_csv(provenance_path)
    if provenance["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Provenance contains duplicate sample IDs.")
    provenance = provenance.set_index(provenance["sample_id"].astype(str))
    if "survival_latent_risk" not in provenance.columns:
        raise RuntimeError("Diagnostic provenance has no latent-risk oracle.")

    model_folds: list[dict[str, Any]] = []
    oracle_folds: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[float]]] = {
        "model": {"time": [], "event": [], "risk": []},
        "oracle": {"time": [], "event": [], "risk": []},
    }
    for holdout_group in range(5):
        fold = _load_fold_ensemble(
            audit_root,
            holdout_group=holdout_group,
            seeds=seeds,
        )
        model_metrics = _evaluate_risk_source(
            train_time=fold["train_time"],
            train_event=fold["train_event"],
            train_risk=fold["train_risk"],
            eval_time=fold["eval_time"],
            eval_event=fold["eval_event"],
            eval_risk=fold["eval_risk"],
            report_horizons=report_horizons,
            integration_grid=integration_grid,
            uno_tau=96.0,
        )
        train_oracle = provenance.loc[
            fold["train_sample_ids"],
            "survival_latent_risk",
        ].to_numpy(dtype=float)
        eval_oracle = provenance.loc[
            fold["eval_sample_ids"],
            "survival_latent_risk",
        ].to_numpy(dtype=float)
        oracle_metrics = _evaluate_risk_source(
            train_time=fold["train_time"],
            train_event=fold["train_event"],
            train_risk=train_oracle,
            eval_time=fold["eval_time"],
            eval_event=fold["eval_event"],
            eval_risk=eval_oracle,
            report_horizons=report_horizons,
            integration_grid=integration_grid,
            uno_tau=96.0,
        )
        model_folds.append(
            {"holdout_group": holdout_group, **model_metrics}
        )
        oracle_folds.append(
            {"holdout_group": holdout_group, **oracle_metrics}
        )
        for name, train_risk, eval_risk in (
            ("model", fold["train_risk"], fold["eval_risk"]),
            ("oracle", train_oracle, eval_oracle),
        ):
            scale = float(np.std(train_risk))
            if scale <= 1e-8:
                raise RuntimeError(f"{name} training risk has zero variance.")
            pooled[name]["time"].extend(fold["eval_time"].tolist())
            pooled[name]["event"].extend(fold["eval_event"].tolist())
            pooled[name]["risk"].extend(
                ((eval_risk - np.mean(train_risk)) / scale).tolist()
            )

    def aggregate(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
        return {
            "risk_source": name,
            "macro_harrell_c_index": float(
                np.mean([row["harrell_c_index"] for row in rows])
            ),
            "macro_uno_c_index_tau_96": float(
                np.mean([row["uno_c_index_tau_96"] for row in rows])
            ),
            "macro_auc_by_horizon": [
                {
                    "horizon": float(horizon),
                    "auc": float(
                        np.mean(
                            [
                                row["auc_by_horizon"][index]["auc"]
                                for row in rows
                            ]
                        )
                    ),
                }
                for index, horizon in enumerate(report_horizons)
            ],
            "macro_normalized_integrated_auc": float(
                np.mean([row["normalized_integrated_auc"] for row in rows])
            ),
            "macro_brier_by_horizon": [
                {
                    "horizon": float(horizon),
                    "brier_score": float(
                        np.mean(
                            [
                                row["brier_by_horizon"][index]["brier_score"]
                                for row in rows
                            ]
                        )
                    ),
                }
                for index, horizon in enumerate(report_horizons)
            ],
            "macro_normalized_integrated_brier_score": float(
                np.mean(
                    [
                        row["normalized_integrated_brier_score"]
                        for row in rows
                    ]
                )
            ),
            "folds": rows,
        }

    model_aggregate = aggregate(model_folds, "five_seed_gnn_ensemble")
    oracle_aggregate = aggregate(
        oracle_folds,
        "diagnostic_generator_latent_risk",
    )
    for name, aggregate_row in (
        ("model", model_aggregate),
        ("oracle", oracle_aggregate),
    ):
        aggregate_row["train_standardized_pooled_oof_harrell_c_index"] = float(
            concordance_index(
                pooled[name]["time"],
                pooled[name]["event"],
                pooled[name]["risk"],
            )
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_only": True,
        "model_selection_allowed": False,
        "protocol": {
            "plan_path": plan_path.as_posix(),
            "plan_sha256": _sha256(plan_path),
            "audit_root": audit_root.as_posix(),
            "cohort_manifest_path": manifest_path.as_posix(),
            "cohort_manifest_sha256": _sha256(manifest_path),
            "cohort_provenance_path": provenance_path.as_posix(),
            "cohort_provenance_sha256": _sha256(provenance_path),
            "dataset_seed": int(manifest["seed"]),
            "observed_real_patient_count": int(
                manifest["observed_real_patient_count"]
            ),
            "generated_sample_count": int(manifest["sample_count"]),
            "report_horizons": report_horizons,
            "integration_grid": integration_grid,
            "uno_tau": 96.0,
        },
        "model": model_aggregate,
        "oracle": oracle_aggregate,
        "oracle_minus_model": {
            "macro_harrell_c_index": float(
                oracle_aggregate["macro_harrell_c_index"]
                - model_aggregate["macro_harrell_c_index"]
            ),
            "macro_uno_c_index_tau_96": float(
                oracle_aggregate["macro_uno_c_index_tau_96"]
                - model_aggregate["macro_uno_c_index_tau_96"]
            ),
            "macro_normalized_integrated_auc": float(
                oracle_aggregate["macro_normalized_integrated_auc"]
                - model_aggregate["macro_normalized_integrated_auc"]
            ),
            "macro_normalized_integrated_brier_score": float(
                oracle_aggregate["macro_normalized_integrated_brier_score"]
                - model_aggregate[
                    "macro_normalized_integrated_brier_score"
                ]
            ),
        },
        "interpretation": (
            "The latent-risk oracle is audit-only and cannot be used as a model "
            "feature. Its purpose is to estimate how much recoverable signal "
            "remains in the generated cohort."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_metric_ceiling_report(
        plan_path=args.plan.resolve(),
        audit_root=args.audit_root.resolve(),
        cohort_root=args.cohort_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
