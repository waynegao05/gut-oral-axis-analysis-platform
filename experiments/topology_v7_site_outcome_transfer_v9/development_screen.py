from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.topology_v7_diagnosis.diagnose import GROUP_COLUMN
from experiments.topology_v7_site_outcome_transfer_v9.features import (
    build_feature_frame,
)
from experiments.topology_v7_site_outcome_transfer_v9.public_prior import (
    fit_public_prior,
)
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "research_config_v7_v3_gnn_locked.yaml"
DEFAULT_DATA = (
    ROOT / "outputs/topology_v7_generator_v3_pilots/covariance_compensated"
)
DEFAULT_BASELINE = (
    ROOT
    / "outputs/topology_v7_generator_v3_pilots"
    / "covariance_compensated_logo/logo_benchmark_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/topology_v7_site_outcome_transfer_v9/development"
)
MODEL_SEED = 20260723


@dataclass(frozen=True)
class CoxSpec:
    name: str
    max_depth: int
    eta: float
    min_child_weight: float
    reg_lambda: float
    reg_alpha: float
    subsample: float
    colsample_bytree: float


COX_SPECS = (
    CoxSpec("stump_l10", 1, 0.03, 20.0, 10.0, 0.1, 0.9, 0.9),
    CoxSpec("shallow_l5", 2, 0.025, 20.0, 5.0, 0.1, 0.9, 0.9),
    CoxSpec("shallow_l15", 2, 0.015, 40.0, 15.0, 0.2, 0.9, 0.9),
    CoxSpec("depth3_l15", 3, 0.015, 40.0, 15.0, 0.2, 0.9, 0.8),
    CoxSpec("depth3_l5", 3, 0.02, 20.0, 5.0, 0.1, 0.85, 0.85),
)


def _signed_survival_label(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    return np.where(event_values > 0.5, time_values, -time_values)


def _impute(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = train[columns].astype(float).replace(
        [np.inf, -np.inf], np.nan
    )
    medians = train_values.median(axis=0).fillna(0.0)
    return (
        train_values.fillna(medians).to_numpy(float),
        validation[columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .to_numpy(float),
        test[columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .to_numpy(float),
    )


def _train_cox(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    spec: CoxSpec,
    seed: int,
) -> dict[str, Any]:
    train_x, validation_x, test_x = _impute(
        train, validation, test, feature_columns
    )
    matrices = {
        "train": xgb.DMatrix(
            train_x,
            label=_signed_survival_label(train["time"], train["event"]),
            feature_names=feature_columns,
        ),
        "validation": xgb.DMatrix(
            validation_x,
            label=_signed_survival_label(
                validation["time"], validation["event"]
            ),
            feature_names=feature_columns,
        ),
        "test": xgb.DMatrix(
            test_x,
            label=_signed_survival_label(test["time"], test["event"]),
            feature_names=feature_columns,
        ),
    }
    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "tree_method": "hist",
        "max_depth": spec.max_depth,
        "eta": spec.eta,
        "min_child_weight": spec.min_child_weight,
        "lambda": spec.reg_lambda,
        "alpha": spec.reg_alpha,
        "subsample": spec.subsample,
        "colsample_bytree": spec.colsample_bytree,
        "seed": seed,
        "nthread": 6,
    }
    evaluation: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        matrices["train"],
        num_boost_round=1400,
        evals=[
            (matrices["train"], "train"),
            (matrices["validation"], "validation"),
        ],
        evals_result=evaluation,
        early_stopping_rounds=70,
        verbose_eval=False,
    )
    iteration = (
        int(booster.best_iteration) + 1
        if booster.best_iteration is not None
        else int(booster.num_boosted_rounds())
    )
    risks = {
        name: np.asarray(
            booster.predict(matrix, output_margin=True), dtype=float
        )
        for name, matrix in matrices.items()
    }
    return {
        "best_iteration": iteration,
        "train_c_index": float(
            concordance_index(
                train["time"], train["event"], risks["train"]
            )
        ),
        "validation_c_index": float(
            concordance_index(
                validation["time"],
                validation["event"],
                risks["validation"],
            )
        ),
        "test_c_index": float(
            concordance_index(test["time"], test["event"], risks["test"])
        ),
        "validation_cox_nloglik": float(
            evaluation["validation"]["cox-nloglik"][iteration - 1]
        ),
    }


def _load_baseline(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["benchmark_runs"]
        if row["dataset"] == "candidate_v3"
        and row["model_name"] == "linear_cox"
        and row["feature_set"] == "edge_identity"
        and int(row["model_seed"]) == 42
    ]
    result = {
        int(row["outer_test_group"]): float(row["test_c_index"])
        for row in rows
    }
    if sorted(result) != [0, 1, 2, 3, 4]:
        raise ValueError("Development baseline does not cover outer groups 0..4.")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_development_screen(
    *,
    template_config_path: Path = DEFAULT_TEMPLATE,
    data_dir: Path = DEFAULT_DATA,
    baseline_path: Path = DEFAULT_BASELINE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    prior = fit_public_prior()
    frame, feature_sets, feature_metadata = build_feature_frame(
        template_config_path=template_config_path,
        data_dir=data_dir,
        public_prior=prior,
    )
    groups = sorted(frame[GROUP_COLUMN].astype(int).unique().tolist())
    if groups != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected generation groups 0..4, got {groups}.")

    runs: list[dict[str, Any]] = []
    for outer_group in groups:
        validation_group = (outer_group + 1) % 5
        test = frame.loc[frame[GROUP_COLUMN].astype(int) == outer_group]
        validation = frame.loc[
            frame[GROUP_COLUMN].astype(int) == validation_group
        ]
        train = frame.loc[
            ~frame[GROUP_COLUMN].astype(int).isin(
                [outer_group, validation_group]
            )
        ]
        for feature_set, feature_columns in feature_sets.items():
            for spec in COX_SPECS:
                metrics = _train_cox(
                    train=train,
                    validation=validation,
                    test=test,
                    feature_columns=feature_columns,
                    spec=spec,
                    seed=MODEL_SEED + outer_group,
                )
                row = {
                    "outer_test_group": outer_group,
                    "inner_validation_group": validation_group,
                    "feature_set": feature_set,
                    "num_features": len(feature_columns),
                    "model_spec": spec.name,
                    **metrics,
                }
                runs.append(row)
                print(
                    f"outer={outer_group} {feature_set}/{spec.name} "
                    f"val={metrics['validation_c_index']:.4f} "
                    f"test={metrics['test_c_index']:.4f}",
                    flush=True,
                )

    run_frame = pd.DataFrame(runs)
    aggregates: list[dict[str, Any]] = []
    for keys, group in run_frame.groupby(
        ["feature_set", "model_spec"], sort=True
    ):
        scores = group["test_c_index"].astype(float)
        aggregates.append(
            {
                "feature_set": str(keys[0]),
                "model_spec": str(keys[1]),
                "mean_test_c_index": float(scores.mean()),
                "std_test_c_index": float(scores.std(ddof=1)),
                "minimum_test_c_index": float(scores.min()),
                "maximum_test_c_index": float(scores.max()),
                "mean_validation_c_index": float(
                    group["validation_c_index"].mean()
                ),
            }
        )
    aggregates.sort(
        key=lambda row: (
            row["mean_test_c_index"],
            row["minimum_test_c_index"],
        ),
        reverse=True,
    )
    selected = aggregates[0]
    selected_runs = run_frame.loc[
        (run_frame["feature_set"] == selected["feature_set"])
        & (run_frame["model_spec"] == selected["model_spec"])
    ].sort_values("outer_test_group")
    baseline = _load_baseline(baseline_path)
    deltas = [
        float(row.test_c_index) - baseline[int(row.outer_test_group)]
        for row in selected_runs.itertuples(index=False)
    ]
    baseline_mean = float(np.mean(list(baseline.values())))
    checks = {
        "mean_delta_at_least_0_005": bool(
            selected["mean_test_c_index"] - baseline_mean >= 0.005
        ),
        "at_least_three_outer_groups_improve": bool(
            sum(delta > 0.0 for delta in deltas) >= 3
        ),
        "worst_fold_delta_at_least_minus_0_005": bool(
            min(deltas) >= -0.005
        ),
    }
    gate_passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development_model_selection",
        "formal_cohort_used": False,
        "data_dir": str(data_dir.as_posix()),
        "public_prior": prior.report,
        "feature_metadata": feature_metadata,
        "candidate_specs": [spec.__dict__ for spec in COX_SPECS],
        "runs": runs,
        "aggregates": aggregates,
        "selected_development_candidate": {
            **selected,
            "baseline_model": "linear_cox/edge_identity",
            "baseline_mean_test_c_index": baseline_mean,
            "fold_deltas": deltas,
            "mean_delta": float(
                selected["mean_test_c_index"] - baseline_mean
            ),
        },
        "development_gate": {
            "passed": gate_passed,
            "checks": checks,
            "decision": (
                "write_formal_lock"
                if gate_passed
                else "reject_without_formal_audit"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "development_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_frame.to_csv(output_dir / "development_screen_runs.csv", index=False)

    if gate_passed:
        selected_spec = next(
            spec
            for spec in COX_SPECS
            if spec.name == selected["model_spec"]
        )
        formal_lock = {
            "schema_version": 1,
            "status": "locked_before_formal_audit",
            "development_summary_sha256": _sha256(summary_path),
            "selected_feature_set": selected["feature_set"],
            "selected_model_spec": selected_spec.__dict__,
            "model_seeds": [7, 21, 42, 123, 2026],
            "outer_protocol": "five generation-group leave-one-group-out",
            "inner_validation_group_rule": "(outer + 1) modulo 5",
            "seed_ensemble": (
                "equal mean of validation-standardized output margins"
            ),
            "formal_rerun_policy": (
                "Run once. Do not change features, parameters, or seeds in "
                "response to formal results."
            ),
            "public_prior": prior.report,
        }
        (output_dir / "formal_lock.json").write_text(
            json.dumps(formal_lock, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only screen for the site outcome-transfer Cox expert."
    )
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_development_screen(
        template_config_path=args.template_config,
        data_dir=args.data_dir,
        baseline_path=args.baseline,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "public_prior": result["public_prior"],
                "selected": result["selected_development_candidate"],
                "gate": result["development_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
