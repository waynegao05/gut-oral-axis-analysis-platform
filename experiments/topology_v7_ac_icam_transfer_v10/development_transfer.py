from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.topology_v7_ac_icam_transfer_v10.source_audit import (
    RidgeCoxModel,
)
from experiments.topology_v7_diagnosis.diagnose import GROUP_COLUMN
from experiments.topology_v7_site_outcome_transfer_v9.development_screen import (
    CoxSpec,
    _impute,
    _signed_survival_label,
)
from experiments.topology_v7_site_outcome_transfer_v9.features import (
    build_feature_frame,
)
from experiments.topology_v7_site_outcome_transfer_v9.public_prior import (
    PANEL_TAXA,
    fit_public_prior,
)
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "config/research/research_config_v7_v3_gnn_locked.yaml"
DEFAULT_DATA = (
    ROOT / "outputs/topology_v7_generator_v3_pilots/covariance_compensated"
)
DEFAULT_SOURCE_SUMMARY = (
    ROOT
    / "outputs/topology_v7_ac_icam_transfer_v10"
    / "source_audit/source_audit_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/topology_v7_ac_icam_transfer_v10/development_transfer"
)
MODEL_SPEC = CoxSpec(
    "v9_locked_stump_l10",
    max_depth=1,
    eta=0.03,
    min_child_weight=20.0,
    reg_lambda=10.0,
    reg_alpha=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
)
MODEL_SEED = 20260723
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2)


def _load_source_model(path: Path) -> tuple[RidgeCoxModel, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["source_gate"]["passed"]:
        raise RuntimeError("The AC-ICAM source gate did not pass.")
    model = RidgeCoxModel(**payload["final_source_model"])
    return model, payload


def _add_source_prior(
    frame: pd.DataFrame,
    model: RidgeCoxModel,
) -> pd.DataFrame:
    result = frame.copy()
    proxy = pd.DataFrame(index=result.index)
    for taxon in PANEL_TAXA:
        proxy[f"normal_clr__{taxon.lower()}"] = result[
            f"stool_clr__{taxon.lower()}"
        ].astype(float)
    stool_raw_columns = [
        f"stool_raw__{taxon.lower()}" for taxon in PANEL_TAXA
    ]
    proxy["normal_log_panel_load"] = np.log(
        np.clip(
            result[stool_raw_columns].astype(float).sum(axis=1).to_numpy(),
            1e-12,
            None,
        )
    )
    result["ac_icam_pfs_prior"] = model.predict(proxy)
    if not np.isfinite(result["ac_icam_pfs_prior"]).all():
        raise RuntimeError("The transferred AC-ICAM prior contains invalid values.")
    return result


def _train_xgb(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> dict[str, Any]:
    train_x, validation_x, test_x = _impute(
        train,
        validation,
        test,
        feature_columns,
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
        "max_depth": MODEL_SPEC.max_depth,
        "eta": MODEL_SPEC.eta,
        "min_child_weight": MODEL_SPEC.min_child_weight,
        "lambda": MODEL_SPEC.reg_lambda,
        "alpha": MODEL_SPEC.reg_alpha,
        "subsample": MODEL_SPEC.subsample,
        "colsample_bytree": MODEL_SPEC.colsample_bytree,
        "seed": int(seed),
        "nthread": 6,
    }
    booster = xgb.train(
        params,
        matrices["train"],
        num_boost_round=1400,
        evals=[
            (matrices["train"], "train"),
            (matrices["validation"], "validation"),
        ],
        early_stopping_rounds=70,
        verbose_eval=False,
    )
    best_iteration = (
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
        "best_iteration": best_iteration,
        "risks": risks,
        "validation_c_index": float(
            concordance_index(
                validation["time"],
                validation["event"],
                risks["validation"],
            )
        ),
        "test_c_index": float(
            concordance_index(
                test["time"],
                test["event"],
                risks["test"],
            )
        ),
    }


def _standardize_by_validation(
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = float(np.mean(validation))
    scale = max(float(np.std(validation)), 1e-12)
    return (validation - center) / scale, (test - center) / scale


def _select_residual_alpha(
    *,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    baseline_validation: np.ndarray,
    baseline_test: np.ndarray,
) -> dict[str, Any]:
    baseline_validation_z, baseline_test_z = _standardize_by_validation(
        baseline_validation,
        baseline_test,
    )
    prior_validation_z, prior_test_z = _standardize_by_validation(
        validation["ac_icam_pfs_prior"].to_numpy(float),
        test["ac_icam_pfs_prior"].to_numpy(float),
    )
    candidates: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        validation_risk = baseline_validation_z + alpha * prior_validation_z
        candidates.append(
            {
                "alpha": float(alpha),
                "validation_c_index": float(
                    concordance_index(
                        validation["time"],
                        validation["event"],
                        validation_risk,
                    )
                ),
            }
        )
    candidates.sort(
        key=lambda row: (row["validation_c_index"], -row["alpha"]),
        reverse=True,
    )
    selected = candidates[0]
    test_risk = baseline_test_z + selected["alpha"] * prior_test_z
    return {
        "selected_alpha": float(selected["alpha"]),
        "validation_c_index": float(selected["validation_c_index"]),
        "test_c_index": float(
            concordance_index(test["time"], test["event"], test_risk)
        ),
        "validation_candidates": candidates,
    }


def run_development_transfer(
    *,
    template_config_path: Path = DEFAULT_TEMPLATE,
    data_dir: Path = DEFAULT_DATA,
    source_summary_path: Path = DEFAULT_SOURCE_SUMMARY,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source_model, source_summary = _load_source_model(source_summary_path)
    debelius_prior = fit_public_prior()
    frame, feature_sets, feature_metadata = build_feature_frame(
        template_config_path=template_config_path,
        data_dir=data_dir,
        public_prior=debelius_prior,
    )
    frame = _add_source_prior(frame, source_model)
    core_columns = feature_sets["core_no_edges"]
    injection_columns = [*core_columns, "ac_icam_pfs_prior"]
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
        baseline = _train_xgb(
            train=train,
            validation=validation,
            test=test,
            feature_columns=core_columns,
            seed=MODEL_SEED + outer_group,
        )
        injection = _train_xgb(
            train=train,
            validation=validation,
            test=test,
            feature_columns=injection_columns,
            seed=MODEL_SEED + outer_group,
        )
        residual = _select_residual_alpha(
            validation=validation,
            test=test,
            baseline_validation=baseline["risks"]["validation"],
            baseline_test=baseline["risks"]["test"],
        )
        prior_only_c_index = float(
            concordance_index(
                test["time"],
                test["event"],
                test["ac_icam_pfs_prior"],
            )
        )
        row = {
            "outer_test_group": int(outer_group),
            "inner_validation_group": int(validation_group),
            "baseline_test_c_index": baseline["test_c_index"],
            "baseline_validation_c_index": baseline[
                "validation_c_index"
            ],
            "baseline_best_iteration": baseline["best_iteration"],
            "feature_injection_test_c_index": injection["test_c_index"],
            "feature_injection_validation_c_index": injection[
                "validation_c_index"
            ],
            "feature_injection_best_iteration": injection["best_iteration"],
            "feature_injection_delta": (
                injection["test_c_index"] - baseline["test_c_index"]
            ),
            "residual_test_c_index": residual["test_c_index"],
            "residual_validation_c_index": residual[
                "validation_c_index"
            ],
            "residual_selected_alpha": residual["selected_alpha"],
            "residual_delta": (
                residual["test_c_index"] - baseline["test_c_index"]
            ),
            "prior_only_test_c_index": prior_only_c_index,
            "residual_validation_candidates": residual[
                "validation_candidates"
            ],
        }
        runs.append(row)
        print(
            f"outer={outer_group} baseline={baseline['test_c_index']:.4f} "
            f"inject={injection['test_c_index']:.4f} "
            f"residual={residual['test_c_index']:.4f} "
            f"alpha={residual['selected_alpha']:.2f}",
            flush=True,
        )

    run_frame = pd.DataFrame(runs)
    baseline_scores = run_frame["baseline_test_c_index"].to_numpy(float)
    methods: list[dict[str, Any]] = []
    for method, column in (
        ("feature_injection", "feature_injection_test_c_index"),
        ("validation_calibrated_residual", "residual_test_c_index"),
    ):
        scores = run_frame[column].to_numpy(float)
        deltas = scores - baseline_scores
        methods.append(
            {
                "method": method,
                "mean_test_c_index": float(scores.mean()),
                "std_test_c_index": float(scores.std(ddof=1)),
                "minimum_test_c_index": float(scores.min()),
                "fold_deltas": [float(value) for value in deltas],
                "mean_delta": float(deltas.mean()),
                "improved_outer_groups": int((deltas > 0.0).sum()),
                "worst_fold_delta": float(deltas.min()),
            }
        )
    methods.sort(
        key=lambda row: (row["mean_delta"], row["worst_fold_delta"]),
        reverse=True,
    )
    selected = methods[0]
    checks = {
        "mean_delta_at_least_0_005": bool(selected["mean_delta"] >= 0.005),
        "at_least_three_outer_groups_improve": bool(
            selected["improved_outer_groups"] >= 3
        ),
        "worst_fold_delta_at_least_minus_0_005": bool(
            selected["worst_fold_delta"] >= -0.005
        ),
    }
    gate_passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development_transfer_screen",
        "formal_cohort_used": False,
        "source_gate": source_summary["source_gate"],
        "source_selected_candidate": source_summary["selected_candidate"],
        "mbr_distillation_used": False,
        "precomputed_edge_weights_used": False,
        "edge_relationships": (
            "learned internally by the locked stump Cox model"
        ),
        "feature_metadata": feature_metadata,
        "model_spec": MODEL_SPEC.__dict__,
        "alpha_grid": list(ALPHA_GRID),
        "runs": runs,
        "baseline": {
            "method": "v9_locked core_no_edges/stump_l10",
            "mean_test_c_index": float(baseline_scores.mean()),
            "std_test_c_index": float(baseline_scores.std(ddof=1)),
        },
        "candidate_methods": methods,
        "selected_development_candidate": selected,
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
    (output_dir / "development_transfer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_frame.drop(columns=["residual_validation_candidates"]).to_csv(
        output_dir / "development_transfer_runs.csv",
        index=False,
    )
    if gate_passed:
        formal_lock = {
            "schema_version": 1,
            "status": "locked_before_formal_audit",
            "selected_method": selected["method"],
            "source_model": source_summary["final_source_model"],
            "model_spec": MODEL_SPEC.__dict__,
            "alpha_grid": list(ALPHA_GRID),
            "model_seeds": [7, 21, 42, 123, 2026],
            "formal_rerun_policy": (
                "Run once without changing features, parameters, or seeds."
            ),
        }
        (output_dir / "formal_lock.json").write_text(
            json.dumps(formal_lock, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only transfer screen for the AC-ICAM prior."
    )
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_development_transfer(
        template_config_path=args.template_config,
        data_dir=args.data_dir,
        source_summary_path=args.source_summary,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "baseline": result["baseline"],
                "selected": result["selected_development_candidate"],
                "gate": result["development_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
