from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.ac_icam_real_outcome_v8.data import (
    CLINICAL_PATH,
    PROCESSED_ROOT,
    V8Cohort,
    load_v8_cohort,
)
from experiments.ac_icam_real_outcome_v8.modeling import (
    RiskCalibration,
    fit_clinical_cox,
    fit_microbiome_cox,
)
from research.metrics import concordance_index
from research.survival_auc_v2 import cumulative_dynamic_auc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/ac_icam_real_outcome_v8"
DEFAULT_SEEDS = (7, 21, 42, 123, 2026)
CLINICAL_L2_VALUES = (0.1, 1.0, 10.0, 100.0)
MICROBIOME_CONFIGS = tuple(
    {
        "prevalence_threshold": prevalence,
        "top_k": top_k,
        "l2": l2,
    }
    for prevalence in (0.10, 0.20)
    for top_k in (5, 10, 20)
    for l2 in (1.0, 10.0, 100.0)
)
BLEND_ALPHAS = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
HORIZONS = (36.0, 60.0)
TARGET_C_INDEX = 0.761


def _strata(
    time: np.ndarray,
    event: np.ndarray,
    *,
    n_splits: int,
) -> np.ndarray:
    frame = pd.DataFrame({"time": time, "event": event.astype(int)})
    for bins in (4, 3, 2):
        time_bin = pd.qcut(
            frame["time"].rank(method="first"),
            q=bins,
            labels=False,
            duplicates="drop",
        )
        combined = (
            frame["event"].astype(str)
            + "_"
            + time_bin.astype(int).astype(str)
        )
        if int(combined.value_counts().min()) >= int(n_splits):
            return combined.to_numpy()
    return frame["event"].to_numpy()


def _calibrated_predictions(
    train_risk: np.ndarray,
    test_risk: np.ndarray,
) -> np.ndarray:
    return RiskCalibration.fit(train_risk).transform(test_risk)


def _inner_splits(
    time: np.ndarray,
    event: np.ndarray,
    *,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=int(seed),
    )
    return list(
        splitter.split(
            np.arange(len(time)),
            _strata(time, event, n_splits=3),
        )
    )


def _clinical_inner_selection(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    include_icr: bool,
    include_treatment: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_rows = list(splits)
    for l2 in CLINICAL_L2_VALUES:
        oof = np.full(len(frame), np.nan, dtype=float)
        successes: list[bool] = []
        for train_indices, validation_indices in split_rows:
            model = fit_clinical_cox(
                frame.iloc[train_indices],
                time[train_indices],
                event[train_indices],
                include_icr=include_icr,
                include_treatment=include_treatment,
                l2=float(l2),
            )
            train_risk = model.predict(frame.iloc[train_indices])
            validation_risk = model.predict(frame.iloc[validation_indices])
            oof[validation_indices] = _calibrated_predictions(
                train_risk,
                validation_risk,
            )
            successes.append(model.model.optimization_success)
        score = float(concordance_index(time, event, oof))
        rows.append(
            {
                "l2": float(l2),
                "c_index": score,
                "oof_risk": oof,
                "all_optimizations_succeeded": bool(all(successes)),
            }
        )
    best = max(rows, key=lambda row: (row["c_index"], row["l2"]))
    return {
        "config": {"l2": float(best["l2"])},
        "c_index": float(best["c_index"]),
        "oof_risk": np.asarray(best["oof_risk"], dtype=float),
        "candidate_scores": [
            {
                key: value
                for key, value in row.items()
                if key != "oof_risk"
            }
            for row in rows
        ],
    }


def _microbiome_inner_selection(
    cohort: V8Cohort,
    time: np.ndarray,
    event: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_rows = list(splits)
    for config in MICROBIOME_CONFIGS:
        oof = np.full(len(time), np.nan, dtype=float)
        successes: list[bool] = []
        selected_features: list[list[str]] = []
        for train_indices, validation_indices in split_rows:
            model = fit_microbiome_cox(
                cohort.tumor[train_indices],
                cohort.normal[train_indices],
                cohort.genera,
                time[train_indices],
                event[train_indices],
                **config,
            )
            train_risk = model.predict(
                cohort.tumor[train_indices],
                cohort.normal[train_indices],
            )
            validation_risk = model.predict(
                cohort.tumor[validation_indices],
                cohort.normal[validation_indices],
            )
            oof[validation_indices] = _calibrated_predictions(
                train_risk,
                validation_risk,
            )
            successes.append(model.model.optimization_success)
            selected_features.append(list(model.transformer.feature_names))
        score = float(concordance_index(time, event, oof))
        rows.append(
            {
                **config,
                "c_index": score,
                "oof_risk": oof,
                "all_optimizations_succeeded": bool(all(successes)),
                "selected_features": selected_features,
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["c_index"],
            row["l2"],
            -row["top_k"],
            row["prevalence_threshold"],
        ),
    )
    return {
        "config": {
            "prevalence_threshold": float(best["prevalence_threshold"]),
            "top_k": int(best["top_k"]),
            "l2": float(best["l2"]),
        },
        "c_index": float(best["c_index"]),
        "oof_risk": np.asarray(best["oof_risk"], dtype=float),
        "candidate_scores": [
            {
                key: value
                for key, value in row.items()
                if key not in {"oof_risk", "selected_features"}
            }
            for row in rows
        ],
    }


def _select_blend(
    clinical_risk: np.ndarray,
    microbiome_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for alpha in BLEND_ALPHAS:
        risk = (
            (1.0 - float(alpha)) * np.asarray(clinical_risk, dtype=float)
            + float(alpha) * np.asarray(microbiome_risk, dtype=float)
        )
        rows.append(
            {
                "alpha": float(alpha),
                "c_index": float(concordance_index(time, event, risk)),
            }
        )
    best = max(rows, key=lambda row: (row["c_index"], -row["alpha"]))
    return {
        "alpha": float(best["alpha"]),
        "c_index": float(best["c_index"]),
        "candidate_scores": rows,
    }


def _fold_auc(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risks: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model_name, risk in risks.items():
        output[model_name] = {}
        for horizon in HORIZONS:
            try:
                output[model_name][str(int(horizon))] = (
                    cumulative_dynamic_auc(
                        train_time=train_time,
                        train_event=train_event,
                        test_time=test_time,
                        test_event=test_event,
                        risk=risk,
                        horizon=horizon,
                    )
                )
            except ValueError as error:
                output[model_name][str(int(horizon))] = {
                    "horizon": float(horizon),
                    "available": False,
                    "reason": str(error),
                }
    return output


def _run_outer_seed(
    cohort: V8Cohort,
    *,
    endpoint: str,
    seed: int,
) -> dict[str, Any]:
    endpoint_prefix = endpoint.lower()
    time = cohort.patients[f"{endpoint_prefix}_time"].to_numpy(float)
    event = cohort.patients[f"{endpoint_prefix}_event"].to_numpy(float)
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(seed),
    )
    fold_splits = list(
        splitter.split(
            np.arange(len(time)),
            _strata(time, event, n_splits=5),
        )
    )
    model_names = (
        "clinical_core",
        "clinical_treatment_sensitivity",
        "clinical_icr",
        "microbiome_internal_relation",
        "safe_blend_core",
        "safe_blend_icr",
        "published_mbr_reference",
    )
    oof = {
        model_name: np.full(len(time), np.nan, dtype=float)
        for model_name in model_names
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        fold_splits,
        start=1,
    ):
        train = V8Cohort(
            patients=cohort.patients.iloc[train_indices].reset_index(drop=True),
            tumor=cohort.tumor[train_indices],
            normal=cohort.normal[train_indices],
            genera=cohort.genera,
            quality_report=cohort.quality_report,
        )
        test = V8Cohort(
            patients=cohort.patients.iloc[test_indices].reset_index(drop=True),
            tumor=cohort.tumor[test_indices],
            normal=cohort.normal[test_indices],
            genera=cohort.genera,
            quality_report=cohort.quality_report,
        )
        train_time = time[train_indices]
        train_event = event[train_indices]
        test_time = time[test_indices]
        test_event = event[test_indices]
        inner = _inner_splits(
            train_time,
            train_event,
            seed=int(seed * 100 + fold_number),
        )
        clinical_core_selection = _clinical_inner_selection(
            train.patients,
            train_time,
            train_event,
            inner,
            include_icr=False,
            include_treatment=False,
        )
        clinical_treatment_selection = _clinical_inner_selection(
            train.patients,
            train_time,
            train_event,
            inner,
            include_icr=False,
            include_treatment=True,
        )
        clinical_icr_selection = _clinical_inner_selection(
            train.patients,
            train_time,
            train_event,
            inner,
            include_icr=True,
            include_treatment=False,
        )
        microbiome_selection = _microbiome_inner_selection(
            train,
            train_time,
            train_event,
            inner,
        )
        core_blend = _select_blend(
            clinical_core_selection["oof_risk"],
            microbiome_selection["oof_risk"],
            train_time,
            train_event,
        )
        icr_blend = _select_blend(
            clinical_icr_selection["oof_risk"],
            microbiome_selection["oof_risk"],
            train_time,
            train_event,
        )

        clinical_core = fit_clinical_cox(
            train.patients,
            train_time,
            train_event,
            include_icr=False,
            include_treatment=False,
            **clinical_core_selection["config"],
        )
        clinical_treatment = fit_clinical_cox(
            train.patients,
            train_time,
            train_event,
            include_icr=False,
            include_treatment=True,
            **clinical_treatment_selection["config"],
        )
        clinical_icr = fit_clinical_cox(
            train.patients,
            train_time,
            train_event,
            include_icr=True,
            include_treatment=False,
            **clinical_icr_selection["config"],
        )
        microbiome = fit_microbiome_cox(
            train.tumor,
            train.normal,
            train.genera,
            train_time,
            train_event,
            **microbiome_selection["config"],
        )
        core_test = _calibrated_predictions(
            clinical_core.predict(train.patients),
            clinical_core.predict(test.patients),
        )
        treatment_test = _calibrated_predictions(
            clinical_treatment.predict(train.patients),
            clinical_treatment.predict(test.patients),
        )
        icr_test = _calibrated_predictions(
            clinical_icr.predict(train.patients),
            clinical_icr.predict(test.patients),
        )
        microbiome_test = _calibrated_predictions(
            microbiome.predict(train.tumor, train.normal),
            microbiome.predict(test.tumor, test.normal),
        )
        fold_risks = {
            "clinical_core": core_test,
            "clinical_treatment_sensitivity": treatment_test,
            "clinical_icr": icr_test,
            "microbiome_internal_relation": microbiome_test,
            "safe_blend_core": (
                (1.0 - core_blend["alpha"]) * core_test
                + core_blend["alpha"] * microbiome_test
            ),
            "safe_blend_icr": (
                (1.0 - icr_blend["alpha"]) * icr_test
                + icr_blend["alpha"] * microbiome_test
            ),
            "published_mbr_reference": test.patients[
                "published_mbr_score"
            ].to_numpy(float),
        }
        for model_name, risk in fold_risks.items():
            oof[model_name][test_indices] = risk
        fold_rows.append(
            {
                "fold": int(fold_number),
                "train_patients": int(len(train_indices)),
                "train_events": int(train_event.sum()),
                "test_patients": int(len(test_indices)),
                "test_events": int(test_event.sum()),
                "selected": {
                    "clinical_core": clinical_core_selection["config"],
                    "clinical_treatment_sensitivity": (
                        clinical_treatment_selection["config"]
                    ),
                    "clinical_icr": clinical_icr_selection["config"],
                    "microbiome_internal_relation": {
                        **microbiome_selection["config"],
                        "features": list(
                            microbiome.transformer.feature_names
                        ),
                    },
                    "safe_blend_core_alpha": core_blend["alpha"],
                    "safe_blend_icr_alpha": icr_blend["alpha"],
                },
                "inner_c_index": {
                    "clinical_core": clinical_core_selection["c_index"],
                    "clinical_treatment_sensitivity": (
                        clinical_treatment_selection["c_index"]
                    ),
                    "clinical_icr": clinical_icr_selection["c_index"],
                    "microbiome_internal_relation": (
                        microbiome_selection["c_index"]
                    ),
                    "safe_blend_core": core_blend["c_index"],
                    "safe_blend_icr": icr_blend["c_index"],
                },
                "test_c_index": {
                    model_name: float(
                        concordance_index(test_time, test_event, risk)
                    )
                    for model_name, risk in fold_risks.items()
                },
                "test_auc": _fold_auc(
                    train_time=train_time,
                    train_event=train_event,
                    test_time=test_time,
                    test_event=test_event,
                    risks=fold_risks,
                ),
            }
        )
        print(
            f"seed={seed} fold={fold_number} "
            f"core={fold_rows[-1]['test_c_index']['clinical_core']:.4f} "
            f"blend={fold_rows[-1]['test_c_index']['safe_blend_core']:.4f} "
            f"blend_icr={fold_rows[-1]['test_c_index']['safe_blend_icr']:.4f}"
        )

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError(f"Seed {seed} produced incomplete OOF predictions.")
    pooled_c = {
        model_name: float(concordance_index(time, event, risk))
        for model_name, risk in oof.items()
    }
    mean_fold_auc: dict[str, dict[str, float | None]] = {}
    for model_name in model_names:
        mean_fold_auc[model_name] = {}
        for horizon in HORIZONS:
            values = [
                row["test_auc"][model_name][str(int(horizon))].get("auc")
                for row in fold_rows
                if row["test_auc"][model_name][str(int(horizon))].get(
                    "available", True
                )
            ]
            mean_fold_auc[model_name][str(int(horizon))] = (
                float(np.mean(values)) if values else None
            )
    return {
        "seed": int(seed),
        "pooled_oof_c_index": pooled_c,
        "mean_fold_auc": mean_fold_auc,
        "folds": fold_rows,
        "oof": oof,
    }


def _fast_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    risk_values = np.asarray(risk, dtype=float)
    first, second = np.triu_indices(len(time_values), k=1)
    first_earlier = (
        (time_values[first] < time_values[second])
        & (event_values[first] > 0.5)
    )
    second_earlier = (
        (time_values[second] < time_values[first])
        & (event_values[second] > 0.5)
    )
    permissible = first_earlier | second_earlier
    if not permissible.any():
        return 0.0
    earlier_risk = np.where(
        first_earlier,
        risk_values[first],
        risk_values[second],
    )[permissible]
    later_risk = np.where(
        first_earlier,
        risk_values[second],
        risk_values[first],
    )[permissible]
    return float(
        np.mean(
            (earlier_risk > later_risk).astype(float)
            + 0.5 * (earlier_risk == later_risk)
        )
    )


def _bootstrap_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    iterations: int,
    seed: int = 2026,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(iterations)):
        indices = rng.integers(0, len(time), size=len(time))
        value = _fast_c_index(
            time[indices],
            event[indices],
            risk[indices],
        )
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise RuntimeError("Bootstrap produced no valid C-index values.")
    array = np.asarray(values, dtype=float)
    return {
        "iterations_requested": int(iterations),
        "iterations_valid": int(len(array)),
        "estimate": float(_fast_c_index(time, event, risk)),
        "ci_95": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
    }


def _ensemble_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        output[str(int(horizon))] = cumulative_dynamic_auc(
            train_time=time,
            train_event=event,
            test_time=time,
            test_event=event,
            risk=risk,
            horizon=horizon,
        )
    return output


def _external_mbr_reference(
    *,
    endpoint: str,
    scope: str,
) -> dict[str, Any]:
    clinical = pd.read_csv(
        CLINICAL_PATH,
        sep="\t",
        skiprows=4,
        dtype=str,
    )
    clinical = clinical.loc[
        clinical["MICROBIOME_COHORT"].eq("ICAM42")
    ].copy()
    stage = pd.to_numeric(clinical["AJCC_PATH_STAGE"], errors="coerce")
    if scope == "stage_i_iii":
        mask = stage.isin([1.0, 2.0, 3.0])
    else:
        mask = stage.isin([1.0, 2.0, 3.0, 4.0])
    time = pd.to_numeric(
        clinical[f"{endpoint.upper()}_MONTHS"],
        errors="coerce",
    )
    event = (
        clinical[f"{endpoint.upper()}_STATUS"]
        .fillna("")
        .str.startswith("1:")
        .astype(float)
    )
    risk = pd.to_numeric(clinical["MBR_SCORE"], errors="coerce")
    complete = mask & time.gt(0.0) & time.notna() & risk.notna()
    return {
        "cohort": "ICAM42",
        "raw_16s_not_available_in_current_public_download": True,
        "patients": int(complete.sum()),
        "events": int(event[complete].sum()),
        "published_mbr_c_index": float(
            concordance_index(
                time[complete],
                event[complete],
                risk[complete],
            )
        ),
        "interpretation": (
            "This is the published frozen MBR score in the independent "
            "ICAM42 validation subset. The V8 full-genus model cannot be "
            "evaluated there because its raw 16S table is not in the "
            "current unrestricted Figshare record."
        ),
    }


def run_benchmark(
    *,
    endpoint: str,
    scope: str,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    bootstrap_iterations: int = 1000,
    processed_dir: Path = PROCESSED_ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    endpoint = endpoint.upper()
    cohort = load_v8_cohort(processed_dir=processed_dir).subset(
        endpoint=endpoint,
        scope=scope,
    )
    time = cohort.patients[f"{endpoint.lower()}_time"].to_numpy(float)
    event = cohort.patients[f"{endpoint.lower()}_event"].to_numpy(float)
    seed_values = tuple(int(seed) for seed in seeds)
    runs = [
        _run_outer_seed(cohort, endpoint=endpoint, seed=seed)
        for seed in seed_values
    ]
    model_names = tuple(runs[0]["oof"].keys())
    ensemble_risk = {
        model_name: np.mean(
            np.column_stack(
                [run["oof"][model_name] for run in runs]
            ),
            axis=1,
        )
        for model_name in model_names
    }
    model_summary: dict[str, Any] = {}
    for model_name in model_names:
        seed_c = [
            run["pooled_oof_c_index"][model_name]
            for run in runs
        ]
        model_summary[model_name] = {
            "seed_c_index": [float(value) for value in seed_c],
            "mean_seed_c_index": float(np.mean(seed_c)),
            "std_seed_c_index": (
                float(np.std(seed_c, ddof=1)) if len(seed_c) > 1 else 0.0
            ),
            "ensemble_oof_c_index": float(
                concordance_index(
                    time,
                    event,
                    ensemble_risk[model_name],
                )
            ),
            "ensemble_oof_c_index_bootstrap": _bootstrap_c_index(
                time,
                event,
                ensemble_risk[model_name],
                iterations=bootstrap_iterations,
            ),
            "ensemble_oof_auc": _ensemble_auc(
                time,
                event,
                ensemble_risk[model_name],
            ),
            "mean_fold_auc_by_seed": [
                run["mean_fold_auc"][model_name] for run in runs
            ],
        }

    feature_counts: Counter[str] = Counter()
    blend_core_alphas: list[float] = []
    blend_icr_alphas: list[float] = []
    for run in runs:
        for fold in run["folds"]:
            feature_counts.update(
                fold["selected"]["microbiome_internal_relation"]["features"]
            )
            blend_core_alphas.append(
                float(fold["selected"]["safe_blend_core_alpha"])
            )
            blend_icr_alphas.append(
                float(fold["selected"]["safe_blend_icr_alpha"])
            )
    threshold = {
        model_name: {
            "target": TARGET_C_INDEX,
            "ensemble_oof_c_index": (
                model_summary[model_name]["ensemble_oof_c_index"]
            ),
            "exceeds_target": bool(
                model_summary[model_name]["ensemble_oof_c_index"]
                > TARGET_C_INDEX
            ),
        }
        for model_name in (
            "clinical_core",
            "safe_blend_core",
            "clinical_icr",
            "safe_blend_icr",
        )
    }
    result = {
        "experiment": "ac_icam_real_outcome_v8",
        "endpoint": endpoint,
        "scope": scope,
        "patients": int(len(time)),
        "events": int(event.sum()),
        "seeds": list(seed_values),
        "models": model_summary,
        "target_audit": threshold,
        "microbiome_stability": {
            "outer_models": int(len(feature_counts) > 0)
            and int(len(seed_values) * 5),
            "top_selected_features": [
                {"feature": feature, "count": int(count)}
                for feature, count in feature_counts.most_common(30)
            ],
            "safe_blend_core_alpha_mean": float(
                np.mean(blend_core_alphas)
            ),
            "safe_blend_core_zero_fraction": float(
                np.mean(np.asarray(blend_core_alphas) == 0.0)
            ),
            "safe_blend_icr_alpha_mean": float(
                np.mean(blend_icr_alphas)
            ),
            "safe_blend_icr_zero_fraction": float(
                np.mean(np.asarray(blend_icr_alphas) == 0.0)
            ),
        },
        "external_reference": _external_mbr_reference(
            endpoint=endpoint,
            scope=scope,
        ),
        "reference_boundaries": {
            "published_mbr_reference": (
                "Historical same-cohort outcome-selected reference; not an "
                "unbiased V8 candidate despite being shown numerically."
            ),
            "mrs_16s_2025": (
                "Panel retained in published_panels.json, but excluded from "
                "candidate training because its genera were selected in the "
                "same 209 AC-ICAM patients."
            ),
            "clinical_treatment_sensitivity": (
                "Adjuvant treatment is a post-surgical treatment field and "
                "is reported only as a sensitivity analysis."
            ),
        },
        "comparability_warning": (
            "V8 uses real colon-cancer outcomes and tissue microbiome. Its "
            "C-index must not be treated as a direct continuation of the "
            "synthetic/noisy V7 benchmark."
        ),
        "runs": [
            {key: value for key, value in run.items() if key != "oof"}
            for run in runs
        ],
    }
    destination = output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{scope}_{endpoint.lower()}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "benchmark_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "patient_id": cohort.patients["patient_id"],
            "time": time,
            "event": event,
            **{
                f"{model_name}_risk": risk
                for model_name, risk in ensemble_risk.items()
            },
        }
    )
    predictions.to_csv(destination / "ensemble_oof_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent AC-ICAM V8 real-outcome benchmark."
    )
    parser.add_argument("--endpoint", choices=("PFS", "OS"), default="PFS")
    parser.add_argument(
        "--scope",
        choices=("all_stage", "stage_i_iii"),
        default="all_stage",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_benchmark(
        endpoint=args.endpoint,
        scope=args.scope,
        seeds=args.seeds,
        bootstrap_iterations=args.bootstrap_iterations,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "endpoint": result["endpoint"],
                "scope": result["scope"],
                "patients": result["patients"],
                "events": result["events"],
                "target_audit": result["target_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
