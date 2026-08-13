from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.ac_icam_real_outcome_v8.benchmark import (
    CLINICAL_L2_VALUES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    _calibrated_predictions,
    _strata,
)
from experiments.ac_icam_real_outcome_v8.data import load_v8_cohort
from experiments.ac_icam_real_outcome_v8.modeling import (
    RiskCalibration,
    fit_clinical_cox,
)
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = (
    ROOT / "config" / "releases" / "ac_icam_real_outcome_pfs_v8.json"
)
HORIZONS = (36.0, 60.0)


def _select_l2(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    *,
    include_icr: bool,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(seed),
    )
    splits = list(
        splitter.split(
            np.arange(len(time)),
            _strata(time, event, n_splits=5),
        )
    )
    candidates: list[dict[str, float]] = []
    for l2 in CLINICAL_L2_VALUES:
        oof = np.full(len(time), np.nan, dtype=float)
        for train_indices, validation_indices in splits:
            model = fit_clinical_cox(
                frame.iloc[train_indices],
                time[train_indices],
                event[train_indices],
                include_icr=include_icr,
                include_treatment=False,
                l2=float(l2),
            )
            oof[validation_indices] = _calibrated_predictions(
                model.predict(frame.iloc[train_indices]),
                model.predict(frame.iloc[validation_indices]),
            )
        candidates.append(
            {
                "l2": float(l2),
                "oof_c_index": float(concordance_index(time, event, oof)),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (row["oof_c_index"], row["l2"]),
    )
    return {
        "l2": float(selected["l2"]),
        "oof_c_index": float(selected["oof_c_index"]),
        "candidate_scores": candidates,
    }


def _breslow_cumulative_hazard(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    horizons: tuple[float, ...],
) -> dict[str, float]:
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    risk_values = np.asarray(risk, dtype=float)
    exp_risk = np.exp(np.clip(risk_values, -30.0, 30.0))
    event_times = np.unique(time_values[event_values > 0.5])
    increments: list[tuple[float, float]] = []
    for event_time in event_times:
        observed = (time_values == event_time) & (event_values > 0.5)
        risk_set = time_values >= event_time
        denominator = float(exp_risk[risk_set].sum())
        if denominator <= 0.0:
            raise RuntimeError("Invalid Breslow risk-set denominator.")
        increments.append(
            (
                float(event_time),
                float(observed.sum()) / denominator,
            )
        )
    return {
        str(int(horizon)): float(
            sum(
                increment
                for event_time, increment in increments
                if event_time <= horizon
            )
        )
        for horizon in horizons
    }


def _serialize_member(
    model: Any,
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
) -> dict[str, Any]:
    raw_risk = model.predict(frame)
    calibration = RiskCalibration.fit(raw_risk)
    transformer = model.transformer
    return {
        "l2": float(model.model.l2),
        "optimization_success": bool(model.model.optimization_success),
        "transformer": {
            "numeric_columns": list(transformer.numeric_columns),
            "categorical_columns": list(transformer.categorical_columns),
            "numeric_medians": transformer.numeric_medians.tolist(),
            "category_levels": [
                list(levels) for levels in transformer.category_levels
            ],
            "feature_names": list(transformer.feature_names),
        },
        "cox_model": {
            "mean": model.model.mean.tolist(),
            "scale": model.model.scale.tolist(),
            "coefficients": model.model.coefficients.tolist(),
        },
        "risk_calibration": {
            "mean": float(calibration.mean),
            "scale": float(calibration.scale),
        },
        "breslow_cumulative_hazard": _breslow_cumulative_hazard(
            time,
            event,
            raw_risk,
            horizons=HORIZONS,
        ),
    }


def _fit_variant(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    *,
    include_icr: bool,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    members: list[dict[str, Any]] = []
    standardized_training_risks: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    for seed in seeds:
        selection = _select_l2(
            frame,
            time,
            event,
            include_icr=include_icr,
            seed=int(seed),
        )
        fitted = fit_clinical_cox(
            frame,
            time,
            event,
            include_icr=include_icr,
            include_treatment=False,
            l2=float(selection["l2"]),
        )
        member = _serialize_member(fitted, frame, time, event)
        member["selection_seed"] = int(seed)
        member["selection_oof_c_index"] = float(selection["oof_c_index"])
        member["selection_candidates"] = selection["candidate_scores"]
        members.append(member)
        raw = fitted.predict(frame)
        standardized_training_risks.append(
            RiskCalibration.fit(raw).transform(raw)
        )
        selected_rows.append(
            {
                "seed": int(seed),
                "l2": float(selection["l2"]),
                "oof_c_index": float(selection["oof_c_index"]),
            }
        )

    member_matrix = np.column_stack(standardized_training_risks)
    ensemble_training_risk = member_matrix.mean(axis=1)
    disagreement = member_matrix.std(axis=1)
    variant = {
        "include_icr": bool(include_icr),
        "members": members,
        "selection_summary": selected_rows,
        "deployment_risk_calibration": {
            "mean": float(ensemble_training_risk.mean()),
            "scale": max(float(ensemble_training_risk.std()), 1e-12),
        },
        "member_disagreement_p90": max(
            float(np.quantile(disagreement, 0.90)),
            0.10,
        ),
    }
    return variant, ensemble_training_risk, disagreement


def _metric_snapshot(summary: dict[str, Any], model_name: str) -> dict[str, Any]:
    model = summary["models"][model_name]
    return {
        "mean_seed_c_index": float(model["mean_seed_c_index"]),
        "std_seed_c_index": float(model["std_seed_c_index"]),
        "ensemble_oof_c_index": float(model["ensemble_oof_c_index"]),
        "ensemble_oof_c_index_bootstrap": model[
            "ensemble_oof_c_index_bootstrap"
        ],
        "ensemble_oof_auc": model["ensemble_oof_auc"],
    }


def build_deployment_artifact(
    *,
    output_path: Path = DEFAULT_ARTIFACT_PATH,
    benchmark_root: Path = DEFAULT_OUTPUT_ROOT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Path:
    cohort = load_v8_cohort().subset(endpoint="PFS", scope="all_stage")
    frame = cohort.patients
    time = frame["pfs_time"].to_numpy(float)
    event = frame["pfs_event"].to_numpy(float)

    benchmark_dir = benchmark_root / "all_stage_pfs"
    summary_path = benchmark_dir / "benchmark_summary.json"
    oof_path = benchmark_dir / "ensemble_oof_predictions.csv"
    if not summary_path.exists() or not oof_path.exists():
        raise FileNotFoundError(
            "Run the formal all-stage PFS benchmark before building the web artifact."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    oof = pd.read_csv(oof_path)
    variants: dict[str, Any] = {}
    for variant_name, include_icr, oof_column, metric_name in (
        ("clinical_core", False, "clinical_core_risk", "clinical_core"),
        ("clinical_icr", True, "clinical_icr_risk", "clinical_icr"),
    ):
        fitted, deployment_risk, _ = _fit_variant(
            frame,
            time,
            event,
            include_icr=include_icr,
            seeds=tuple(int(seed) for seed in seeds),
        )
        reference_risk = oof[oof_column].to_numpy(float)
        fitted["deployment_to_oof_calibration"] = {
            "source_mean": float(deployment_risk.mean()),
            "source_scale": max(float(deployment_risk.std()), 1e-12),
            "target_mean": float(reference_risk.mean()),
            "target_scale": max(float(reference_risk.std()), 1e-12),
        }
        fitted["reference_oof_risks"] = np.sort(reference_risk).tolist()
        fitted["risk_thresholds"] = {
            "low_upper": float(np.quantile(reference_risk, 1.0 / 3.0)),
            "medium_upper": float(np.quantile(reference_risk, 2.0 / 3.0)),
        }
        fitted["formal_metrics"] = _metric_snapshot(summary, metric_name)
        variants[variant_name] = fitted

    numeric_ranges = {}
    for column in ("age", "stage", "path_t", "path_n", "path_m", "icr_score"):
        values = frame[column].to_numpy(float)
        values = values[np.isfinite(values)]
        numeric_ranges[column] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    artifact = {
        "schema_version": 1,
        "release_name": "ac_icam_real_outcome_pfs_v8",
        "backend": "ac_icam_real_outcome_clinical_pfs",
        "endpoint": "PFS",
        "scope": "AJCC stage I-IV colorectal cancer",
        "training_cohort": {
            "name": "AC-ICAM paired-tissue cohort",
            "patients": int(len(frame)),
            "events": int(event.sum()),
            "genera_available": int(len(cohort.genera)),
            "follow_up_unit": "months",
            "source_article": "https://doi.org/10.1038/s41591-023-02324-5",
            "clinical_source": "https://www.cbioportal.org/study/summary?id=coad_silu_2022",
            "microbiome_source": "https://doi.org/10.6084/m9.figshare.16944775",
        },
        "required_clinical_fields": [
            "age",
            "sex",
            "stage",
            "path_t",
            "path_n",
            "tumor_location",
            "tumor_morphology",
        ],
        "optional_clinical_fields": ["path_m", "icr_score"],
        "numeric_training_ranges": numeric_ranges,
        "variants": variants,
        "deployment_policy": {
            "default_variant": "clinical_core",
            "icr_variant_rule": (
                "Use clinical_icr only when a measured tumor-RNA ICR score is supplied."
            ),
            "microbiome_used_for_risk": False,
            "treatment_used_for_risk": False,
            "reason": (
                "Five-seed evaluation found no PFS improvement from the microbiome "
                "safe blend or adjuvant-treatment sensitivity model."
            ),
        },
        "limitations": [
            "Internal repeated cross-validation, not external clinical validation.",
            "The cohort contains colorectal-cancer patients and is not a screening model.",
            "The model is research decision support and does not provide diagnosis or treatment advice.",
            "The ICR variant requires a measured tumor-RNA score and must not be inferred from routine web fields.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and serialize the AC-ICAM V8 web deployment artifact."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
    )
    args = parser.parse_args()
    path = build_deployment_artifact(output_path=args.output)
    print(path)
    print(f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
