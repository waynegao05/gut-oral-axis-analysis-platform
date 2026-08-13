from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from experiments.topology_v7_ac_icam_transfer_v10.ac_icam import (
    PROCESSED_ROOT,
    load_ac_icam_cohort,
)
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "outputs/topology_v7_ac_icam_transfer_v10/source_audit"
)
CV_SEEDS = (7, 21, 42, 123, 2026)
L2_VALUES = (0.01, 0.1, 1.0)
MINIMUM_MEAN_SOURCE_C_INDEX = 0.56
MINIMUM_SEED_SOURCE_C_INDEX = 0.53


@dataclass(frozen=True)
class RidgeCoxModel:
    feature_columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    l2: float
    optimization_success: bool

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        values = (
            frame[list(self.feature_columns)]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy()
        )
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(mean, values.shape)[missing]
        standardized = (values - mean) / scale
        return np.asarray(standardized @ coefficients, dtype=float)


def _cox_loss_gradient(
    coefficients: np.ndarray,
    values: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    margin = values @ coefficients
    event_times = np.unique(time[event > 0.5])
    loss = 0.0
    gradient = np.zeros_like(coefficients)
    num_events = float(event.sum())
    if num_events <= 0:
        raise ValueError("Cox training requires at least one observed event.")

    for current_time in event_times:
        observed = (time == current_time) & (event > 0.5)
        risk_set = time >= current_time
        risk_margin = margin[risk_set]
        shift = float(np.max(risk_margin))
        weights = np.exp(risk_margin - shift)
        denominator = float(weights.sum())
        log_denominator = shift + np.log(denominator)
        observed_count = int(observed.sum())
        loss -= float(margin[observed].sum())
        loss += observed_count * log_denominator
        weighted_mean = (
            values[risk_set] * weights[:, None]
        ).sum(axis=0) / denominator
        gradient -= values[observed].sum(axis=0)
        gradient += observed_count * weighted_mean

    loss = loss / num_events + 0.5 * l2 * float(
        coefficients @ coefficients
    )
    gradient = gradient / num_events + l2 * coefficients
    return float(loss), np.asarray(gradient, dtype=float)


def fit_ridge_cox(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    l2: float,
) -> RidgeCoxModel:
    values = (
        frame[feature_columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    mean = values.mean(axis=0).fillna(0.0)
    values = values.fillna(mean)
    scale = values.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    standardized = ((values - mean) / scale).to_numpy(float)
    time = frame["time"].astype(float).to_numpy()
    event = frame["event"].astype(float).to_numpy()

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return _cox_loss_gradient(
            beta, standardized, time, event, float(l2)
        )

    result = minimize(
        objective,
        np.zeros(len(feature_columns), dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    return RidgeCoxModel(
        feature_columns=tuple(feature_columns),
        mean=tuple(float(value) for value in mean.to_numpy()),
        scale=tuple(float(value) for value in scale.to_numpy()),
        coefficients=tuple(float(value) for value in result.x),
        l2=float(l2),
        optimization_success=bool(result.success),
    )


def _strata(frame: pd.DataFrame) -> np.ndarray:
    time_bin = pd.qcut(
        frame["time"].rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    )
    combined = (
        frame["event"].astype(int).astype(str)
        + "_"
        + time_bin.astype(int).astype(str)
    )
    if combined.value_counts().min() < 5:
        return frame["event"].astype(int).to_numpy()
    return combined.to_numpy()


def _cross_validated_score(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    optimizations: list[bool] = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    strata = _strata(frame)
    for train_indices, test_indices in folds.split(frame, strata):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        model = fit_ridge_cox(train, feature_columns, l2=l2)
        predictions[test_indices] = model.predict(test)
        optimizations.append(model.optimization_success)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Source cross-validation produced invalid risks.")
    return {
        "seed": int(seed),
        "c_index": float(
            concordance_index(
                frame["time"],
                frame["event"],
                predictions,
            )
        ),
        "all_optimizations_succeeded": bool(all(optimizations)),
    }


def _mbr_reference(frame: pd.DataFrame) -> dict[str, Any]:
    complete = frame.loc[frame["mbr_score"].notna()].copy()
    if complete.empty:
        return {"available": False}
    return {
        "available": True,
        "num_patients": int(len(complete)),
        "c_index": float(
            concordance_index(
                complete["time"],
                complete["event"],
                complete["mbr_score"],
            )
        ),
    }


def run_source_audit(
    *,
    processed_dir: Path = PROCESSED_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    frame, cohort_report = load_ac_icam_cohort(
        output_dir=processed_dir,
        endpoint="PFS",
    )
    feature_sets = cohort_report["feature_columns"]
    candidates: list[dict[str, Any]] = []
    for feature_set, feature_columns in feature_sets.items():
        for l2 in L2_VALUES:
            seed_results = [
                _cross_validated_score(
                    frame,
                    feature_columns,
                    l2=l2,
                    seed=seed,
                )
                for seed in CV_SEEDS
            ]
            scores = np.asarray(
                [row["c_index"] for row in seed_results], dtype=float
            )
            row = {
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "l2": float(l2),
                "seed_results": seed_results,
                "mean_c_index": float(scores.mean()),
                "std_c_index": float(scores.std(ddof=1)),
                "minimum_c_index": float(scores.min()),
                "maximum_c_index": float(scores.max()),
            }
            candidates.append(row)
            print(
                f"{feature_set}/l2={l2:g} "
                f"C={row['mean_c_index']:.4f} "
                f"min={row['minimum_c_index']:.4f}",
                flush=True,
            )

    candidates.sort(
        key=lambda row: (row["mean_c_index"], row["minimum_c_index"]),
        reverse=True,
    )
    selected = candidates[0]
    checks = {
        "mean_c_index_at_least_0_56": bool(
            selected["mean_c_index"] >= MINIMUM_MEAN_SOURCE_C_INDEX
        ),
        "minimum_seed_c_index_at_least_0_53": bool(
            selected["minimum_c_index"] >= MINIMUM_SEED_SOURCE_C_INDEX
        ),
        "all_optimizations_succeeded": bool(
            all(
                seed_result["all_optimizations_succeeded"]
                for seed_result in selected["seed_results"]
            )
        ),
    }
    gate_passed = all(checks.values())
    final_model = fit_ridge_cox(
        frame,
        selected["feature_columns"],
        l2=float(selected["l2"]),
    )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "public_source_only",
        "v7_formal_cohort_used": False,
        "cohort": cohort_report,
        "cv_seeds": list(CV_SEEDS),
        "candidate_l2_values": list(L2_VALUES),
        "candidates": candidates,
        "selected_candidate": selected,
        "published_mbr_reference": _mbr_reference(frame),
        "source_gate": {
            "passed": gate_passed,
            "checks": checks,
            "decision": (
                "allow_v7_development_screen"
                if gate_passed
                else "reject_before_v7_development"
            ),
        },
        "final_source_model": asdict(final_model),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if gate_passed:
        (output_dir / "source_prior.json").write_text(
            json.dumps(asdict(final_model), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the five-genus AC-ICAM survival prior."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_source_audit(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "cohort": result["cohort"],
                "selected_candidate": result["selected_candidate"],
                "source_gate": result["source_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
