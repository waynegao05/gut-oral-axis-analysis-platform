from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from experiments.topology_v7_ac_icam_transfer_v10.ac_icam import (
    PROCESSED_ROOT,
    load_ac_icam_cohort,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "outputs/topology_v7_ac_icam_transfer_v10/mbr_distillation"
)
CV_SEEDS = (7, 21, 42, 123, 2026)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
MINIMUM_MEAN_SPEARMAN = 0.35
MINIMUM_SEED_SPEARMAN = 0.30
MINIMUM_MEAN_R2 = 0.08


@dataclass(frozen=True)
class DistilledRiskModel:
    feature_columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
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
        return (
            standardized @ np.asarray(self.coefficients, dtype=float)
            + self.intercept
        )


def fit_distilled_risk(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
) -> DistilledRiskModel:
    values = (
        frame[feature_columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    mean = values.mean(axis=0).fillna(0.0)
    values = values.fillna(mean)
    scale = values.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    standardized = ((values - mean) / scale).to_numpy(float)
    model = Ridge(alpha=float(alpha))
    model.fit(standardized, frame["mbr_score"].astype(float).to_numpy())
    return DistilledRiskModel(
        feature_columns=tuple(feature_columns),
        mean=tuple(float(value) for value in mean.to_numpy()),
        scale=tuple(float(value) for value in scale.to_numpy()),
        coefficients=tuple(float(value) for value in model.coef_),
        intercept=float(model.intercept_),
        alpha=float(alpha),
    )


def _source_cv(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    strata = pd.qcut(
        frame["mbr_score"].rank(method="first"),
        q=5,
        labels=False,
        duplicates="drop",
    ).astype(int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_indices, test_indices in folds.split(frame, strata):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        model = fit_distilled_risk(
            train,
            feature_columns,
            alpha=alpha,
        )
        predictions[test_indices] = model.predict(test)
    observed = frame["mbr_score"].astype(float).to_numpy()
    return {
        "seed": int(seed),
        "spearman": float(spearmanr(observed, predictions).statistic),
        "r2": float(r2_score(observed, predictions)),
    }


def run_mbr_distillation(
    *,
    processed_dir: Path = PROCESSED_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    frame, cohort_report = load_ac_icam_cohort(
        output_dir=processed_dir,
        endpoint="PFS",
    )
    frame = frame.loc[frame["mbr_score"].notna()].reset_index(drop=True)
    feature_sets = cohort_report["feature_columns"]
    candidates: list[dict[str, Any]] = []
    for feature_set, feature_columns in feature_sets.items():
        for alpha in RIDGE_ALPHAS:
            seed_results = [
                _source_cv(
                    frame,
                    feature_columns,
                    alpha=alpha,
                    seed=seed,
                )
                for seed in CV_SEEDS
            ]
            spearman = np.asarray(
                [row["spearman"] for row in seed_results], dtype=float
            )
            r2 = np.asarray([row["r2"] for row in seed_results], dtype=float)
            row = {
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "alpha": float(alpha),
                "seed_results": seed_results,
                "mean_spearman": float(spearman.mean()),
                "std_spearman": float(spearman.std(ddof=1)),
                "minimum_spearman": float(spearman.min()),
                "mean_r2": float(r2.mean()),
                "minimum_r2": float(r2.min()),
            }
            candidates.append(row)
            print(
                f"{feature_set}/alpha={alpha:g} "
                f"rho={row['mean_spearman']:.4f} "
                f"R2={row['mean_r2']:.4f}",
                flush=True,
            )
    candidates.sort(
        key=lambda row: (
            row["mean_spearman"],
            row["mean_r2"],
            row["minimum_spearman"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    checks = {
        "mean_spearman_at_least_0_35": bool(
            selected["mean_spearman"] >= MINIMUM_MEAN_SPEARMAN
        ),
        "minimum_seed_spearman_at_least_0_30": bool(
            selected["minimum_spearman"] >= MINIMUM_SEED_SPEARMAN
        ),
        "mean_r2_at_least_0_08": bool(
            selected["mean_r2"] >= MINIMUM_MEAN_R2
        ),
    }
    gate_passed = all(checks.values())
    final_model = fit_distilled_risk(
        frame,
        selected["feature_columns"],
        alpha=float(selected["alpha"]),
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "public_source_mbr_distillation_only",
        "v7_formal_cohort_used": False,
        "cohort": cohort_report,
        "num_complete_mbr_patients": int(len(frame)),
        "cv_seeds": list(CV_SEEDS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "candidates": candidates,
        "selected_candidate": selected,
        "distillation_gate": {
            "passed": gate_passed,
            "checks": checks,
            "decision": (
                "allow_as_v7_development_prior"
                if gate_passed
                else "reject_before_v7_development"
            ),
        },
        "final_distilled_model": asdict(final_model),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mbr_distillation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if gate_passed:
        (output_dir / "mbr_distilled_prior.json").write_text(
            json.dumps(asdict(final_model), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill the published AC-ICAM MBR score to V7 taxa."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_mbr_distillation(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "distillation_gate": result["distillation_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
