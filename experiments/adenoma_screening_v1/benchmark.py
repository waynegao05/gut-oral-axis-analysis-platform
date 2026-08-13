from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    ROOT
    / "data"
    / "public"
    / "zeller_crc_adenoma_2014"
    / "processed"
    / "zeller_five_genus_screening.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "adenoma_screening_v1"
PROTOCOL_PATH = Path(__file__).with_name("protocol_lock.json")

FEATURE_COLUMNS = (
    "fusobacterium_abundance_pct",
    "porphyromonas_abundance_pct",
    "prevotella_abundance_pct",
    "streptococcus_abundance_pct",
    "lactobacillus_abundance_pct",
)
EXPECTED_COUNTS = {
    "healthy": 61,
    "small_adenoma": 27,
    "large_adenoma": 15,
    "crc": 53,
}
SEEDS = (7, 21, 42, 123, 2026)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
OPERATING_POINTS = {"specificity_90": 0.90, "specificity_95": 0.95}


def load_screening_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sample_id",
        "subject_id",
        "disease_group",
        "screening_label",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["sample_id"].duplicated().any() or frame["subject_id"].duplicated().any():
        raise ValueError("Exactly one unique sample per subject is required.")
    counts = frame["disease_group"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected disease group counts: {counts}")
    labels = frame["screening_label"].to_numpy(dtype=int)
    expected_labels = (frame["disease_group"] != "healthy").to_numpy(dtype=int)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("screening_label does not match the locked group definition.")
    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Abundances must be finite percentages in [0, 100].")
    return frame


def transform_abundance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Abundances must be finite percentages in [0, 100].")
    return np.log10(values / 100.0 + 1e-6)


def make_model(c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )


def inner_oof_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    c_value: float,
) -> np.ndarray:
    probabilities = np.full(y.shape[0], np.nan, dtype=float)
    for train_index, validation_index in splits:
        model = make_model(c_value)
        model.fit(x[train_index], y[train_index])
        probabilities[validation_index] = model.predict_proba(x[validation_index])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Inner OOF predictions are incomplete.")
    return probabilities


def select_regularization(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    candidates: list[tuple[float, float, np.ndarray]] = []
    records: list[dict[str, float]] = []
    for c_value in C_GRID:
        probabilities = inner_oof_probabilities(x, y, splits, c_value)
        auc = float(roc_auc_score(y, probabilities))
        candidates.append((auc, float(c_value), probabilities))
        records.append({"c": float(c_value), "inner_oof_auc": auc})
    best_auc, best_c, best_probabilities = sorted(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )[0]
    if not math.isfinite(best_auc):
        raise RuntimeError("No finite inner OOF AUC was produced.")
    return best_c, best_probabilities, records


def binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    negatives = y_true == 0
    positives = y_true == 1
    specificity = float(np.mean(y_pred[negatives] == 0))
    sensitivity = float(np.mean(y_pred[positives] == 1))
    return specificity, sensitivity


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_specificity: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not 0.0 < target_specificity < 1.0:
        raise ValueError("target_specificity must be in (0, 1).")
    thresholds = np.unique(
        np.concatenate(
            [
                probabilities,
                [np.nextafter(float(np.max(probabilities)), np.inf)],
            ]
        )
    )
    feasible: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        specificity, sensitivity = binary_rates(y_true, predictions)
        if specificity + 1e-12 >= target_specificity:
            feasible.append((sensitivity, specificity, float(threshold)))
    if not feasible:
        raise RuntimeError("No threshold satisfies the target specificity.")
    sensitivity, specificity, threshold = sorted(
        feasible,
        key=lambda item: (-item[0], item[1], item[2]),
    )[0]
    return {
        "threshold": threshold,
        "inner_oof_specificity": specificity,
        "inner_oof_sensitivity": sensitivity,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive.")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def rate_record(successes: int, total: int) -> dict[str, Any]:
    return {
        "value": successes / total,
        "numerator": int(successes),
        "denominator": int(total),
        "ci95_wilson": wilson_interval(successes, total),
    }


def bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 20260813,
) -> list[float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    negative_index = np.flatnonzero(y_true == 0)
    positive_index = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_negative = rng.choice(negative_index, size=negative_index.size, replace=True)
        sampled_positive = rng.choice(positive_index, size=positive_index.size, replace=True)
        sampled = np.concatenate([sampled_negative, sampled_positive])
        estimates[iteration] = roc_auc_score(y_true[sampled], scores[sampled])
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def auc_record(y_true: np.ndarray, scores: np.ndarray, *, seed: int) -> dict[str, Any]:
    return {
        "value": float(roc_auc_score(y_true, scores)),
        "ci95_bootstrap": bootstrap_auc(y_true, scores, seed=seed),
        "negative_n": int(np.sum(y_true == 0)),
        "positive_n": int(np.sum(y_true == 1)),
    }


def run_nested_oof(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = transform_abundance(frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float))
    y = frame["screening_label"].to_numpy(dtype=int)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seen = np.zeros(len(frame), dtype=int)
        for fold, (train_index, test_index) in enumerate(outer_cv.split(x, y), start=1):
            seen[test_index] += 1
            inner_cv = StratifiedKFold(
                n_splits=4,
                shuffle=True,
                random_state=seed * 100 + fold,
            )
            inner_splits = list(inner_cv.split(x[train_index], y[train_index]))
            best_c, inner_probabilities, c_records = select_regularization(
                x[train_index],
                y[train_index],
                inner_splits,
            )
            thresholds = {
                name: select_threshold(
                    y[train_index],
                    inner_probabilities,
                    target_specificity,
                )
                for name, target_specificity in OPERATING_POINTS.items()
            }
            model = make_model(best_c)
            model.fit(x[train_index], y[train_index])
            test_probabilities = model.predict_proba(x[test_index])[:, 1]

            fold_row: dict[str, Any] = {
                "seed": seed,
                "fold": fold,
                "train_n": int(train_index.size),
                "test_n": int(test_index.size),
                "selected_c": best_c,
                "inner_c_auc_json": json.dumps(c_records, separators=(",", ":")),
            }
            for name, threshold_data in thresholds.items():
                for key, value in threshold_data.items():
                    fold_row[f"{name}_{key}"] = value
            fold_rows.append(fold_row)

            for local_position, sample_index in enumerate(test_index):
                row: dict[str, Any] = {
                    "sample_id": frame.iloc[sample_index]["sample_id"],
                    "subject_id": frame.iloc[sample_index]["subject_id"],
                    "disease_group": frame.iloc[sample_index]["disease_group"],
                    "screening_label": int(y[sample_index]),
                    "seed": seed,
                    "fold": fold,
                    "probability": float(test_probabilities[local_position]),
                }
                for name, threshold_data in thresholds.items():
                    threshold = threshold_data["threshold"]
                    row[f"{name}_threshold"] = threshold
                    row[f"{name}_prediction"] = int(
                        test_probabilities[local_position] >= threshold
                    )
                prediction_rows.append(row)
        if not np.all(seen == 1):
            raise RuntimeError(f"Seed {seed} did not predict every sample exactly once.")

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    expected_rows = len(frame) * len(SEEDS)
    if len(predictions) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} predictions; got {len(predictions)}.")
    return predictions, folds


def consensus_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identity = ["sample_id", "subject_id", "disease_group", "screening_label"]
    grouped = predictions.groupby(identity, sort=False, as_index=False)
    consensus = grouped["probability"].mean().rename(
        columns={"probability": "mean_oof_probability"}
    )
    for name in OPERATING_POINTS:
        votes = grouped[f"{name}_prediction"].sum().rename(
            columns={f"{name}_prediction": f"{name}_positive_votes"}
        )
        consensus = consensus.merge(
            votes,
            on=identity,
            how="left",
            validate="one_to_one",
        )
        consensus[f"{name}_prediction"] = (
            consensus[f"{name}_positive_votes"] >= 3
        ).astype(int)
    if len(consensus) * len(SEEDS) != len(predictions):
        raise RuntimeError("Consensus aggregation lost or duplicated subjects.")
    return consensus


def operating_metrics(consensus: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    predicted = consensus[prediction_column].to_numpy(dtype=int)
    group = consensus["disease_group"].to_numpy(dtype=str)
    healthy = group == "healthy"
    lesion = ~healthy

    false_positive = int(np.sum(predicted[healthy] == 1))
    true_negative = int(np.sum(predicted[healthy] == 0))
    true_positive = int(np.sum(predicted[lesion] == 1))
    false_negative = int(np.sum(predicted[lesion] == 0))
    result: dict[str, Any] = {
        "false_positive_rate": rate_record(false_positive, int(np.sum(healthy))),
        "specificity": rate_record(true_negative, int(np.sum(healthy))),
        "any_neoplasia_sensitivity": rate_record(true_positive, int(np.sum(lesion))),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }
    for group_name in ("small_adenoma", "large_adenoma", "crc"):
        mask = group == group_name
        successes = int(np.sum(predicted[mask] == 1))
        result[f"{group_name}_sensitivity"] = rate_record(
            successes,
            int(np.sum(mask)),
        )
    return result


def seed_level_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed, seed_frame in predictions.groupby("seed", sort=True):
        record: dict[str, Any] = {
            "seed": int(seed),
            "auc": float(
                roc_auc_score(
                    seed_frame["screening_label"],
                    seed_frame["probability"],
                )
            ),
        }
        for name in OPERATING_POINTS:
            metrics = operating_metrics(seed_frame, f"{name}_prediction")
            record[f"{name}_fpr"] = metrics["false_positive_rate"]["value"]
            record[f"{name}_small_adenoma_sensitivity"] = metrics[
                "small_adenoma_sensitivity"
            ]["value"]
        records.append(record)
    return records


def build_metrics(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    consensus: pd.DataFrame,
) -> dict[str, Any]:
    y = consensus["screening_label"].to_numpy(dtype=int)
    scores = consensus["mean_oof_probability"].to_numpy(dtype=float)
    group = consensus["disease_group"].to_numpy(dtype=str)
    aucs: dict[str, Any] = {
        "any_neoplasia_vs_healthy": auc_record(y, scores, seed=20260813),
    }
    for offset, group_name in enumerate(("small_adenoma", "large_adenoma", "crc"), start=1):
        mask = (group == "healthy") | (group == group_name)
        comparison_y = (group[mask] == group_name).astype(int)
        aucs[f"{group_name}_vs_healthy"] = auc_record(
            comparison_y,
            scores[mask],
            seed=20260813 + offset,
        )

    operating = {
        name: {
            "inner_oof_target_specificity": target,
            **operating_metrics(consensus, f"{name}_prediction"),
        }
        for name, target in OPERATING_POINTS.items()
    }
    return {
        "protocol_id": "zeller_five_genus_screening_nested_oof_v1",
        "sample_count": int(len(frame)),
        "group_counts": {
            key: int(value)
            for key, value in frame["disease_group"].value_counts().to_dict().items()
        },
        "features": list(FEATURE_COLUMNS),
        "seeds": list(SEEDS),
        "auc": aucs,
        "operating_points": operating,
        "seed_level": seed_level_metrics(predictions),
        "reporting_boundary": (
            "Retrospective internal cross-validation of one public stool metagenomic "
            "cohort. This is not prospective clinical validation and does not validate "
            "the analytical performance of the colorimetric kit."
        ),
    }


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    consensus: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions_long.csv", index=False)
    folds.to_csv(output_dir / "outer_fold_records.csv", index=False)
    consensus.to_csv(output_dir / "oof_consensus_predictions.csv", index=False)
    (output_dir / "screening_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    summary_rows: list[dict[str, Any]] = []
    for comparison, auc_data in metrics["auc"].items():
        summary_rows.append(
            {
                "operating_point": "ranking",
                "metric": f"auc_{comparison}",
                "estimate": auc_data["value"],
                "ci95_lower": auc_data["ci95_bootstrap"][0],
                "ci95_upper": auc_data["ci95_bootstrap"][1],
                "numerator": "",
                "denominator": auc_data["negative_n"] + auc_data["positive_n"],
                "ci_method": "subject-level bootstrap",
            }
        )
    rate_names = (
        "false_positive_rate",
        "specificity",
        "any_neoplasia_sensitivity",
        "small_adenoma_sensitivity",
        "large_adenoma_sensitivity",
        "crc_sensitivity",
    )
    for operating_point, operating_data in metrics["operating_points"].items():
        for metric_name in rate_names:
            rate_data = operating_data[metric_name]
            summary_rows.append(
                {
                    "operating_point": operating_point,
                    "metric": metric_name,
                    "estimate": rate_data["value"],
                    "ci95_lower": rate_data["ci95_wilson"][0],
                    "ci95_upper": rate_data["ci95_wilson"][1],
                    "numerator": rate_data["numerator"],
                    "denominator": rate_data["denominator"],
                    "ci_method": "Wilson",
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "screening_metrics_table.csv",
        index=False,
    )

    fpr, tpr, thresholds = roc_curve(
        consensus["screening_label"],
        consensus["mean_oof_probability"],
    )
    pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds}
    ).to_csv(output_dir / "roc_curve.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked five-genus colorectal neoplasia screening benchmark."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Locked protocol not found: {PROTOCOL_PATH}")
    frame = load_screening_data(args.data)
    predictions, folds = run_nested_oof(frame)
    consensus = consensus_predictions(predictions)
    metrics = build_metrics(frame, predictions, consensus)
    write_outputs(args.output_dir, predictions, folds, consensus, metrics)

    primary = metrics["operating_points"]["specificity_90"]
    print(f"Wrote screening benchmark to: {args.output_dir.resolve()}")
    print(f"OOF ensemble AUC: {metrics['auc']['any_neoplasia_vs_healthy']['value']:.4f}")
    print(f"FPR at primary operating point: {primary['false_positive_rate']['value']:.4f}")
    print(
        "Small adenoma sensitivity at primary operating point: "
        f"{primary['small_adenoma_sensitivity']['value']:.4f}"
    )


if __name__ == "__main__":
    main()
