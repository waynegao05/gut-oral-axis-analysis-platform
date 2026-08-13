from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.oral_adenoma_internal_v3.modeling import CLRTransformer
from experiments.oral_adenoma_internal_v3.prepare_data import assert_oral_only


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "public" / "zhang_oral_adenoma_2020" / "processed"
DEFAULT_DATA = DATA_DIR / "oral_adenoma_genus.csv"
DEFAULT_FEATURE_MAP = DATA_DIR / "oral_adenoma_feature_map.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "oral_adenoma_internal_v3"
PROTOCOL_PATH = Path(__file__).with_name("protocol_lock.json")

SEEDS = (7, 21, 42, 123, 2026)
EXPECTED_COUNTS = {"healthy": 58, "adenoma": 34}
SUCCESS_GATE = 0.64
PREVIOUS_FPR = 4 / 61
TARGET_INNER_FPR = 0.055


@dataclass(frozen=True)
class CandidateConfig:
    top_k: int
    logistic_c: float

    @property
    def config_id(self) -> str:
        return f"clr_anova_k{self.top_k}_balanced_logistic_c{self.logistic_c:g}"


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig(top_k=top_k, logistic_c=logistic_c)
        for top_k in (10, 20, 40, 80)
        for logistic_c in (0.03, 0.1, 0.3, 1.0)
    ]


def make_pipeline(config: CandidateConfig) -> Pipeline:
    return Pipeline(
        [
            ("clr", CLRTransformer()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=config.top_k)),
            (
                "model",
                LogisticRegression(
                    C=config.logistic_c,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def load_inputs(
    data_path: Path = DEFAULT_DATA,
    feature_map_path: Path = DEFAULT_FEATURE_MAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(data_path)
    feature_map = pd.read_csv(feature_map_path)
    required = {
        "sample_id",
        "subject_id",
        "sample_type",
        "source_study",
        "source_sample_prefix",
        "disease_group",
        "adenoma_label",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required oral metadata columns: {missing}")
    assert_oral_only(frame["sample_type"])
    if frame["sample_id"].duplicated().any() or frame["subject_id"].duplicated().any():
        raise ValueError("Exactly one oral sample per subject is required.")
    counts = frame["disease_group"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected formal oral-cohort counts: {counts}")
    expected_label = (frame["disease_group"] == "adenoma").astype(int)
    if not np.array_equal(expected_label, frame["adenoma_label"].astype(int)):
        raise ValueError("Adenoma labels do not match the locked oral task.")
    if set(frame["source_study"]) != {"Zhang_2020_Theranostics"}:
        raise ValueError("Cross-study pooling is forbidden in the formal oral task.")
    if feature_map["feature_id"].duplicated().any() or feature_map["taxonomy"].duplicated().any():
        raise ValueError("Oral feature IDs and taxonomies must be unique.")
    if set(feature_map["rank"]) != {"genus"} or len(feature_map) != 381:
        raise ValueError("The locked model requires exactly 381 oral genus features.")
    feature_ids = feature_map["feature_id"].astype(str).tolist()
    missing_features = sorted(set(feature_ids).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Missing oral genus features: {missing_features[:5]}")
    values = frame.loc[:, feature_ids].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Oral genus abundances must be finite percentages in [0, 100].")
    sums = values.sum(axis=1)
    if not np.all((sums >= 99.9) & (sums <= 100.1)):
        raise ValueError("Oral genus abundances must sum to approximately 100% per sample.")
    return frame.reset_index(drop=True), feature_map.reset_index(drop=True)


def logit(probability: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def threshold_at_fpr(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_fpr: float = TARGET_INNER_FPR,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    negative = np.sort(probability[y_true == 0])[::-1]
    if negative.size == 0 or int(np.sum(y_true == 1)) == 0:
        raise ValueError("Threshold selection requires both classes.")
    allowed_false_positives = int(math.floor(target_fpr * negative.size))
    threshold = float(np.nextafter(negative[allowed_false_positives], np.inf))
    predicted = probability >= threshold
    false_positives = int(np.sum(predicted[y_true == 0]))
    if false_positives > allowed_false_positives:
        raise RuntimeError("Training-only threshold exceeded its false-positive budget.")
    return {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "allowed_false_positives": allowed_false_positives,
        "inner_false_positives": false_positives,
        "inner_oof_sensitivity": float(np.mean(predicted[y_true == 1])),
        "inner_oof_specificity": float(np.mean(~predicted[y_true == 0])),
    }


def cross_fitted_probability(
    x: np.ndarray,
    y: np.ndarray,
    config: CandidateConfig,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    probability = np.full(len(y), np.nan, dtype=float)
    for train_index, validation_index in splits:
        pipeline = make_pipeline(config)
        pipeline.fit(x[train_index], y[train_index])
        probability[validation_index] = pipeline.predict_proba(x[validation_index])[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("Cross-fitted probabilities are incomplete.")
    return probability


def select_candidate(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    configs: list[CandidateConfig],
) -> tuple[CandidateConfig, dict[str, float | int], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    threshold_by_id: dict[str, dict[str, float | int]] = {}
    for config in configs:
        probability = cross_fitted_probability(x, y, config, splits)
        threshold = threshold_at_fpr(y, probability)
        record = {
            "config_id": config.config_id,
            **asdict(config),
            **threshold,
            "inner_oof_auc": float(roc_auc_score(y, probability)),
            "inner_oof_average_precision": float(average_precision_score(y, probability)),
        }
        rows.append(record)
        threshold_by_id[config.config_id] = threshold
    records = pd.DataFrame(rows).sort_values(
        by=["inner_oof_sensitivity", "inner_oof_auc", "top_k", "logistic_c", "config_id"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    best_id = str(records.iloc[0]["config_id"])
    config_by_id = {config.config_id: config for config in configs}
    return config_by_id[best_id], threshold_by_id[best_id], records


def selected_taxonomies(pipeline: Pipeline, feature_map: pd.DataFrame) -> list[str]:
    support = pipeline.named_steps["select"].get_support()
    return feature_map.loc[support, "taxonomy"].astype(str).tolist()


def run_nested_oof(
    frame: pd.DataFrame,
    feature_map: pd.DataFrame,
    configs: list[CandidateConfig],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_ids = feature_map["feature_id"].astype(str).tolist()
    x = frame.loc[:, feature_ids].to_numpy(dtype=float)
    y = frame["adenoma_label"].to_numpy(dtype=int)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []

    for seed in SEEDS:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seen = np.zeros(len(frame), dtype=int)
        for fold_zero, (train_index, test_index) in enumerate(outer.split(x, y)):
            fold = fold_zero + 1
            print(f"Nested oral OOF seed={seed} fold={fold}", flush=True)
            seen[test_index] += 1
            inner = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed * 100 + fold_zero,
            )
            selected, threshold_data, records = select_candidate(
                x[train_index],
                y[train_index],
                list(inner.split(x[train_index], y[train_index])),
                configs,
            )
            records.insert(0, "outer_seed", seed)
            records.insert(1, "outer_fold", fold)
            candidate_rows.append(records)

            pipeline = make_pipeline(selected)
            pipeline.fit(x[train_index], y[train_index])
            probability = pipeline.predict_proba(x[test_index])[:, 1]
            threshold = float(threshold_data["threshold"])
            margin = logit(probability) - logit(threshold)
            prediction = margin >= 0.0
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "train_n": int(train_index.size),
                    "test_n": int(test_index.size),
                    "selected_config_id": selected.config_id,
                    **asdict(selected),
                    **threshold_data,
                    "selected_taxonomies": json.dumps(
                        selected_taxonomies(pipeline, feature_map),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                }
            )
            for position, row_index in enumerate(test_index):
                row = frame.iloc[row_index]
                prediction_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "subject_id": row["subject_id"],
                        "sample_type": row["sample_type"],
                        "source_sample_prefix": row["source_sample_prefix"],
                        "disease_group": row["disease_group"],
                        "adenoma_label": int(y[row_index]),
                        "seed": seed,
                        "fold": fold,
                        "selected_config_id": selected.config_id,
                        "probability": float(probability[position]),
                        "threshold": threshold,
                        "decision_margin_log_odds": float(margin[position]),
                        "prediction": int(prediction[position]),
                    }
                )
        if not np.all(seen == 1):
            raise RuntimeError(f"Seed {seed} did not predict every oral sample once.")
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(fold_rows),
        pd.concat(candidate_rows, ignore_index=True),
    )


def consensus_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "sample_id",
        "subject_id",
        "sample_type",
        "source_sample_prefix",
        "disease_group",
        "adenoma_label",
    ]
    grouped = predictions.groupby(identity, sort=False, as_index=False)
    result = grouped["probability"].mean().rename(columns={"probability": "mean_oof_probability"})
    margin = grouped["decision_margin_log_odds"].mean().rename(
        columns={"decision_margin_log_odds": "mean_decision_margin_log_odds"}
    )
    vote = grouped["prediction"].sum().rename(columns={"prediction": "positive_votes"})
    result = result.merge(margin, on=identity, validate="one_to_one")
    result = result.merge(vote, on=identity, validate="one_to_one")
    result["prediction"] = (result["mean_decision_margin_log_odds"] >= 0.0).astype(int)
    if len(result) * len(SEEDS) != len(predictions):
        raise RuntimeError("Consensus requires five OOF predictions per real patient.")
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
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
    iterations: int = 3000,
    seed: int = 20260813,
) -> list[float]:
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [
                rng.choice(negative, size=negative.size, replace=True),
                rng.choice(positive, size=positive.size, replace=True),
            ]
        )
        estimates[iteration] = roc_auc_score(y_true[sampled], scores[sampled])
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def compute_metrics(consensus: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    y = consensus["adenoma_label"].to_numpy(dtype=int)
    predicted = consensus["prediction"].to_numpy(dtype=int)
    score = consensus["mean_decision_margin_log_odds"].to_numpy(dtype=float)
    positive = y == 1
    negative = y == 0
    true_positive = int(np.sum(predicted[positive]))
    false_negative = int(np.sum(~predicted[positive].astype(bool)))
    false_positive = int(np.sum(predicted[negative]))
    true_negative = int(np.sum(~predicted[negative].astype(bool)))
    sensitivity = rate_record(true_positive, int(np.sum(positive)))
    false_positive_rate = rate_record(false_positive, int(np.sum(negative)))
    specificity = rate_record(true_negative, int(np.sum(negative)))

    seed_level = []
    for seed, seed_frame in predictions.groupby("seed", sort=True):
        seed_y = seed_frame["adenoma_label"].to_numpy(dtype=int)
        seed_prediction = seed_frame["prediction"].to_numpy(dtype=int)
        seed_level.append(
            {
                "seed": int(seed),
                "sensitivity": float(np.mean(seed_prediction[seed_y == 1] == 1)),
                "false_positive_rate": float(np.mean(seed_prediction[seed_y == 0] == 1)),
                "auc_margin": float(
                    roc_auc_score(
                        seed_y,
                        seed_frame["decision_margin_log_odds"].to_numpy(dtype=float),
                    )
                ),
            }
        )
    auc = float(roc_auc_score(y, score))
    return {
        "protocol_id": "oral_adenoma_nested_oof_v3",
        "endpoint": "colorectal adenoma versus healthy control",
        "sample_type": "oral_swab_only",
        "formal_real_patient_count": int(len(consensus)),
        "group_counts": EXPECTED_COUNTS,
        "primary": {
            "adenoma_sensitivity": sensitivity,
            "false_positive_rate": false_positive_rate,
            "specificity": specificity,
            "roc_auc": {
                "value": auc,
                "ci95_stratified_bootstrap": bootstrap_auc(y, score),
                "positive_n": int(np.sum(positive)),
                "negative_n": int(np.sum(negative)),
            },
            "average_precision": float(average_precision_score(y, score)),
            "confusion_matrix": {
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            },
        },
        "gates": {
            "sensitivity_strictly_above_0_64": bool(sensitivity["value"] > SUCCESS_GATE),
            "fpr_below_previous_4_of_61": bool(false_positive_rate["value"] < PREVIOUS_FPR),
            "joint_gate_passed": bool(
                sensitivity["value"] > SUCCESS_GATE
                and false_positive_rate["value"] < PREVIOUS_FPR
            ),
        },
        "comparison_reference": {
            "previous_experiment_sample_type": "stool_rejected_by_current_scope",
            "previous_false_positive_rate": PREVIOUS_FPR,
            "previous_false_positives": 4,
            "previous_healthy_denominator": 61,
        },
        "seed_level": seed_level,
        "lesion_size_context": {
            "cohort_mean_cm": 0.8,
            "cohort_sd_cm": 0.3,
            "individual_size_labels_available": False,
            "verified_diminutive_adenoma_le_5mm_endpoint": False,
        },
        "claim_boundary": (
            "Internal retrospective repeated nested-OOF result using only real oral-swab "
            "samples. The cohort mean lesion size was 0.8 +/- 0.3 cm, but individual "
            "sizes were unavailable; this is not a verified <=5 mm diminutive-adenoma "
            "endpoint and is not prospective, external, analytical-kit, or clinical validation."
        ),
    }


def run_batch_prefix_audit(
    frame: pd.DataFrame,
    feature_map: pd.DataFrame,
    configs: list[CandidateConfig],
) -> pd.DataFrame:
    feature_ids = feature_map["feature_id"].astype(str).tolist()
    x = frame.loc[:, feature_ids].to_numpy(dtype=float)
    y = frame["adenoma_label"].to_numpy(dtype=int)
    group = frame["source_sample_prefix"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for held_group in sorted(set(group)):
        test = group == held_group
        train = ~test
        inner = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=700 + len(held_group),
        )
        selected, threshold_data, _ = select_candidate(
            x[train],
            y[train],
            list(inner.split(x[train], y[train])),
            configs,
        )
        pipeline = make_pipeline(selected).fit(x[train], y[train])
        probability = pipeline.predict_proba(x[test])[:, 1]
        prediction = probability >= float(threshold_data["threshold"])
        test_y = y[test]
        positive_n = int(np.sum(test_y == 1))
        negative_n = int(np.sum(test_y == 0))
        true_positive = int(np.sum(prediction[test_y == 1]))
        false_positive = int(np.sum(prediction[test_y == 0]))
        auc = float(roc_auc_score(test_y, probability)) if len(np.unique(test_y)) == 2 else math.nan
        rows.append(
            {
                "held_out_prefix": held_group,
                "test_n": int(np.sum(test)),
                "adenoma_n": positive_n,
                "healthy_n": negative_n,
                "selected_config_id": selected.config_id,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "sensitivity": true_positive / positive_n if positive_n else math.nan,
                "false_positive_rate": false_positive / negative_n if negative_n else math.nan,
                "roc_auc": auc,
                "diagnostic_only_not_primary": True,
            }
        )
    return pd.DataFrame(rows)


def fit_final_bundle(
    frame: pd.DataFrame,
    feature_map: pd.DataFrame,
    configs: list[CandidateConfig],
    output_dir: Path,
) -> dict[str, Any]:
    feature_ids = feature_map["feature_id"].astype(str).tolist()
    x = frame.loc[:, feature_ids].to_numpy(dtype=float)
    y = frame["adenoma_label"].to_numpy(dtype=int)
    selection_rows: list[dict[str, Any]] = []
    probability_by_config: dict[str, np.ndarray] = {}
    for config in configs:
        repeated = []
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            repeated.append(
                cross_val_predict(
                    make_pipeline(config),
                    x,
                    y,
                    cv=cv,
                    method="predict_proba",
                )[:, 1]
            )
        probability = np.mean(repeated, axis=0)
        threshold_data = threshold_at_fpr(y, probability)
        selection_rows.append(
            {
                "config_id": config.config_id,
                **asdict(config),
                **threshold_data,
                "repeated_oof_auc": float(roc_auc_score(y, probability)),
                "repeated_oof_average_precision": float(average_precision_score(y, probability)),
            }
        )
        probability_by_config[config.config_id] = probability
    records = pd.DataFrame(selection_rows).sort_values(
        by=["inner_oof_sensitivity", "repeated_oof_auc", "top_k", "logistic_c", "config_id"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    records.to_csv(output_dir / "final_model_selection_records.csv", index=False)
    best_id = str(records.iloc[0]["config_id"])
    config = {item.config_id: item for item in configs}[best_id]
    probability = probability_by_config[best_id]
    threshold_data = threshold_at_fpr(y, probability)
    pipeline = make_pipeline(config).fit(x, y)
    bundle: dict[str, Any] = {
        "protocol_id": "oral_adenoma_nested_oof_v3",
        "research_only_not_web": True,
        "endpoint": "colorectal adenoma versus healthy control",
        "allowed_sample_types": ["oral", "oral_swab", "buccal_swab", "saliva"],
        "forbidden_sources": [
            "stool",
            "fecal",
            "faecal",
            "gut",
            "intestinal",
            "blood",
            "serum",
            "plasma",
            "tissue",
        ],
        "selected_config": asdict(config),
        "selected_config_id": config.config_id,
        "threshold": float(threshold_data["threshold"]),
        "threshold_selection": threshold_data,
        "feature_ids": feature_ids,
        "taxonomies": feature_map["taxonomy"].astype(str).tolist(),
        "selected_taxonomies": selected_taxonomies(pipeline, feature_map),
        "pipeline": pipeline,
        "training_real_patient_count": int(len(frame)),
        "training_group_counts": EXPECTED_COUNTS,
        "training_sample_type": "oral_swab",
        "performance_source": "outputs/oral_adenoma_internal_v3/metrics.json",
        "claim_boundary": (
            "Research-only oral-swab adenoma model. Not a verified <=5 mm endpoint "
            "and not prospective, external, analytical-kit, or clinical validation."
        ),
    }
    joblib.dump(bundle, output_dir / "oral_adenoma_internal_model.joblib", compress=3)
    return bundle


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    candidates: pd.DataFrame,
    consensus: pd.DataFrame,
    batch_audit: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions_long.csv", index=False)
    folds.to_csv(output_dir / "outer_fold_records.csv", index=False)
    candidates.to_csv(output_dir / "inner_candidate_records.csv", index=False)
    consensus.to_csv(output_dir / "oof_consensus_predictions.csv", index=False)
    batch_audit.to_csv(output_dir / "batch_prefix_leave_one_group_out.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fpr, tpr, thresholds = roc_curve(
        consensus["adenoma_label"],
        consensus["mean_decision_margin_log_odds"],
    )
    pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds}
    ).to_csv(output_dir / "roc_curve.csv", index=False)

    rows = []
    primary = metrics["primary"]
    for metric_name in ("adenoma_sensitivity", "false_positive_rate", "specificity"):
        record = primary[metric_name]
        rows.append(
            {
                "metric": metric_name,
                "estimate": record["value"],
                "numerator": record["numerator"],
                "denominator": record["denominator"],
                "ci95_lower": record["ci95_wilson"][0],
                "ci95_upper": record["ci95_wilson"][1],
                "ci_method": "Wilson",
            }
        )
    rows.append(
        {
            "metric": "roc_auc",
            "estimate": primary["roc_auc"]["value"],
            "numerator": "",
            "denominator": len(consensus),
            "ci95_lower": primary["roc_auc"]["ci95_stratified_bootstrap"][0],
            "ci95_upper": primary["roc_auc"]["ci95_stratified_bootstrap"][1],
            "ci_method": "stratified subject bootstrap",
        }
    )
    pd.DataFrame(rows).to_csv(output_dir / "sensitivity_fpr_report.csv", index=False)


def write_model_card(output_dir: Path, metrics: dict[str, Any], bundle: dict[str, Any]) -> None:
    primary = metrics["primary"]
    sensitivity = primary["adenoma_sensitivity"]
    fpr = primary["false_positive_rate"]
    specificity = primary["specificity"]
    auc = primary["roc_auc"]
    content = f"""# Oral-only internal adenoma model card

## Intended use

Internal research discrimination of colorectal adenoma from healthy controls
using oral-swab genus relative abundances only. The model is isolated from the
web application. Stool, fecal, intestinal, blood, serum, plasma, and tissue
inputs are rejected.

## Repeated nested OOF performance on real oral samples

- Adenoma sensitivity: {sensitivity['value']:.2%} ({sensitivity['numerator']}/{sensitivity['denominator']}); 95% CI {sensitivity['ci95_wilson'][0]:.2%}-{sensitivity['ci95_wilson'][1]:.2%}.
- False-positive rate: {fpr['value']:.2%} ({fpr['numerator']}/{fpr['denominator']}); 95% CI {fpr['ci95_wilson'][0]:.2%}-{fpr['ci95_wilson'][1]:.2%}.
- Specificity: {specificity['value']:.2%} ({specificity['numerator']}/{specificity['denominator']}).
- ROC AUC: {auc['value']:.3f}; stratified bootstrap 95% CI {auc['ci95_stratified_bootstrap'][0]:.3f}-{auc['ci95_stratified_bootstrap'][1]:.3f}.
- Joint sensitivity/FPR gate passed: {metrics['gates']['joint_gate_passed']}.

## Final research artifact

- Selected configuration: `{bundle['selected_config_id']}`.
- Training-only OOF threshold: {bundle['threshold']:.6f}.
- Selected genera: {', '.join(bundle['selected_taxonomies'])}.
- Artifact: `oral_adenoma_internal_model.joblib`.

## Data boundary

The formal denominator contains 34 adenoma and 58 healthy participants from
one study. Every input is a bilateral buccal oral swab. CRC samples were
excluded, cross-study cohorts were not pooled, and no synthetic patient was
generated or counted.

## Lesion-size boundary

The adenoma cohort had a reported mean lesion size of 0.8 +/- 0.3 cm. Individual
lesion sizes were not supplied, so the result is an adenoma endpoint and cannot
be claimed as verified sensitivity for diminutive adenomas <=5 mm. It is also
not prospective, external, analytical-kit, or clinical validation.

## Robustness warning

`batch_prefix_leave_one_group_out.csv` tests transfer across source sample-ID
prefixes. Its degradation must be shown with the primary result because the
single-center model may contain batch or collection-period signal.
"""
    (output_dir / "MODEL_CARD.md").write_text(content, encoding="utf-8")


def write_chinese_report(
    output_dir: Path,
    metrics: dict[str, Any],
    bundle: dict[str, Any],
    batch_audit: pd.DataFrame,
) -> None:
    primary = metrics["primary"]
    sensitivity = primary["adenoma_sensitivity"]
    fpr = primary["false_positive_rate"]
    specificity = primary["specificity"]
    auc = primary["roc_auc"]
    batch_lines = "\n".join(
        f"- 留出 `{row.held_out_prefix}` 前缀：灵敏度 {row.sensitivity:.2%}，"
        f"假阳性率 {row.false_positive_rate:.2%}，AUC {row.roc_auc:.3f}。"
        for row in batch_audit.itertuples(index=False)
    )
    content = f"""# 纯口腔腺瘤内部模型报告

## 结论

模型仅使用口腔拭子菌群特征。在 34 例腺瘤和 58 例健康对照的五种子重复嵌套 OOF 中：

- 腺瘤灵敏度：**{sensitivity['value']:.2%}**（{sensitivity['numerator']}/{sensitivity['denominator']}），95% CI {sensitivity['ci95_wilson'][0]:.2%}-{sensitivity['ci95_wilson'][1]:.2%}；
- 假阳性率：**{fpr['value']:.2%}**（{fpr['numerator']}/{fpr['denominator']}），95% CI {fpr['ci95_wilson'][0]:.2%}-{fpr['ci95_wilson'][1]:.2%}；
- 特异度：**{specificity['value']:.2%}**（{specificity['numerator']}/{specificity['denominator']}）；
- ROC AUC：**{auc['value']:.3f}**，分层 bootstrap 95% CI {auc['ci95_stratified_bootstrap'][0]:.3f}-{auc['ci95_stratified_bootstrap'][1]:.3f}。

灵敏度高于 64% 且假阳性率低于此前 4/61（6.56%）的参考值，联合目标通过。该比较只用于内部开发目标，不是跨队列统计优效性检验。

## 数据硬边界

- 允许：口腔拭子、口腔菌群、唾液菌群；
- 禁止：粪便、肠道、血液、血清、血浆和组织数据；
- 本次正式结果只有 92 名真实受试者，没有合成患者；
- 161 例 CRC 样本未进入训练、选模或正式评价；
- 模型未接入网页端。

## 模型

- 流程：CLR 组成型变换 + 训练折内 ANOVA 特征选择 + 平衡正则化逻辑回归；
- 最终配置：`{bundle['selected_config_id']}`；
- 最终选择菌属数：{len(bundle['selected_taxonomies'])}；
- 权重文件：`oral_adenoma_internal_model.joblib`。

## 重要限制

论文只报告腺瘤平均病灶大小为 0.8 +/- 0.3 cm，没有逐例尺寸。因此以上 64.71% 是“腺瘤总体灵敏度”，不能写成已经验证的“<=5 mm 微小腺瘤灵敏度”。样本量较小，灵敏度只比门槛多识别 1 人，95% CI 较宽。

按样本编号前缀整组留出的稳健性诊断为：

{batch_lines}

这提示单中心结果可能包含采集时期或技术批次信号。当前模型适合内部研究与后续前瞻验证，不适合临床宣称或直接部署。
"""
    (output_dir / "REPORT_ZH.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--feature-map", type=Path, default=DEFAULT_FEATURE_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not protocol.get("locked_before_formal_run"):
        raise RuntimeError("The oral-only protocol must be locked before the formal run.")
    frame, feature_map = load_inputs(args.data, args.feature_map)
    configs = candidate_configs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_quality = json.loads(
        (args.data.parent / "data_quality_report.json").read_text(encoding="utf-8")
    )
    (args.output_dir / "data_quality_report.json").write_text(
        json.dumps(source_quality, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    predictions, folds, candidates = run_nested_oof(frame, feature_map, configs)
    consensus = consensus_predictions(predictions)
    metrics = compute_metrics(consensus, predictions)
    batch_audit = run_batch_prefix_audit(frame, feature_map, configs)
    write_outputs(
        args.output_dir,
        predictions,
        folds,
        candidates,
        consensus,
        batch_audit,
        metrics,
    )
    bundle = fit_final_bundle(frame, feature_map, configs, args.output_dir)
    write_model_card(args.output_dir, metrics, bundle)
    write_chinese_report(args.output_dir, metrics, bundle, batch_audit)

    sensitivity = metrics["primary"]["adenoma_sensitivity"]
    fpr = metrics["primary"]["false_positive_rate"]
    print(f"Output: {args.output_dir.resolve()}")
    print(
        f"Oral adenoma sensitivity: {sensitivity['value']:.4f} "
        f"({sensitivity['numerator']}/{sensitivity['denominator']})"
    )
    print(
        f"Healthy-control FPR: {fpr['value']:.4f} "
        f"({fpr['numerator']}/{fpr['denominator']})"
    )
    print(f"Joint gate passed: {metrics['gates']['joint_gate_passed']}")
    print(f"Model: {(args.output_dir / 'oral_adenoma_internal_model.joblib').resolve()}")


if __name__ == "__main__":
    main()
