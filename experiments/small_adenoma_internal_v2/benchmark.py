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
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "public" / "zeller_crc_adenoma_2014" / "processed"
DEFAULT_DATA = DATA_DIR / "zeller_taxonomy_internal_v2.csv"
DEFAULT_FEATURE_MAP = DATA_DIR / "zeller_taxonomy_feature_map_v2.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "small_adenoma_internal_v2"
PROTOCOL_PATH = Path(__file__).with_name("protocol_lock.json")

FORMAL_COUNTS = {"healthy": 61, "small_adenoma": 27}
ALL_COUNTS = {
    "healthy": 61,
    "small_adenoma": 27,
    "large_adenoma": 15,
    "crc": 53,
}
SEEDS = (7, 21, 42, 123, 2026)
TARGET_SENSITIVITY = 0.80
SUCCESS_GATE = 0.64
PREVIOUS_FALSE_POSITIVES = 4
FPR_IMPROVEMENT_MAX_FALSE_POSITIVES = 3


@dataclass(frozen=True)
class CandidateConfig:
    feature_set: str
    top_k: int
    model_name: str
    include_clinical: bool
    augmentation: str

    @property
    def config_id(self) -> str:
        clinical = "clinical" if self.include_clinical else "taxa"
        return (
            f"{self.feature_set}_k{self.top_k}_{self.model_name}_"
            f"{clinical}_{self.augmentation}"
        )


@dataclass
class FoldPreprocessor:
    candidate_feature_ids: tuple[str, ...]
    selected_feature_ids: tuple[str, ...]
    selected_taxonomies: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    clinical_medians: dict[str, float]
    include_clinical: bool
    scaler: StandardScaler

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        y: np.ndarray,
        feature_map: pd.DataFrame,
        config: CandidateConfig,
    ) -> "FoldPreprocessor":
        ranks = {
            "genus": {"genus"},
            "species": {"species"},
            "genus_species": {"genus", "species"},
        }
        if config.feature_set not in ranks:
            raise ValueError(f"Unknown feature set: {config.feature_set}")
        candidate_map = feature_map.loc[
            feature_map["rank"].isin(ranks[config.feature_set])
        ].copy()
        feature_ids = candidate_map["feature_id"].astype(str).tolist()
        raw = frame.loc[:, feature_ids].to_numpy(dtype=float)
        prevalence = np.mean(raw > 0.0, axis=0)
        keep = prevalence >= 0.10
        if int(np.sum(keep)) < 2:
            raise ValueError("Prevalence filtering retained fewer than two taxa.")
        retained_ids = np.asarray(feature_ids, dtype=object)[keep]
        retained_map = candidate_map.set_index("feature_id").loc[retained_ids]
        transformed = transform_abundance(raw[:, keep])
        lower = np.quantile(transformed, 0.01, axis=0)
        upper = np.quantile(transformed, 0.99, axis=0)
        transformed = np.clip(transformed, lower, upper)

        auc_distances = np.asarray(
            [abs(univariate_auc(y, transformed[:, column]) - 0.5) for column in range(transformed.shape[1])],
            dtype=float,
        )
        order = sorted(
            range(len(retained_ids)),
            key=lambda index: (-auc_distances[index], str(retained_ids[index])),
        )
        selected_positions = np.asarray(order[: min(config.top_k, len(order))], dtype=int)
        selected_ids = tuple(str(value) for value in retained_ids[selected_positions])
        selected_taxonomies = tuple(
            str(value)
            for value in retained_map.loc[list(selected_ids), "taxonomy"].tolist()
        )
        selected = transformed[:, selected_positions]

        clinical_medians = {
            "age": finite_median(frame["age"].to_numpy(dtype=float)),
            "bmi": finite_median(frame["bmi"].to_numpy(dtype=float)),
        }
        if config.include_clinical:
            selected = np.column_stack(
                [selected, clinical_matrix(frame, clinical_medians)]
            )
        scaler = StandardScaler().fit(selected)
        return cls(
            candidate_feature_ids=tuple(str(value) for value in retained_ids),
            selected_feature_ids=selected_ids,
            selected_taxonomies=selected_taxonomies,
            lower_bounds=np.asarray(lower[selected_positions], dtype=float),
            upper_bounds=np.asarray(upper[selected_positions], dtype=float),
            clinical_medians=clinical_medians,
            include_clinical=config.include_clinical,
            scaler=scaler,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.loc[:, list(self.selected_feature_ids)].to_numpy(dtype=float)
        transformed = transform_abundance(raw)
        transformed = np.clip(transformed, self.lower_bounds, self.upper_bounds)
        if self.include_clinical:
            transformed = np.column_stack(
                [transformed, clinical_matrix(frame, self.clinical_medians)]
            )
        output = self.scaler.transform(transformed)
        if not np.isfinite(output).all():
            raise ValueError("Preprocessing produced a non-finite value.")
        return output

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        clinical = ("age", "bmi", "sex_male") if self.include_clinical else ()
        return self.selected_feature_ids + clinical


def candidate_configs() -> list[CandidateConfig]:
    feature_specs = (
        ("genus", 15),
        ("genus", 30),
        ("species", 15),
        ("species", 30),
        ("species", 60),
        ("genus_species", 30),
        ("genus_species", 60),
    )
    models = ("log_l2_c01", "log_l2_c1", "log_l1_c01", "extra_trees")
    return [
        CandidateConfig(feature_set, top_k, model_name, include_clinical, augmentation)
        for feature_set, top_k in feature_specs
        for model_name in models
        for include_clinical in (False, True)
        for augmentation in ("none", "minority_mixup")
    ]


def load_inputs(
    data_path: Path = DEFAULT_DATA,
    feature_map_path: Path = DEFAULT_FEATURE_MAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(data_path)
    feature_map = pd.read_csv(feature_map_path)
    required_metadata = {
        "sample_id",
        "subject_id",
        "disease_group",
        "small_adenoma_label",
        "age",
        "bmi",
        "sex",
    }
    missing = sorted(required_metadata.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if frame["sample_id"].duplicated().any() or frame["subject_id"].duplicated().any():
        raise ValueError("Exactly one sample per subject is required.")
    counts = frame["disease_group"].value_counts().to_dict()
    if counts != ALL_COUNTS:
        raise ValueError(f"Unexpected full-cohort counts: {counts}")
    if feature_map["feature_id"].duplicated().any():
        raise ValueError("Taxonomy feature IDs must be unique.")
    expected_features = set(feature_map["feature_id"].astype(str))
    missing_features = sorted(expected_features.difference(frame.columns))
    if missing_features:
        raise ValueError(f"Missing taxonomy features: {missing_features[:5]}")
    if set(feature_map["rank"]) != {"genus", "species"}:
        raise ValueError("Only genus and species features are permitted.")
    values = frame.loc[:, feature_map["feature_id"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Taxonomy abundances must be finite percentages in [0, 100].")
    if not frame["sex"].isin(["Female", "Male"]).all():
        raise ValueError("Unexpected sex category.")

    formal = frame.loc[
        frame["disease_group"].isin(["healthy", "small_adenoma"])
    ].copy()
    formal_counts = formal["disease_group"].value_counts().to_dict()
    if formal_counts != FORMAL_COUNTS:
        raise ValueError(f"Unexpected formal-task counts: {formal_counts}")
    expected_label = (formal["disease_group"] == "small_adenoma").astype(int)
    if not np.array_equal(expected_label, formal["small_adenoma_label"].astype(int)):
        raise ValueError("Small-adenoma labels do not match the locked task.")
    transfer = frame.loc[
        frame["disease_group"].isin(["large_adenoma", "crc"])
    ].copy()
    return formal.reset_index(drop=True), transfer.reset_index(drop=True), feature_map


def transform_abundance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Abundances must be finite percentages in [0, 100].")
    return np.log10(values / 100.0 + 1e-6)


def finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot impute a column with no finite values.")
    return float(np.median(finite))


def clinical_matrix(frame: pd.DataFrame, medians: dict[str, float]) -> np.ndarray:
    age = frame["age"].to_numpy(dtype=float)
    bmi = frame["bmi"].to_numpy(dtype=float)
    age = np.where(np.isfinite(age), age, medians["age"])
    bmi = np.where(np.isfinite(bmi), bmi, medians["bmi"])
    sex_male = (frame["sex"].astype(str).to_numpy() == "Male").astype(float)
    return np.column_stack([age, bmi, sex_male])


def univariate_auc(y_true: np.ndarray, values: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    values = np.asarray(values, dtype=float)
    positive = y_true == 1
    negative = y_true == 0
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(negative))
    ranks = rankdata(values, method="average")
    u_statistic = float(np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2.0)
    return u_statistic / (n_positive * n_negative)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def make_model(config: CandidateConfig, seed: int, *, final: bool = False) -> Any:
    if config.model_name == "log_l2_c01":
        return LogisticRegression(
            C=0.1,
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
    if config.model_name == "log_l2_c1":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
    if config.model_name == "log_l1_c01":
        return LogisticRegression(
            C=0.1,
            penalty="l1",
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
    if config.model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=512 if final else 128,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {config.model_name}")


def augment_minority(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if method == "none":
        return x, y, 0
    if method != "minority_mixup":
        raise ValueError(f"Unknown augmentation method: {method}")
    positive_index = np.flatnonzero(y == 1)
    negative_count = int(np.sum(y == 0))
    needed = max(0, negative_count - positive_index.size)
    if needed == 0:
        return x, y, 0
    rng = np.random.default_rng(seed)
    first = rng.choice(positive_index, size=needed, replace=True)
    second = rng.choice(positive_index, size=needed, replace=True)
    weight = rng.beta(2.0, 2.0, size=(needed, 1))
    synthetic = weight * x[first] + (1.0 - weight) * x[second]
    synthetic += rng.normal(0.0, 0.03, size=synthetic.shape)
    return (
        np.vstack([x, synthetic]),
        np.concatenate([y, np.ones(needed, dtype=int)]),
        needed,
    )


def fit_partition(
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    feature_map: pd.DataFrame,
    config: CandidateConfig,
    seed: int,
    *,
    final: bool = False,
) -> tuple[FoldPreprocessor, Any, int]:
    preprocessor = FoldPreprocessor.fit(train_frame, y_train, feature_map, config)
    x_train = preprocessor.transform(train_frame)
    x_fit, y_fit, generated = augment_minority(
        x_train,
        y_train,
        config.augmentation,
        seed,
    )
    model = make_model(config, seed, final=final)
    model.fit(x_fit, y_fit)
    return preprocessor, model, generated


def predict_probability(preprocessor: FoldPreprocessor, model: Any, frame: pd.DataFrame) -> np.ndarray:
    x = preprocessor.transform(frame)
    probabilities = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced a non-finite probability.")
    return probabilities


def threshold_for_sensitivity(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(
        np.concatenate(
            [probabilities, [np.nextafter(float(np.min(probabilities)), -np.inf)]]
        )
    )
    feasible: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        sensitivity = float(np.mean(predicted[y_true == 1] == 1))
        specificity = float(np.mean(predicted[y_true == 0] == 0))
        if sensitivity + 1e-12 >= target_sensitivity:
            feasible.append((specificity, sensitivity, float(threshold)))
    if not feasible:
        raise RuntimeError("No threshold meets the target sensitivity.")
    specificity, sensitivity, threshold = sorted(
        feasible,
        key=lambda item: (-item[0], -item[1], -item[2]),
    )[0]
    return {
        "threshold": threshold,
        "inner_oof_sensitivity": sensitivity,
        "inner_oof_specificity": specificity,
    }


def inner_oof_for_config(
    frame: pd.DataFrame,
    y: np.ndarray,
    feature_map: pd.DataFrame,
    config: CandidateConfig,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[np.ndarray, int]:
    probabilities = np.full(len(frame), np.nan, dtype=float)
    generated_total = 0
    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        fit_seed = stable_seed(seed, fold, config.config_id)
        preprocessor, model, generated = fit_partition(
            frame.iloc[train_index],
            y[train_index],
            feature_map,
            config,
            fit_seed,
        )
        probabilities[validation_index] = predict_probability(
            preprocessor,
            model,
            frame.iloc[validation_index],
        )
        generated_total += generated
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Inner OOF probabilities are incomplete.")
    return probabilities, generated_total


def select_candidate(
    frame: pd.DataFrame,
    y: np.ndarray,
    feature_map: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    configs: list[CandidateConfig],
) -> tuple[CandidateConfig, np.ndarray, dict[str, float], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    probability_by_id: dict[str, np.ndarray] = {}
    threshold_by_id: dict[str, dict[str, float]] = {}
    for config in configs:
        probabilities, generated_total = inner_oof_for_config(
            frame,
            y,
            feature_map,
            config,
            splits,
            seed,
        )
        threshold = threshold_for_sensitivity(y, probabilities)
        auc = float(roc_auc_score(y, probabilities))
        record = {
            "config_id": config.config_id,
            **asdict(config),
            "inner_oof_auc": auc,
            **threshold,
            "generated_training_rows_across_inner_folds": generated_total,
        }
        candidates.append(record)
        probability_by_id[config.config_id] = probabilities
        threshold_by_id[config.config_id] = threshold

    records = pd.DataFrame(candidates)
    records = records.sort_values(
        by=[
            "inner_oof_specificity",
            "inner_oof_auc",
            "top_k",
            "config_id",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    best_id = str(records.iloc[0]["config_id"])
    config_by_id = {config.config_id: config for config in configs}
    return (
        config_by_id[best_id],
        probability_by_id[best_id],
        threshold_by_id[best_id],
        records,
    )


def run_nested_oof(
    frame: pd.DataFrame,
    feature_map: pd.DataFrame,
    configs: list[CandidateConfig],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame["small_adenoma_label"].to_numpy(dtype=int)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []

    for seed in SEEDS:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seen = np.zeros(len(frame), dtype=int)
        for fold, (train_index, test_index) in enumerate(outer.split(frame, y), start=1):
            print(
                f"Nested OOF seed={seed} fold={fold}: evaluating {len(configs)} candidates",
                flush=True,
            )
            seen[test_index] += 1
            train_frame = frame.iloc[train_index].reset_index(drop=True)
            y_train = y[train_index]
            inner = StratifiedKFold(
                n_splits=4,
                shuffle=True,
                random_state=stable_seed(seed, fold, "inner"),
            )
            inner_splits = list(inner.split(train_frame, y_train))
            selected, _, threshold_data, records = select_candidate(
                train_frame,
                y_train,
                feature_map,
                inner_splits,
                stable_seed(seed, fold, "selection"),
                configs,
            )
            records.insert(0, "outer_seed", seed)
            records.insert(1, "outer_fold", fold)
            candidate_rows.append(records)

            fit_seed = stable_seed(seed, fold, selected.config_id, "outer_fit")
            preprocessor, model, generated = fit_partition(
                train_frame,
                y_train,
                feature_map,
                selected,
                fit_seed,
            )
            test_probability = predict_probability(
                preprocessor,
                model,
                frame.iloc[test_index],
            )
            threshold = float(threshold_data["threshold"])
            test_prediction = (test_probability >= threshold).astype(int)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "train_n": int(train_index.size),
                    "test_n": int(test_index.size),
                    "selected_config_id": selected.config_id,
                    **asdict(selected),
                    **threshold_data,
                    "outer_generated_training_rows": generated,
                    "selected_features": json.dumps(
                        list(preprocessor.selected_taxonomies),
                        separators=(",", ":"),
                    ),
                }
            )
            for position, row_index in enumerate(test_index):
                prediction_rows.append(
                    {
                        "sample_id": frame.iloc[row_index]["sample_id"],
                        "subject_id": frame.iloc[row_index]["subject_id"],
                        "disease_group": frame.iloc[row_index]["disease_group"],
                        "small_adenoma_label": int(y[row_index]),
                        "seed": seed,
                        "fold": fold,
                        "selected_config_id": selected.config_id,
                        "probability": float(test_probability[position]),
                        "threshold": threshold,
                        "decision_margin": float(test_probability[position] - threshold),
                        "prediction": int(test_prediction[position]),
                    }
                )
        if not np.all(seen == 1):
            raise RuntimeError(f"Seed {seed} did not predict every formal sample once.")
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(fold_rows),
        pd.concat(candidate_rows, ignore_index=True),
    )


def consensus_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identity = ["sample_id", "subject_id", "disease_group", "small_adenoma_label"]
    grouped = predictions.groupby(identity, sort=False, as_index=False)
    probability = grouped["probability"].mean().rename(
        columns={"probability": "mean_oof_probability"}
    )
    margin = grouped["decision_margin"].mean().rename(
        columns={"decision_margin": "mean_decision_margin"}
    )
    votes = grouped["prediction"].sum().rename(
        columns={"prediction": "positive_votes"}
    )
    consensus = probability.merge(margin, on=identity, validate="one_to_one")
    consensus = consensus.merge(votes, on=identity, validate="one_to_one")
    consensus["prediction"] = (consensus["positive_votes"] >= 3).astype(int)
    if len(consensus) * len(SEEDS) != len(predictions):
        raise RuntimeError("Consensus did not preserve five predictions per patient.")
    return consensus


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
    iterations: int = 2000,
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
    y = consensus["small_adenoma_label"].to_numpy(dtype=int)
    predicted = consensus["prediction"].to_numpy(dtype=int)
    score = consensus["mean_decision_margin"].to_numpy(dtype=float)
    positive = y == 1
    negative = y == 0
    true_positive = int(np.sum(predicted[positive] == 1))
    false_negative = int(np.sum(predicted[positive] == 0))
    false_positive = int(np.sum(predicted[negative] == 1))
    true_negative = int(np.sum(predicted[negative] == 0))
    sensitivity = rate_record(true_positive, int(np.sum(positive)))
    fpr = rate_record(false_positive, int(np.sum(negative)))
    specificity = rate_record(true_negative, int(np.sum(negative)))

    seed_level: list[dict[str, Any]] = []
    for seed, seed_frame in predictions.groupby("seed", sort=True):
        seed_y = seed_frame["small_adenoma_label"].to_numpy(dtype=int)
        seed_predicted = seed_frame["prediction"].to_numpy(dtype=int)
        seed_level.append(
            {
                "seed": int(seed),
                "sensitivity": float(np.mean(seed_predicted[seed_y == 1] == 1)),
                "false_positive_rate": float(np.mean(seed_predicted[seed_y == 0] == 1)),
                "auc_margin": float(
                    roc_auc_score(seed_y, seed_frame["decision_margin"].to_numpy(dtype=float))
                ),
            }
        )
    return {
        "protocol_id": "small_adenoma_internal_nested_oof_v2",
        "formal_real_patient_count": int(len(consensus)),
        "group_counts": FORMAL_COUNTS,
        "primary": {
            "small_adenoma_sensitivity": sensitivity,
            "false_positive_rate": fpr,
            "specificity": specificity,
            "auc": {
                "value": float(roc_auc_score(y, score)),
                "ci95_bootstrap": bootstrap_auc(y, score),
                "negative_n": int(np.sum(negative)),
                "positive_n": int(np.sum(positive)),
            },
            "confusion_matrix": {
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            },
        },
        "gates": {
            "sensitivity_strictly_above_0_64": bool(sensitivity["value"] > SUCCESS_GATE),
            "false_positives_below_v1_four_of_61": bool(
                false_positive <= FPR_IMPROVEMENT_MAX_FALSE_POSITIVES
            ),
            "joint_stretch_gate_passed": bool(
                sensitivity["value"] > SUCCESS_GATE
                and false_positive <= FPR_IMPROVEMENT_MAX_FALSE_POSITIVES
            ),
        },
        "comparison_reference": {
            "five_genus_v1_false_positives": PREVIOUS_FALSE_POSITIVES,
            "five_genus_v1_healthy_denominator": 61,
            "strict_fpr_improvement_requires_at_most": FPR_IMPROVEMENT_MAX_FALSE_POSITIVES,
        },
        "seed_level": seed_level,
        "claim_boundary": (
            "Internal retrospective nested-OOF research result. Synthetic rows were "
            "training-only and are excluded from every reported denominator. This is "
            "not external, prospective, kit-analytical, or clinical validation."
        ),
    }


def select_and_fit_final_bundle(
    formal: pd.DataFrame,
    transfer: pd.DataFrame,
    feature_map: pd.DataFrame,
    configs: list[CandidateConfig],
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = formal["small_adenoma_label"].to_numpy(dtype=int)
    print(f"Final bundle selection: evaluating {len(configs)} candidates", flush=True)
    full_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8801)
    selected, full_oof, threshold_data, records = select_candidate(
        formal,
        y,
        feature_map,
        list(full_cv.split(formal, y)),
        8801,
        configs,
    )
    records.to_csv(output_dir / "final_model_selection_records.csv", index=False)
    members: list[dict[str, Any]] = []
    for seed in SEEDS:
        fit_seed = stable_seed(seed, selected.config_id, "final_bundle")
        preprocessor, model, generated = fit_partition(
            formal,
            y,
            feature_map,
            selected,
            fit_seed,
            final=True,
        )
        members.append(
            {
                "seed": seed,
                "preprocessor": preprocessor,
                "model": model,
                "generated_training_rows": generated,
            }
        )

    bundle: dict[str, Any] = {
        "protocol_id": "small_adenoma_internal_nested_oof_v2",
        "research_only_not_web": True,
        "selected_config": asdict(selected),
        "selected_config_id": selected.config_id,
        "threshold": float(threshold_data["threshold"]),
        "threshold_inner_oof_sensitivity": float(
            threshold_data["inner_oof_sensitivity"]
        ),
        "threshold_inner_oof_specificity": float(
            threshold_data["inner_oof_specificity"]
        ),
        "member_vote_rule": "mean probability >= threshold",
        "members": members,
        "feature_map": feature_map,
        "expected_input_columns": list(formal.columns),
        "performance_source": "outputs/small_adenoma_internal_v2/metrics.json",
    }
    joblib.dump(bundle, output_dir / "small_adenoma_internal_model.joblib", compress=3)

    transfer_probabilities = []
    for member in members:
        transfer_probabilities.append(
            predict_probability(member["preprocessor"], member["model"], transfer)
        )
    transfer_output = transfer.loc[
        :, ["sample_id", "subject_id", "disease_group"]
    ].copy()
    transfer_output["mean_probability"] = np.mean(transfer_probabilities, axis=0)
    transfer_output["prediction"] = (
        transfer_output["mean_probability"] >= bundle["threshold"]
    ).astype(int)
    return bundle, transfer_output


def write_model_card(
    output_dir: Path,
    metrics: dict[str, Any],
    bundle: dict[str, Any],
) -> None:
    primary = metrics["primary"]
    sensitivity = primary["small_adenoma_sensitivity"]
    fpr = primary["false_positive_rate"]
    specificity = primary["specificity"]
    auc = primary["auc"]
    content = f"""# Internal small-adenoma model card

## Intended use

Research-only discrimination of small adenoma (<10 mm) from healthy controls.
The bundle is not connected to the web application.

## Nested OOF performance on real patients

- Small-adenoma sensitivity: {sensitivity['value']:.1%} ({sensitivity['numerator']}/{sensitivity['denominator']}); 95% CI {sensitivity['ci95_wilson'][0]:.1%}-{sensitivity['ci95_wilson'][1]:.1%}.
- False-positive rate: {fpr['value']:.1%} ({fpr['numerator']}/{fpr['denominator']}); 95% CI {fpr['ci95_wilson'][0]:.1%}-{fpr['ci95_wilson'][1]:.1%}.
- Specificity: {specificity['value']:.1%} ({specificity['numerator']}/{specificity['denominator']}).
- ROC AUC: {auc['value']:.3f}; bootstrap 95% CI {auc['ci95_bootstrap'][0]:.3f}-{auc['ci95_bootstrap'][1]:.3f}.
- Sensitivity >64% gate: {metrics['gates']['sensitivity_strictly_above_0_64']}.
- False positives below the v1 reference of 4/61: {metrics['gates']['false_positives_below_v1_four_of_61']}.
- Joint sensitivity/FPR stretch gate: {metrics['gates']['joint_stretch_gate_passed']}.

## Final research artifact

- Selected configuration: `{bundle['selected_config_id']}`.
- Full-development OOF threshold: {bundle['threshold']:.6f}.
- Artifact: `small_adenoma_internal_model.joblib`.

## Data-generation boundary

Minority mixup, when selected, was generated independently within each training
partition. No synthetic sample was placed in validation/test folds or counted in
the reported 27 small adenomas and 61 healthy controls.

## Limitation

This is one public retrospective cohort. The numbers are not prospective,
external, colorimetric-kit analytical, or clinical validation results.
"""
    (output_dir / "MODEL_CARD.md").write_text(content, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    candidates: pd.DataFrame,
    consensus: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions_long.csv", index=False)
    folds.to_csv(output_dir / "outer_fold_records.csv", index=False)
    candidates.to_csv(output_dir / "inner_candidate_records.csv", index=False)
    consensus.to_csv(output_dir / "oof_consensus_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fpr, tpr, thresholds = roc_curve(
        consensus["small_adenoma_label"],
        consensus["mean_decision_margin"],
    )
    pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds}
    ).to_csv(output_dir / "roc_curve.csv", index=False)

    primary = metrics["primary"]
    rows = []
    for metric_name in (
        "small_adenoma_sensitivity",
        "false_positive_rate",
        "specificity",
    ):
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
            "estimate": primary["auc"]["value"],
            "numerator": "",
            "denominator": len(consensus),
            "ci95_lower": primary["auc"]["ci95_bootstrap"][0],
            "ci95_upper": primary["auc"]["ci95_bootstrap"][1],
            "ci_method": "subject-level bootstrap",
        }
    )
    pd.DataFrame(rows).to_csv(output_dir / "sensitivity_fpr_report.csv", index=False)


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
        raise RuntimeError("The formal protocol must be locked before running.")
    formal, transfer, feature_map = load_inputs(args.data, args.feature_map)
    configs = candidate_configs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_quality = {
        "formal_real_patients": int(len(formal)),
        "healthy": int(np.sum(formal["disease_group"] == "healthy")),
        "small_adenoma_lt_10_mm": int(
            np.sum(formal["disease_group"] == "small_adenoma")
        ),
        "unique_samples": int(formal["sample_id"].nunique()),
        "unique_subjects": int(formal["subject_id"].nunique()),
        "genus_features": int(np.sum(feature_map["rank"] == "genus")),
        "species_features": int(np.sum(feature_map["rank"] == "species")),
        "missing_bmi": int(formal["bmi"].isna().sum()),
        "invalid_abundance_values": 0,
    }
    (args.output_dir / "data_quality_report.json").write_text(
        json.dumps(data_quality, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions, folds, candidates = run_nested_oof(formal, feature_map, configs)
    consensus = consensus_predictions(predictions)
    metrics = compute_metrics(consensus, predictions)
    write_outputs(args.output_dir, predictions, folds, candidates, consensus, metrics)
    bundle, transfer_output = select_and_fit_final_bundle(
        formal,
        transfer,
        feature_map,
        configs,
        args.output_dir,
    )
    transfer_output.to_csv(args.output_dir / "transfer_only_predictions.csv", index=False)
    write_model_card(args.output_dir, metrics, bundle)

    sensitivity = metrics["primary"]["small_adenoma_sensitivity"]["value"]
    fpr_value = metrics["primary"]["false_positive_rate"]["value"]
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Small-adenoma sensitivity: {sensitivity:.4f}")
    print(f"Healthy-control false-positive rate: {fpr_value:.4f}")
    print(f"Joint stretch gate passed: {metrics['gates']['joint_stretch_gate_passed']}")
    print(f"Model: {(args.output_dir / 'small_adenoma_internal_model.joblib').resolve()}")


if __name__ == "__main__":
    main()
