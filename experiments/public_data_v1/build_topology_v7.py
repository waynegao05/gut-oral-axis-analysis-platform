from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, LedoitWolf, OAS
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.metrics import concordance_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_TAXA = (
    "Fusobacterium",
    "Porphyromonas",
    "Prevotella",
    "Streptococcus",
    "Lactobacillus",
)
EDGE_PAIRS = (
    ("Fusobacterium", "Porphyromonas"),
    ("Fusobacterium", "Prevotella"),
    ("Fusobacterium", "Streptococcus"),
    ("Fusobacterium", "Lactobacillus"),
    ("Porphyromonas", "Prevotella"),
    ("Porphyromonas", "Streptococcus"),
    ("Porphyromonas", "Lactobacillus"),
    ("Prevotella", "Streptococcus"),
    ("Prevotella", "Lactobacillus"),
    ("Streptococcus", "Lactobacillus"),
)
GENERATOR_VERSION = "topology_v7_hybrid_generator_v2"
PREVIOUS_GENERATOR_VERSION = "topology_v7_hybrid_generator_v1"


@dataclass(frozen=True)
class SourcePaths:
    public_features: Path
    v6_graph: Path
    v6_clinical: Path
    v6_metabolite: Path
    v6_label: Path


@dataclass(frozen=True)
class OutputPaths:
    graph: Path
    clinical: Path
    metabolite: Path
    label: Path
    oral_gut: Path
    provenance: Path
    manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _logit(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(probability / (1.0 - probability))


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)


def _clr(values: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    composition = np.maximum(np.asarray(values, dtype=float), pseudocount)
    composition /= composition.sum(axis=1, keepdims=True)
    log_values = np.log(composition)
    return log_values - log_values.mean(axis=1, keepdims=True)


def _empirical_quantile_map(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy()
    quantiles = np.clip(ranks, 1.0 / (len(values) + 1.0), len(values) / (len(values) + 1.0))
    try:
        return np.quantile(reference, quantiles, method="linear")
    except TypeError:
        return np.quantile(reference, quantiles, interpolation="linear")


def _frozen_empirical_quantile_map(
    values: np.ndarray,
    calibration_source: np.ndarray,
    calibration_target: np.ndarray,
) -> np.ndarray:
    source = np.sort(np.asarray(calibration_source, dtype=float))
    if len(source) < 3 or np.ptp(source) <= 1e-12:
        raise ValueError("Frozen quantile calibration requires a non-constant source reference.")
    probabilities = (np.arange(len(source), dtype=float) + 0.5) / len(source)
    try:
        target = np.quantile(calibration_target, probabilities, method="linear")
    except TypeError:
        target = np.quantile(calibration_target, probabilities, interpolation="linear")
    return np.interp(np.asarray(values, dtype=float), source, target, left=target[0], right=target[-1])


def _positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-8)
    return (vectors * values) @ vectors.T


def _sample_gaussian_residuals(
    rng: np.random.Generator,
    residuals: np.ndarray,
    count: int,
    scale: float,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    covariance = OAS().fit(residuals).covariance_
    covariance = _positive_semidefinite(covariance) * float(scale) ** 2
    return rng.multivariate_normal(np.zeros(residuals.shape[1]), covariance, size=count)


def _site_composition(
    public: pd.DataFrame,
    site: str,
    max_features: int,
) -> tuple[np.ndarray, list[str]]:
    prefix = f"{site}__"
    columns = [column for column in public.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No {site} abundance columns were found in the public feature table.")

    mandatory = [f"{prefix}genus:{name}" for name in PANEL_TAXA]
    missing = [column for column in mandatory if column not in columns]
    if missing:
        raise ValueError(f"The public feature table is missing required panel taxa: {missing}")

    prevalence = (public[columns].astype(float) > 0.0).sum(axis=0)
    means = public[columns].astype(float).mean(axis=0)
    candidates = [column for column in columns if prevalence[column] >= 3 and column not in mandatory]
    candidates.sort(key=lambda column: (-float(means[column]), column))
    selected = mandatory + candidates[: max(0, max_features - len(mandatory))]

    selected_values = public[selected].to_numpy(dtype=float)
    other = np.maximum(1.0 - selected_values.sum(axis=1), 0.0)
    composition = np.column_stack([selected_values, other])
    composition /= np.maximum(composition.sum(axis=1, keepdims=True), 1e-12)
    labels = [column.removeprefix(prefix) for column in selected] + ["__other__"]
    return composition, labels


def _assign_anchor_groups(target: np.ndarray, groups: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignments = np.full(len(target), -1, dtype=int)
    for class_value in sorted(np.unique(target).tolist()):
        indices = np.flatnonzero(target == class_value)
        rng.shuffle(indices)
        assignments[indices] = np.arange(len(indices), dtype=int) % groups
    if (assignments < 0).any():
        raise RuntimeError("Not every public anchor was assigned to a generation group.")
    return assignments


def _anchor_balance_statistics(
    assignments: np.ndarray,
    standardized_features: np.ndarray,
    groups: int,
) -> dict[str, Any]:
    group_means = np.vstack(
        [standardized_features[assignments == group_id].mean(axis=0) for group_id in range(groups)]
    )
    group_scales = np.vstack(
        [standardized_features[assignments == group_id].std(axis=0, ddof=0) for group_id in range(groups)]
    )
    group_sizes = [int(np.sum(assignments == group_id)) for group_id in range(groups)]
    return {
        "maximum_absolute_standardized_mean": float(np.max(np.abs(group_means))),
        "mean_absolute_standardized_mean": float(np.mean(np.abs(group_means))),
        "maximum_absolute_standardized_scale_delta": float(np.max(np.abs(group_scales - 1.0))),
        "group_sizes": group_sizes,
    }


def _assign_anchor_groups_balanced(
    target: np.ndarray,
    balance_features: np.ndarray,
    groups: int,
    seed: int,
    searches: int = 20000,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    features = StandardScaler().fit_transform(np.asarray(balance_features, dtype=float))
    best_assignments: np.ndarray | None = None
    best_objective = float("inf")
    best_stats: dict[str, Any] | None = None

    for _ in range(int(searches)):
        assignments = np.full(len(target), -1, dtype=int)
        for class_value in sorted(np.unique(target).tolist()):
            indices = np.flatnonzero(target == class_value).copy()
            rng.shuffle(indices)
            offset = int(rng.integers(0, groups))
            assignments[indices] = (np.arange(len(indices), dtype=int) + offset) % groups
        stats = _anchor_balance_statistics(assignments, features, groups)
        objective = (
            stats["maximum_absolute_standardized_mean"]
            + 0.20 * stats["mean_absolute_standardized_mean"]
            + 0.10 * stats["maximum_absolute_standardized_scale_delta"]
        )
        if objective < best_objective:
            best_objective = objective
            best_assignments = assignments.copy()
            best_stats = stats

    if best_assignments is None or best_stats is None:
        raise RuntimeError("Balanced anchor assignment did not produce a candidate partition.")
    best_stats = {
        **best_stats,
        "searches": int(searches),
        "objective": float(best_objective),
        "method": "outcome_blind_randomized_balanced_partition_search",
    }
    return best_assignments, best_stats


def _fuse_panel(
    saliva: np.ndarray,
    stool: np.ndarray,
    observed_saliva: np.ndarray,
    observed_stool: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    saliva_prevalence = (observed_saliva > 0.0).mean(axis=0)
    stool_prevalence = (observed_stool > 0.0).mean(axis=0)
    total = np.maximum(saliva_prevalence + stool_prevalence, 1e-12)
    saliva_weights = saliva_prevalence / total
    stool_weights = stool_prevalence / total
    fused = saliva * saliva_weights + stool * stool_weights
    weights = {
        name: {
            "saliva": float(saliva_weights[index]),
            "stool": float(stool_weights[index]),
        }
        for index, name in enumerate(PANEL_TAXA)
    }
    return fused, weights


def _public_panel(public: pd.DataFrame, site: str) -> np.ndarray:
    return public[[f"{site}__genus:{name}" for name in PANEL_TAXA]].to_numpy(dtype=float)


def _generate_microbiome(
    public: pd.DataFrame,
    v6_abundance: pd.DataFrame,
    *,
    sample_count: int,
    seed: int,
    generation_groups: int,
    max_features_per_site: int,
    latent_components: int,
    anchor_shrinkage_to_class_mean: float = 0.10,
    latent_noise_scale: float = 0.10,
    balance_target_by_group: bool = False,
    balance_anchor_features: bool = False,
    anchor_balance_searches: int = 20000,
    anchor_prior_strength: float | None = None,
    frozen_quantile_calibration: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 <= anchor_shrinkage_to_class_mean <= 1.0:
        raise ValueError("anchor_shrinkage_to_class_mean must be between 0 and 1.")
    if latent_noise_scale < 0.0:
        raise ValueError("latent_noise_scale must be non-negative.")
    if anchor_prior_strength is not None and anchor_prior_strength <= 0.0:
        raise ValueError("anchor_prior_strength must be positive when provided.")

    rng = np.random.default_rng(seed)
    target = public["target_crc"].astype(int).to_numpy()
    patient_ids = public["patient_id"].astype(str).to_numpy()

    saliva_comp, saliva_labels = _site_composition(public, "saliva", max_features_per_site)
    stool_comp, stool_labels = _site_composition(public, "stool", max_features_per_site)
    saliva_clr = _clr(saliva_comp)
    stool_clr = _clr(stool_comp)
    joint_clr = np.column_stack([saliva_clr, stool_clr])

    scaler = StandardScaler()
    standardized = scaler.fit_transform(joint_clr)
    component_count = min(int(latent_components), len(public) - 2, standardized.shape[1])
    pca = PCA(n_components=component_count, random_state=seed)
    scores = pca.fit_transform(standardized)
    global_covariance = LedoitWolf().fit(scores).covariance_
    observed_saliva_panel = _public_panel(public, "saliva")
    observed_stool_panel = _public_panel(public, "stool")
    observed_fused, fusion_weights = _fuse_panel(
        observed_saliva_panel,
        observed_stool_panel,
        observed_saliva_panel,
        observed_stool_panel,
    )
    if balance_anchor_features:
        anchor_groups, anchor_balance = _assign_anchor_groups_balanced(
            target,
            np.column_stack([scores, _logit(observed_fused)]),
            generation_groups,
            seed + 11,
            searches=anchor_balance_searches,
        )
    else:
        anchor_groups = _assign_anchor_groups(target, generation_groups, seed + 11)
        balance_features = StandardScaler().fit_transform(
            np.column_stack([scores, _logit(observed_fused)])
        )
        anchor_balance = {
            **_anchor_balance_statistics(anchor_groups, balance_features, generation_groups),
            "searches": 1,
            "method": "crc_stratified_random_round_robin",
        }

    class_parameters: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for class_value in sorted(np.unique(target).tolist()):
        class_scores = scores[target == class_value]
        class_covariance = LedoitWolf().fit(class_scores).covariance_
        blended = 0.70 * class_covariance + 0.30 * global_covariance
        class_parameters[int(class_value)] = (
            class_scores.mean(axis=0),
            _positive_semidefinite(blended),
        )

    group_sequence = np.arange(sample_count, dtype=int) % generation_groups
    rng.shuffle(group_sequence)
    target_probability = float(target.mean())
    if balance_target_by_group:
        generated_target = np.zeros(sample_count, dtype=int)
        for group_id in sorted(np.unique(group_sequence).tolist()):
            group_indices = np.flatnonzero(group_sequence == group_id)
            positive_count = int(round(target_probability * len(group_indices)))
            group_targets = np.zeros(len(group_indices), dtype=int)
            group_targets[:positive_count] = 1
            rng.shuffle(group_targets)
            generated_target[group_indices] = group_targets
    else:
        generated_target = rng.binomial(1, target_probability, size=sample_count).astype(int)
    generated_scores = np.zeros((sample_count, component_count), dtype=float)
    primary_indices = np.zeros(sample_count, dtype=int)
    secondary_indices = np.zeros(sample_count, dtype=int)
    interpolation_weights = np.zeros(sample_count, dtype=float)
    local_anchor_weights = np.zeros(sample_count, dtype=float)

    for row_index, (group_id, class_value) in enumerate(zip(group_sequence, generated_target)):
        candidates = np.flatnonzero((anchor_groups == group_id) & (target == class_value))
        if len(candidates) == 0:
            candidates = np.flatnonzero(target == class_value)
        if len(candidates) == 1:
            first = second = int(candidates[0])
        else:
            first, second = rng.choice(candidates, size=2, replace=False).astype(int).tolist()
        weight = float(rng.beta(2.0, 2.0))
        class_mean, class_covariance = class_parameters[int(class_value)]
        local_score = weight * scores[first] + (1.0 - weight) * scores[second]
        if anchor_prior_strength is None:
            local_anchor_weight = 1.0 - anchor_shrinkage_to_class_mean
        else:
            local_anchor_weight = float(
                np.sqrt(len(candidates) / (len(candidates) + anchor_prior_strength))
            )
        local_score = (
            local_anchor_weight * local_score
            + (1.0 - local_anchor_weight) * class_mean
        )
        noise = rng.multivariate_normal(
            np.zeros(component_count),
            class_covariance * latent_noise_scale,
        )
        generated_scores[row_index] = local_score + noise
        primary_indices[row_index] = first
        secondary_indices[row_index] = second
        interpolation_weights[row_index] = weight
        local_anchor_weights[row_index] = local_anchor_weight

    reconstructed = scaler.inverse_transform(pca.inverse_transform(generated_scores))
    saliva_width = saliva_comp.shape[1]
    generated_saliva = _softmax_rows(reconstructed[:, :saliva_width])
    generated_stool = _softmax_rows(reconstructed[:, saliva_width:])

    saliva_index = {label: index for index, label in enumerate(saliva_labels)}
    stool_index = {label: index for index, label in enumerate(stool_labels)}
    generated_saliva_panel = np.column_stack(
        [generated_saliva[:, saliva_index[f"genus:{name}"]] for name in PANEL_TAXA]
    )
    generated_stool_panel = np.column_stack(
        [generated_stool[:, stool_index[f"genus:{name}"]] for name in PANEL_TAXA]
    )
    generated_fused, _ = _fuse_panel(
        generated_saliva_panel,
        generated_stool_panel,
        observed_saliva_panel,
        observed_stool_panel,
    )

    calibrated = np.zeros_like(generated_fused)
    observed_calibrated = np.zeros_like(observed_fused)
    for column_index, taxon in enumerate(PANEL_TAXA):
        reference = v6_abundance[taxon].to_numpy(dtype=float)
        if frozen_quantile_calibration:
            calibrated[:, column_index] = _frozen_empirical_quantile_map(
                generated_fused[:, column_index], observed_fused[:, column_index], reference
            )
            observed_calibrated[:, column_index] = _frozen_empirical_quantile_map(
                observed_fused[:, column_index], observed_fused[:, column_index], reference
            )
        else:
            calibrated[:, column_index] = _empirical_quantile_map(
                generated_fused[:, column_index], reference
            )
            observed_calibrated[:, column_index] = _empirical_quantile_map(
                observed_fused[:, column_index], reference
            )

    sample_ids = np.asarray([f"S{index + 1}" for index in range(sample_count)])
    abundance = pd.DataFrame(calibrated, columns=PANEL_TAXA)
    abundance.insert(0, "sample_id", sample_ids)

    oral_gut_rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(sample_ids):
        for taxon_index, taxon in enumerate(PANEL_TAXA):
            oral_gut_rows.append(
                {
                    "sample_id": sample_id,
                    "taxon": taxon,
                    "saliva_relative_abundance": generated_saliva_panel[row_index, taxon_index],
                    "stool_relative_abundance": generated_stool_panel[row_index, taxon_index],
                    "fused_raw_abundance": generated_fused[row_index, taxon_index],
                    "model_abundance": calibrated[row_index, taxon_index],
                }
            )
    oral_gut = pd.DataFrame(oral_gut_rows)

    provenance = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "generation_group_id": group_sequence,
            "primary_anchor_patient_id": patient_ids[primary_indices],
            "secondary_anchor_patient_id": patient_ids[secondary_indices],
            "anchor_target_crc": generated_target,
            "anchor_condition": np.where(generated_target == 1, "Adenocarcinoma", "Adenoma"),
            "latent_interpolation_weight": interpolation_weights,
            "local_anchor_weight": local_anchor_weights,
            "microbiome_source": "model_generated_from_public_paired_oral_gut",
            "clinical_source": "model_generated_from_topology_v6_prior",
            "metabolite_source": "model_generated_from_topology_v6_prior",
            "function_source": "model_inferred_from_topology_v6_prior",
            "edge_source": "model_inferred_from_public_graphical_lasso",
            "label_source": "model_generated_survival_proxy_from_topology_v6_prior",
            "generator_seed": int(seed),
        }
    )

    metadata = {
        "observed_public_patients": int(len(public)),
        "generated_samples": int(sample_count),
        "generation_groups": int(generation_groups),
        "latent_components": int(component_count),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "max_features_per_site": int(max_features_per_site),
        "selected_saliva_components": int(len(saliva_labels)),
        "selected_stool_components": int(len(stool_labels)),
        "target_crc_rate_observed": float(target.mean()),
        "target_crc_rate_generated": float(generated_target.mean()),
        "target_crc_rate_by_generation_group": {
            str(int(group_id)): float(generated_target[group_sequence == group_id].mean())
            for group_id in sorted(np.unique(group_sequence).tolist())
        },
        "anchor_shrinkage_to_class_mean": float(anchor_shrinkage_to_class_mean),
        "anchor_prior_strength": (
            None if anchor_prior_strength is None else float(anchor_prior_strength)
        ),
        "local_anchor_weight_range": [
            float(local_anchor_weights.min()),
            float(local_anchor_weights.max()),
        ],
        "latent_noise_scale": float(latent_noise_scale),
        "approximate_class_covariance_fraction": float(
            np.mean(local_anchor_weights**2) * 0.60 + latent_noise_scale
        ),
        "target_balanced_within_generation_group": bool(balance_target_by_group),
        "anchor_balance": anchor_balance,
        "quantile_calibration": (
            "frozen_public_anchor_to_v6_reference"
            if frozen_quantile_calibration
            else "generated_cohort_empirical_rank_to_v6_reference"
        ),
        "fusion_weights": fusion_weights,
        "observed_calibrated_panel": observed_calibrated,
    }
    return abundance, oral_gut, provenance, metadata


def _pivot_v6_graph(graph: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows = graph.drop_duplicates(["sample_id", "node_name"])
    abundance = node_rows.pivot(index="sample_id", columns="node_name", values="abundance")
    function = node_rows.pivot(index="sample_id", columns="node_name", values="function_score")
    edge_rows = graph.assign(edge_key=graph["src"].astype(str) + " -> " + graph["dst"].astype(str))
    edges = edge_rows.pivot(index="sample_id", columns="edge_key", values="edge_weight")
    return (
        abundance.reindex(columns=PANEL_TAXA).astype(float),
        function.reindex(columns=PANEL_TAXA).astype(float),
        edges.reindex(columns=[f"{src} -> {dst}" for src, dst in EDGE_PAIRS]).astype(float),
    )


def _fit_random_forest_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    min_samples_leaf: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=256,
        min_samples_leaf=min_samples_leaf,
        max_features=0.8,
        bootstrap=True,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def _model_generated_modalities(
    v6_abundance: pd.DataFrame,
    v6_function: pd.DataFrame,
    v6_edges: pd.DataFrame,
    v6_clinical: pd.DataFrame,
    v6_metabolite: pd.DataFrame,
    generated_abundance: pd.DataFrame,
    generation_group: np.ndarray,
    observed_public_panel: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    v6_ids = v6_abundance.index.astype(str)
    clinical_index = v6_clinical.assign(sample_id=v6_clinical["sample_id"].astype(str)).set_index("sample_id")
    metabolite_index = v6_metabolite.assign(sample_id=v6_metabolite["sample_id"].astype(str)).set_index("sample_id")
    clinical_columns = ["age", "bmi", "smoking", "family_history"]
    metabolite_columns = ["bile_acids", "scfa", "tryptophan_metabolism"]
    clinical_index = clinical_index.loc[v6_ids, clinical_columns].astype(float)
    metabolite_index = metabolite_index.loc[v6_ids, metabolite_columns].astype(float)

    x_abundance = v6_abundance.to_numpy(dtype=float)
    new_abundance = generated_abundance[list(PANEL_TAXA)].to_numpy(dtype=float)
    continuous_model = _fit_random_forest_regressor(
        x_abundance,
        clinical_index[["age", "bmi"]].to_numpy(dtype=float),
        seed=seed + 1,
        min_samples_leaf=12,
    )
    continuous_prediction = continuous_model.predict(new_abundance)
    continuous_residual = (
        clinical_index[["age", "bmi"]].to_numpy(dtype=float)
        - np.asarray(continuous_model.oob_prediction_, dtype=float)
    )
    continuous_prediction += _sample_gaussian_residuals(
        rng, continuous_residual, len(generated_abundance), scale=0.80
    )
    continuous_prediction[:, 0] = np.clip(continuous_prediction[:, 0], 22.0, 84.0)
    continuous_prediction[:, 1] = np.clip(continuous_prediction[:, 1], 16.0, 39.0)

    binary_values: dict[str, np.ndarray] = {}
    binary_metrics: dict[str, float] = {}
    for offset, column in enumerate(("smoking", "family_history"), start=2):
        classifier = RandomForestClassifier(
            n_estimators=256,
            min_samples_leaf=12,
            max_features=0.8,
            bootstrap=True,
            oob_score=True,
            random_state=seed + offset,
            n_jobs=-1,
        )
        labels = clinical_index[column].astype(int).to_numpy()
        classifier.fit(x_abundance, labels)
        probability = classifier.predict_proba(new_abundance)[:, 1]
        binary_values[column] = rng.binomial(1, probability).astype(int)
        binary_metrics[f"{column}_oob_auc"] = float(
            roc_auc_score(labels, classifier.oob_decision_function_[:, 1])
        )

    clinical = pd.DataFrame(
        {
            "sample_id": generated_abundance["sample_id"].astype(str),
            "age": np.round(continuous_prediction[:, 0], 2),
            "bmi": np.round(continuous_prediction[:, 1], 2),
            "smoking": binary_values["smoking"],
            "family_history": binary_values["family_history"],
            "generation_group_id": generation_group.astype(int),
        }
    )

    x_metabolite = np.column_stack(
        [x_abundance, clinical_index[clinical_columns].to_numpy(dtype=float)]
    )
    new_x_metabolite = np.column_stack(
        [new_abundance, clinical[clinical_columns].to_numpy(dtype=float)]
    )
    metabolite_model = _fit_random_forest_regressor(
        x_metabolite,
        metabolite_index[metabolite_columns].to_numpy(dtype=float),
        seed=seed + 10,
        min_samples_leaf=10,
    )
    metabolite_prediction = metabolite_model.predict(new_x_metabolite)
    metabolite_residual = (
        metabolite_index[metabolite_columns].to_numpy(dtype=float)
        - np.asarray(metabolite_model.oob_prediction_, dtype=float)
    )
    metabolite_prediction += _sample_gaussian_residuals(
        rng, metabolite_residual, len(generated_abundance), scale=0.75
    )
    metabolite_prediction = _clip_probability(metabolite_prediction)
    metabolite = pd.DataFrame(metabolite_prediction, columns=metabolite_columns)
    metabolite.insert(0, "sample_id", generated_abundance["sample_id"].astype(str).to_numpy())

    function_x = np.column_stack(
        [
            x_abundance,
            clinical_index[clinical_columns].to_numpy(dtype=float),
            metabolite_index[metabolite_columns].to_numpy(dtype=float),
        ]
    )
    new_function_x = np.column_stack(
        [
            new_abundance,
            clinical[clinical_columns].to_numpy(dtype=float),
            metabolite[metabolite_columns].to_numpy(dtype=float),
        ]
    )
    function_model = _fit_random_forest_regressor(
        function_x,
        v6_function.to_numpy(dtype=float),
        seed=seed + 20,
        min_samples_leaf=10,
    )
    function_prediction = _clip_probability(function_model.predict(new_function_x))
    function = pd.DataFrame(function_prediction, columns=PANEL_TAXA)
    function.insert(0, "sample_id", generated_abundance["sample_id"].astype(str).to_numpy())

    observed_scaler = StandardScaler()
    observed_z = observed_scaler.fit_transform(
        _logit(observed_public_panel)
    )
    edge_method = "graphical_lasso_cv"
    try:
        association_model = GraphicalLassoCV(alphas=12, cv=5, max_iter=1000).fit(observed_z)
        precision = association_model.precision_
        selected_alpha = float(association_model.alpha_)
    except (FloatingPointError, ValueError):
        covariance = LedoitWolf().fit(observed_z).covariance_
        precision = np.linalg.pinv(covariance)
        selected_alpha = 0.0
        edge_method = "ledoit_wolf_precision_fallback"
    diagonal = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 1.0)
    generated_z = observed_scaler.transform(
        _logit(new_abundance)
    )
    edge_values = np.zeros((len(generated_abundance), len(EDGE_PAIRS)), dtype=float)
    taxon_index = {name: index for index, name in enumerate(PANEL_TAXA)}
    for edge_index, (src, dst) in enumerate(EDGE_PAIRS):
        src_index = taxon_index[src]
        dst_index = taxon_index[dst]
        signed_association = float(partial[src_index, dst_index])
        global_strength = abs(signed_association)
        local_signal = np.clip(
            0.5 + 0.5 * np.tanh(np.sign(signed_association or 1.0) * generated_z[:, src_index] * generated_z[:, dst_index] / 2.0),
            0.0,
            1.0,
        )
        edge_values[:, edge_index] = np.clip(
            0.02 + 0.96 * (0.70 * global_strength + 0.30 * local_signal),
            0.02,
            0.98,
        )
    edges = pd.DataFrame(edge_values, columns=[f"{src} -> {dst}" for src, dst in EDGE_PAIRS])
    edges.insert(0, "sample_id", generated_abundance["sample_id"].astype(str).to_numpy())

    metrics: dict[str, Any] = {
        "clinical_continuous_oob_r2": float(
            r2_score(
                clinical_index[["age", "bmi"]].to_numpy(dtype=float),
                np.asarray(continuous_model.oob_prediction_, dtype=float),
                multioutput="variance_weighted",
            )
        ),
        **binary_metrics,
        "metabolite_oob_r2": float(
            r2_score(
                metabolite_index[metabolite_columns].to_numpy(dtype=float),
                np.asarray(metabolite_model.oob_prediction_, dtype=float),
                multioutput="variance_weighted",
            )
        ),
        "function_oob_r2": float(
            r2_score(
                v6_function.to_numpy(dtype=float),
                np.asarray(function_model.oob_prediction_, dtype=float),
                multioutput="variance_weighted",
            )
        ),
        "edge_model": edge_method,
        "edge_model_alpha": selected_alpha,
        "edge_partial_correlations": {
            f"{src} -> {dst}": float(partial[PANEL_TAXA.index(src), PANEL_TAXA.index(dst)])
            for src, dst in EDGE_PAIRS
        },
    }
    return clinical, metabolite, function.join(edges.set_index("sample_id"), on="sample_id"), metrics


def _build_graph_table(
    v6_graph: pd.DataFrame,
    abundance: pd.DataFrame,
    function_and_edges: pd.DataFrame,
) -> pd.DataFrame:
    first_sample = str(v6_graph["sample_id"].iloc[0])
    template = v6_graph.loc[v6_graph["sample_id"].astype(str) == first_sample, ["node_name", "src", "dst"]]
    template = template.reset_index(drop=True)
    if list(zip(template["src"], template["dst"])) != list(EDGE_PAIRS):
        raise ValueError("The topology_v6 graph template does not match the expected ten panel edges.")

    abundance_index = abundance.set_index("sample_id")
    function_index = function_and_edges.set_index("sample_id")
    rows: list[pd.DataFrame] = []
    for sample_id in abundance["sample_id"].astype(str):
        graph = template.copy()
        graph.insert(0, "sample_id", sample_id)
        graph["abundance"] = graph["node_name"].map(abundance_index.loc[sample_id]).astype(float)
        graph["function_score"] = graph["node_name"].map(function_index.loc[sample_id]).astype(float)
        graph["edge_weight"] = [
            float(function_index.loc[sample_id, f"{src} -> {dst}"])
            for src, dst in graph[["src", "dst"]].itertuples(index=False, name=None)
        ]
        rows.append(graph)
    result = pd.concat(rows, ignore_index=True)
    return result[
        ["sample_id", "node_name", "src", "dst", "abundance", "function_score", "edge_weight"]
    ]


def _zscore_component(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    scale = np.std(array, axis=0, ddof=0)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    standardized = (array - np.mean(array, axis=0)) / scale
    return np.clip(standardized, -4.0, 4.0)


def _generate_survival_labels(
    v6_abundance: pd.DataFrame,
    v6_function: pd.DataFrame,
    v6_edges: pd.DataFrame,
    v6_clinical: pd.DataFrame,
    v6_metabolite: pd.DataFrame,
    v6_label: pd.DataFrame,
    generated_abundance: pd.DataFrame,
    generated_function_edges: pd.DataFrame,
    generated_clinical: pd.DataFrame,
    generated_metabolite: pd.DataFrame,
    generation_group: np.ndarray,
    *,
    seed: int,
    censor_location_mode: str = "realized_event_rate_calibration",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    v6_ids = v6_abundance.index.astype(str)
    clinical_columns = ["age", "bmi", "smoking", "family_history"]
    metabolite_columns = ["bile_acids", "scfa", "tryptophan_metabolism"]
    clinical_index = v6_clinical.assign(sample_id=v6_clinical["sample_id"].astype(str)).set_index("sample_id")
    metabolite_index = v6_metabolite.assign(sample_id=v6_metabolite["sample_id"].astype(str)).set_index("sample_id")
    label_index = v6_label.assign(sample_id=v6_label["sample_id"].astype(str)).set_index("sample_id")
    edge_columns = [f"{src} -> {dst}" for src, dst in EDGE_PAIRS]

    x_v6 = np.column_stack(
        [
            v6_abundance.to_numpy(dtype=float),
            clinical_index.loc[v6_ids, clinical_columns].to_numpy(dtype=float),
            metabolite_index.loc[v6_ids, metabolite_columns].to_numpy(dtype=float),
            v6_function.to_numpy(dtype=float),
            v6_edges.to_numpy(dtype=float),
        ]
    )
    generated_index = generated_function_edges.set_index("sample_id")
    x_generated = np.column_stack(
        [
            generated_abundance[list(PANEL_TAXA)].to_numpy(dtype=float),
            generated_clinical[clinical_columns].to_numpy(dtype=float),
            generated_metabolite[metabolite_columns].to_numpy(dtype=float),
            generated_index.loc[generated_abundance["sample_id"], list(PANEL_TAXA)].to_numpy(dtype=float),
            generated_index.loc[generated_abundance["sample_id"], edge_columns].to_numpy(dtype=float),
        ]
    )
    labels = label_index.loc[v6_ids]
    event_mask = labels["event"].astype(int).to_numpy() == 1
    event_model = _fit_random_forest_regressor(
        x_v6[event_mask],
        np.log(labels.loc[event_mask, "time"].to_numpy(dtype=float)),
        seed=seed + 1,
        min_samples_leaf=10,
    )
    abundance_z = _zscore_component(generated_abundance[list(PANEL_TAXA)].to_numpy(float))
    clinical_z = _zscore_component(generated_clinical[clinical_columns].to_numpy(float))
    metabolite_z = _zscore_component(generated_metabolite[metabolite_columns].to_numpy(float))
    function_z = _zscore_component(
        generated_index.loc[generated_abundance["sample_id"], list(PANEL_TAXA)].to_numpy(float)
    )
    edge_z = _zscore_component(
        generated_index.loc[generated_abundance["sample_id"], edge_columns].to_numpy(float)
    )

    node_weights = np.asarray([0.32, 0.24, 0.12, -0.18, -0.28], dtype=float)
    clinical_weights = np.asarray([0.38, 0.27, 0.19, 0.16], dtype=float)
    metabolite_weights = np.asarray([0.38, -0.40, 0.22], dtype=float)
    function_weights = np.asarray([0.30, 0.20, 0.08, -0.16, -0.24], dtype=float)
    node_weight_lookup = dict(zip(PANEL_TAXA, node_weights.tolist()))
    edge_weights = np.asarray(
        [
            (node_weight_lookup[src] + node_weight_lookup[dst]) / 2.0
            for src, dst in EDGE_PAIRS
        ],
        dtype=float,
    )

    component_names = [
        "v6_teacher",
        "microbiome",
        "clinical",
        "metabolite",
        "function",
        "topology",
    ]
    teacher_risk = -np.asarray(event_model.predict(x_generated), dtype=float)
    raw_components = np.column_stack(
        [
            teacher_risk,
            abundance_z @ node_weights,
            clinical_z @ clinical_weights,
            metabolite_z @ metabolite_weights,
            function_z @ function_weights,
            edge_z @ edge_weights,
        ]
    )
    standardized_components = _zscore_component(raw_components)
    component_weights = np.asarray([0.30, 0.24, 0.16, 0.14, 0.09, 0.07], dtype=float)
    latent_risk = _zscore_component(standardized_components @ component_weights).ravel()

    event_signal_scale = 0.35
    event_noise_to_signal_ratio = 1.25
    event_noise_sd = event_signal_scale * event_noise_to_signal_ratio
    event_standard_normal = rng.normal(0.0, 1.0, size=len(x_generated))
    censor_standard_normal = rng.normal(0.0, 1.0, size=len(x_generated))
    event_log_time = (
        np.log(78.0)
        - event_signal_scale * latent_risk
        + event_noise_sd * event_standard_normal
    )
    event_time = np.exp(event_log_time)

    target_event_rate = float(labels["event"].mean())
    censor_noise_sd = 0.55
    if censor_location_mode == "realized_event_rate_calibration":
        target_event_count = int(round(target_event_rate * len(event_time)))
        censor_location_thresholds = np.sort(
            np.log(event_time) - censor_noise_sd * censor_standard_normal
        )
        if target_event_count <= 0 or target_event_count >= len(event_time):
            raise RuntimeError("Target event count must leave both event and censored samples.")
        censor_location = 0.5 * (
            censor_location_thresholds[target_event_count - 1]
            + censor_location_thresholds[target_event_count]
        )
    elif censor_location_mode == "analytic_prior_calibration":
        difference_sd = float(
            np.sqrt(event_signal_scale**2 + event_noise_sd**2 + censor_noise_sd**2)
        )
        censor_location = float(
            np.log(78.0) + NormalDist().inv_cdf(target_event_rate) * difference_sd
        )
    else:
        raise ValueError(f"Unsupported censor_location_mode: {censor_location_mode}")
    censor_time = np.exp(censor_location + censor_noise_sd * censor_standard_normal)
    events = (event_time <= censor_time).astype(int)
    observed_time = np.where(events == 1, event_time, censor_time)
    observed_time = np.clip(np.rint(observed_time), 6.0, 132.0).astype(int)
    result = pd.DataFrame(
        {
            "sample_id": generated_abundance["sample_id"].astype(str),
            "time": observed_time,
            "event": events,
        }
    )
    deterministic_c_index = float(concordance_index(observed_time, events, latent_risk))
    realized_oracle_c_index = float(concordance_index(observed_time, events, -event_time))
    group_c_indices = {
        str(int(group_id)): float(
            concordance_index(
                observed_time[np.asarray(generation_group) == group_id],
                events[np.asarray(generation_group) == group_id],
                latent_risk[np.asarray(generation_group) == group_id],
            )
        )
        for group_id in sorted(np.unique(generation_group).tolist())
    }
    censor_risk_correlation = float(np.corrcoef(latent_risk, np.log(censor_time))[0, 1])
    signal_sd = float(np.std(event_signal_scale * latent_risk, ddof=0))
    audit = pd.DataFrame(
        {
            "sample_id": generated_abundance["sample_id"].astype(str),
            **{
                f"survival_component_{name}": standardized_components[:, index]
                for index, name in enumerate(component_names)
            },
            "survival_latent_risk": latent_risk,
            "survival_event_noise": event_noise_sd * event_standard_normal,
            "survival_event_time": event_time,
            "survival_censor_time": censor_time,
        }
    )
    if censor_location_mode == "analytic_prior_calibration":
        audit["survival_censor_noise"] = censor_noise_sd * censor_standard_normal
    metrics = {
        "v6_teacher_event_time_oob_r2": float(
            r2_score(
                np.log(labels.loc[event_mask, "time"].to_numpy(dtype=float)),
                np.asarray(event_model.oob_prediction_, dtype=float),
            )
        ),
        "latent_risk_method": "standardized_weighted_multimodal_score_with_v6_event_teacher",
        "latent_risk_component_weights": dict(zip(component_names, component_weights.tolist())),
        "node_abundance_weights": dict(zip(PANEL_TAXA, node_weights.tolist())),
        "clinical_weights": dict(zip(clinical_columns, clinical_weights.tolist())),
        "metabolite_weights": dict(zip(metabolite_columns, metabolite_weights.tolist())),
        "function_weights": dict(zip(PANEL_TAXA, function_weights.tolist())),
        "event_time_model": "log_normal_aft_with_controlled_signal_to_noise",
        "event_signal_scale": event_signal_scale,
        "event_signal_log_time_sd": signal_sd,
        "event_noise_log_time_sd": event_noise_sd,
        "event_noise_to_signal_sd_ratio": event_noise_sd / max(signal_sd, 1e-12),
        "event_noise_sampling": "independent_pseudorandom_normal",
        "censoring_model": (
            "feature_independent_log_normal_with_analytic_prior_rate_calibration"
            if censor_location_mode == "analytic_prior_calibration"
            else "feature_independent_log_normal_with_realized_global_rate_calibration"
        ),
        "censor_location_mode": censor_location_mode,
        "censor_log_time_location": censor_location,
        "censor_log_time_sd": censor_noise_sd,
        "censor_noise_sampling": "independent_pseudorandom_normal",
        "generation_group_used_for_outcome_generation": False,
        "generation_group_used_for_outcome_audit_only": True,
        "censor_log_time_latent_risk_correlation": censor_risk_correlation,
        "target_event_rate_from_v6": target_event_rate,
        "generated_event_rate": float(result["event"].mean()),
        "deterministic_latent_risk_c_index": deterministic_c_index,
        "realized_hidden_event_time_oracle_c_index": realized_oracle_c_index,
        "generation_group_latent_risk_c_index": group_c_indices,
        "minimum_generation_group_latent_risk_c_index": min(group_c_indices.values()),
        "generation_group_latent_risk_c_index_spread": (
            max(group_c_indices.values()) - min(group_c_indices.values())
        ),
        "validation_target_band": {
            "deterministic_latent_risk_c_index": [0.72, 0.80],
            "minimum_generation_group_latent_risk_c_index": 0.68,
            "event_noise_to_signal_sd_ratio": [1.10, 1.40],
            "absolute_censor_risk_correlation_maximum": 0.08,
        },
        "generated_time_min": int(result["time"].min()),
        "generated_time_median": float(result["time"].median()),
        "generated_time_max": int(result["time"].max()),
        "semantics": "fully model-generated right-censored development proxy; not observed follow-up",
    }
    return result, metrics, audit


def _archive_v6(sources: SourcePaths, archive_dir: Path) -> dict[str, str]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: dict[str, str] = {}
    for source in (sources.v6_graph, sources.v6_clinical, sources.v6_metabolite, sources.v6_label):
        destination = archive_dir / source.name
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise FileExistsError(f"Archive destination differs from its source: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        archived[destination.relative_to(PROJECT_ROOT).as_posix()] = _sha256(destination)
    return archived


def _archive_previous_v7(output_dir: Path, archive_dir: Path) -> dict[str, str]:
    outputs = _resolve_paths(output_dir)
    if not outputs.manifest.exists():
        return {}
    existing_manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    existing_version = str(existing_manifest.get("generator_version", "unknown"))
    if existing_version == GENERATOR_VERSION:
        archive_manifest_path = archive_dir / "archive_manifest.json"
        if not archive_manifest_path.exists():
            return {}
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        if archive_manifest.get("generator_version") != PREVIOUS_GENERATOR_VERSION:
            raise RuntimeError(
                f"Unexpected topology_v7 archive version: {archive_manifest_path}"
            )
        archived = {
            str(path): str(digest) for path, digest in archive_manifest.get("files", {}).items()
        }
        for relative_path, expected_hash in archived.items():
            archived_path = PROJECT_ROOT / relative_path
            if not archived_path.exists() or _sha256(archived_path) != expected_hash:
                raise RuntimeError(f"Archived topology_v7 file failed verification: {archived_path}")
        return archived
    if existing_version != PREVIOUS_GENERATOR_VERSION:
        raise RuntimeError(
            f"Refusing to overwrite unrecognized topology_v7 generator version: {existing_version}"
        )

    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: dict[str, str] = {}
    for source in outputs.__dict__.values():
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Existing topology_v7 archive source is missing: {source}")
        destination = archive_dir / source.name
        source_hash = _sha256(source)
        if destination.exists() and _sha256(destination) != source_hash:
            raise FileExistsError(f"Archive destination differs from its source: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        archived[destination.relative_to(PROJECT_ROOT).as_posix()] = source_hash

    archive_manifest = {
        "schema_version": 1,
        "dataset_version": "topology_v7",
        "generator_version": existing_version,
        "archive_reason": (
            "Preserved before replacing the residual-dominated survival generator with the "
            "controlled-signal generator_v2."
        ),
        "local_files_preserved": True,
        "files": archived,
    }
    manifest_path = archive_dir / "archive_manifest.json"
    manifest_text = json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise FileExistsError(f"Archive manifest differs from expected content: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text, encoding="utf-8")
    readme_path = archive_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# topology_v7 generator_v1 archive\n\n"
            "This directory preserves the original topology_v7 files before the survival-label "
            "generator was repaired. The archived generator_v1 used residual-dominated event and "
            "censor-time models and is retained only for reproducibility and rollback.\n",
            encoding="utf-8",
        )
    return archived


def _quality_report(
    graph: pd.DataFrame,
    clinical: pd.DataFrame,
    metabolite: pd.DataFrame,
    label: pd.DataFrame,
    provenance: pd.DataFrame,
) -> dict[str, Any]:
    node = graph.drop_duplicates(["sample_id", "node_name"]).pivot(
        index="sample_id", columns="node_name", values="abundance"
    )
    unique_vectors = int(len(node.drop_duplicates()))
    group_anchor_sets = {
        int(group_id): sorted(
            set(frame["primary_anchor_patient_id"]).union(frame["secondary_anchor_patient_id"])
        )
        for group_id, frame in provenance.groupby("generation_group_id")
    }
    group_pairs = list(group_anchor_sets.items())
    overlaps = []
    for left_index, (left_group, left_anchors) in enumerate(group_pairs):
        for right_group, right_anchors in group_pairs[left_index + 1 :]:
            shared = sorted(set(left_anchors).intersection(right_anchors))
            if shared:
                overlaps.append({"left": left_group, "right": right_group, "anchors": shared})
    return {
        "num_samples": int(node.shape[0]),
        "graph_rows": int(len(graph)),
        "unique_abundance_vectors": unique_vectors,
        "exact_duplicate_abundance_vectors": int(len(node) - unique_vectors),
        "finite_values": bool(
            np.isfinite(
                np.concatenate(
                    [
                        graph[["abundance", "function_score", "edge_weight"]].to_numpy(float).ravel(),
                        clinical.drop(columns="sample_id").to_numpy(float).ravel(),
                        metabolite.drop(columns="sample_id").to_numpy(float).ravel(),
                        label[["time", "event"]].to_numpy(float).ravel(),
                    ]
                )
            ).all()
        ),
        "generation_group_sizes": {
            str(int(key)): int(value)
            for key, value in provenance["generation_group_id"].value_counts().sort_index().items()
        },
        "anchor_overlap_between_generation_groups": overlaps,
        "event_rate": float(label["event"].mean()),
        "ranges": {
            "abundance": [float(graph["abundance"].min()), float(graph["abundance"].max())],
            "function_score": [float(graph["function_score"].min()), float(graph["function_score"].max())],
            "edge_weight": [float(graph["edge_weight"].min()), float(graph["edge_weight"].max())],
            "age": [float(clinical["age"].min()), float(clinical["age"].max())],
            "bmi": [float(clinical["bmi"].min()), float(clinical["bmi"].max())],
            "metabolites": [
                float(metabolite.drop(columns="sample_id").min().min()),
                float(metabolite.drop(columns="sample_id").max().max()),
            ],
            "time": [int(label["time"].min()), int(label["time"].max())],
        },
    }


def _resolve_paths(output_dir: Path) -> OutputPaths:
    return OutputPaths(
        graph=output_dir / "topology_v7_sample_graph_table.csv",
        clinical=output_dir / "topology_v7_sample_clinical_table.csv",
        metabolite=output_dir / "topology_v7_sample_metabolite_table.csv",
        label=output_dir / "topology_v7_sample_label_table.csv",
        oral_gut=output_dir / "topology_v7_sample_oral_gut_table.csv",
        provenance=output_dir / "topology_v7_sample_provenance.csv",
        manifest=output_dir / "topology_v7_manifest.json",
    )


def build_topology_v7(
    *,
    sources: SourcePaths,
    output_dir: Path,
    archive_dir: Path,
    previous_v7_archive_dir: Path | None = None,
    sample_count: int = 3600,
    seed: int = 20260720,
) -> dict[str, Any]:
    if sample_count < 100:
        raise ValueError("topology_v7 requires at least 100 generated samples.")
    for path in sources.__dict__.values():
        if not Path(path).exists():
            raise FileNotFoundError(path)

    if previous_v7_archive_dir is None:
        previous_v7_archive_dir = PROJECT_ROOT / "archive/datasets/topology_v7_generator_v1"
    previous_v7_archive = _archive_previous_v7(output_dir, previous_v7_archive_dir)
    archived = _archive_v6(sources, archive_dir)
    public = pd.read_csv(sources.public_features)
    v6_graph = pd.read_csv(sources.v6_graph)
    v6_clinical = pd.read_csv(sources.v6_clinical)
    v6_metabolite = pd.read_csv(sources.v6_metabolite)
    v6_label = pd.read_csv(sources.v6_label)
    v6_abundance, v6_function, v6_edges = _pivot_v6_graph(v6_graph)

    abundance, oral_gut, provenance, microbiome_metrics = _generate_microbiome(
        public,
        v6_abundance,
        sample_count=sample_count,
        seed=seed,
        generation_groups=5,
        max_features_per_site=96,
        latent_components=8,
    )
    observed_public_panel = np.asarray(microbiome_metrics.pop("observed_calibrated_panel"), dtype=float)
    clinical, metabolite, function_edges, modality_metrics = _model_generated_modalities(
        v6_abundance,
        v6_function,
        v6_edges,
        v6_clinical,
        v6_metabolite,
        abundance,
        provenance["generation_group_id"].to_numpy(dtype=int),
        observed_public_panel,
        seed=seed + 100,
    )
    graph = _build_graph_table(v6_graph, abundance, function_edges)
    label, label_metrics, survival_audit = _generate_survival_labels(
        v6_abundance,
        v6_function,
        v6_edges,
        v6_clinical,
        v6_metabolite,
        v6_label,
        abundance,
        function_edges,
        clinical,
        metabolite,
        provenance["generation_group_id"].to_numpy(dtype=int),
        seed=seed + 200,
    )
    provenance = provenance.merge(survival_audit, on="sample_id", how="inner", validate="one_to_one")
    provenance["label_source"] = "controlled_multimodal_aft_survival_proxy"

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _resolve_paths(output_dir)
    quality = _quality_report(graph, clinical, metabolite, label, provenance)
    if quality["num_samples"] != sample_count:
        raise RuntimeError("Generated sample count does not match the requested cohort size.")
    if quality["exact_duplicate_abundance_vectors"] != 0:
        raise RuntimeError("Generated microbiome panel contains exact duplicate vectors.")
    if quality["anchor_overlap_between_generation_groups"]:
        raise RuntimeError("Public anchors overlap between generation groups.")
    if not quality["finite_values"]:
        raise RuntimeError("Generated tables contain non-finite values.")
    target_band = label_metrics["validation_target_band"]
    deterministic_c_index = float(label_metrics["deterministic_latent_risk_c_index"])
    if not (
        float(target_band["deterministic_latent_risk_c_index"][0])
        <= deterministic_c_index
        <= float(target_band["deterministic_latent_risk_c_index"][1])
    ):
        raise RuntimeError("Generated survival signal is outside its declared C-index band.")
    if float(label_metrics["minimum_generation_group_latent_risk_c_index"]) < float(
        target_band["minimum_generation_group_latent_risk_c_index"]
    ):
        raise RuntimeError("At least one generation group has insufficient survival signal.")
    noise_ratio = float(label_metrics["event_noise_to_signal_sd_ratio"])
    if not (
        float(target_band["event_noise_to_signal_sd_ratio"][0])
        <= noise_ratio
        <= float(target_band["event_noise_to_signal_sd_ratio"][1])
    ):
        raise RuntimeError("Generated event-time noise ratio is outside its declared band.")
    if abs(float(label_metrics["censor_log_time_latent_risk_correlation"])) > float(
        target_band["absolute_censor_risk_correlation_maximum"]
    ):
        raise RuntimeError("Generated censoring is too strongly associated with latent risk.")

    graph.to_csv(outputs.graph, index=False, float_format="%.8f")
    clinical.to_csv(outputs.clinical, index=False)
    metabolite.to_csv(outputs.metabolite, index=False, float_format="%.8f")
    label.to_csv(outputs.label, index=False)
    oral_gut.to_csv(outputs.oral_gut, index=False, float_format="%.10f")
    provenance.to_csv(outputs.provenance, index=False, float_format="%.8f")

    source_hashes = {
        Path(path).relative_to(PROJECT_ROOT).as_posix(): _sha256(Path(path))
        for path in sources.__dict__.values()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "topology_v7",
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "sample_count": int(sample_count),
        "observed_real_patient_count": int(len(public)),
        "observed_real_patient_rows_in_v7": 0,
        "dataset_class": "hybrid_model_generated_development_cohort",
        "sources": {
            "public_microbiome_anchor": "russo_crc_oral_gut_2023",
            "missing_modality_prior": "topology_v6",
            "sha256": source_hashes,
        },
        "generation": {
            "microbiome": {
                "method": "class_conditional_pca_local_gaussian_on_paired_saliva_stool_compositions",
                **microbiome_metrics,
            },
            "clinical_and_metabolite": {
                "method": "random_forest_conditional_prediction_with_oob_residual_sampling",
                **modality_metrics,
            },
            "function_score": "random_forest_prediction_from_v6_prior_targets",
            "edge_weight": "public_graphical_model_partial_association_with_sample_specific_modulation",
            "survival": {
                "method": "controlled_multimodal_log_normal_aft_with_independent_censoring",
                **label_metrics,
            },
        },
        "quality": quality,
        "archive": archived,
        "previous_v7_archive": previous_v7_archive,
        "prohibited_model_features": [
            column for column in survival_audit.columns if column != "sample_id"
        ],
        "limitations": [
            "All 3600 v7 rows are model-generated; they are not 3600 observed patients.",
            "Only the paired oral-gut microbiome distribution is anchored in an open real cohort of 42 patients.",
            "Clinical variables, metabolites, function scores, and right-censored outcomes inherit synthetic topology_v6 priors.",
            "Generated survival labels are development proxies and cannot support clinical efficacy, prognosis, or external-validation claims.",
            "The survival signal-to-noise band is a transparent generator design constraint, not a clinical performance claim.",
            "Performance measured on v7 evaluates recovery of the generator, not independent clinical generalization.",
            "The source public cohort remains the external real-data validation dataset and must be evaluated separately.",
        ],
    }
    manifest["outputs"] = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in (
            outputs.graph,
            outputs.clinical,
            outputs.metabolite,
            outputs.label,
            outputs.oral_gut,
            outputs.provenance,
        )
    }
    outputs.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _default_sources() -> SourcePaths:
    return SourcePaths(
        public_features=PROJECT_ROOT
        / "data/public/russo_crc_oral_gut_2023/processed/paired_patient_features.csv",
        v6_graph=PROJECT_ROOT / "data/research/topology_v6_sample_graph_table.csv",
        v6_clinical=PROJECT_ROOT / "data/research/topology_v6_sample_clinical_table.csv",
        v6_metabolite=PROJECT_ROOT / "data/research/topology_v6_sample_metabolite_table.csv",
        v6_label=PROJECT_ROOT / "data/research/topology_v6_sample_label_table.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the topology_v7 hybrid generated cohort.")
    parser.add_argument("--samples", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/research")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=PROJECT_ROOT / "archive/datasets/topology_v6",
    )
    parser.add_argument(
        "--previous-v7-archive-dir",
        type=Path,
        default=PROJECT_ROOT / "archive/datasets/topology_v7_generator_v1",
    )
    args = parser.parse_args()
    manifest = build_topology_v7(
        sources=_default_sources(),
        output_dir=args.output_dir.resolve(),
        archive_dir=args.archive_dir.resolve(),
        previous_v7_archive_dir=args.previous_v7_archive_dir.resolve(),
        sample_count=args.samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "dataset_version": manifest["dataset_version"],
                "sample_count": manifest["sample_count"],
                "quality": manifest["quality"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
