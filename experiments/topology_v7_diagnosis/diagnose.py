from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from experiments.temporal_independent_v3.topology_aft_fusion import (
    AFT_PRESETS,
    _impute_from_train,
    _make_aft_dmatrix,
    _train_aft_candidate,
    build_topology_fingerprint_dataframe,
    select_feature_set,
)
from experiments.public_data_v1.build_topology_v7 import (
    EDGE_PAIRS,
    PANEL_TAXA,
    _fit_random_forest_regressor,
    _pivot_v6_graph,
)
from research.baseline_compare import prepare_split_data, train_tabular_cox
from research.data import build_dataset_from_csv, split_sample_table
from research.ensemble_v2 import build_model
from research.metrics import concordance_index


GROUP_COLUMN = "generation_group_id"
ID_COLUMNS = {"sample_id", "time", "event", GROUP_COLUMN}


@dataclass(frozen=True)
class FrameSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    summary: dict[str, Any]


def _as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_as_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _random_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> FrameSplit:
    stratify = frame["event"].astype(int) if frame["event"].nunique() > 1 else None
    train_val, test = train_test_split(
        frame,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )
    effective_val_ratio = val_ratio / (1.0 - test_ratio)
    inner_stratify = train_val["event"].astype(int) if train_val["event"].nunique() > 1 else None
    train, val = train_test_split(
        train_val,
        test_size=effective_val_ratio,
        random_state=seed + 1,
        stratify=inner_stratify,
    )
    summary = {
        "split_seed": int(seed),
        "split_strategy": "diagnostic_random_event_stratified_train_val_test_split",
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "test_size": int(len(test)),
        "train_groups": sorted(train[GROUP_COLUMN].astype(int).unique().tolist()),
        "val_groups": sorted(val[GROUP_COLUMN].astype(int).unique().tolist()),
        "test_groups": sorted(test[GROUP_COLUMN].astype(int).unique().tolist()),
    }
    return FrameSplit(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        summary=summary,
    )


def _group_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> FrameSplit:
    train, val, test, summary = split_sample_table(
        frame,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if summary["split_strategy"] != "generation_group_disjoint_train_val_test_split":
        raise RuntimeError("The controlled group split did not preserve generation groups.")
    return FrameSplit(train=train, val=val, test=test, summary=summary)


def _feature_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    frame, all_columns, metadata = build_topology_fingerprint_dataframe(config)
    if GROUP_COLUMN not in frame.columns:
        raise ValueError("topology_v7 diagnosis requires generation_group_id.")

    deployable_columns = [column for column in all_columns if column not in ID_COLUMNS]
    leaked = sorted(ID_COLUMNS.intersection(deployable_columns))
    if leaked:
        raise RuntimeError(f"Metadata or labels leaked into features: {leaked}")

    clinical_metabolite = list(config["model"]["clinical_columns"]) + list(
        config["model"]["metabolite_columns"]
    )
    edge_identity = select_feature_set(deployable_columns, config, "edge_identity")
    topology_only = select_feature_set(deployable_columns, config, "topology_only")
    feature_sets = {
        "clinical_metabolite": clinical_metabolite,
        "edge_identity": edge_identity,
        "full_topology": deployable_columns,
        "topology_only": topology_only,
    }
    for name, columns in feature_sets.items():
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Feature set {name} is missing columns: {missing[:5]}")
        if GROUP_COLUMN in columns:
            raise RuntimeError(f"generation_group_id leaked into {name}.")

    metadata = copy.deepcopy(metadata)
    metadata["feature_columns"] = deployable_columns
    metadata["num_features"] = len(deployable_columns)
    metadata["excluded_metadata_columns"] = sorted(ID_COLUMNS.intersection(frame.columns))
    return frame, feature_sets, metadata


def _cox_run(
    split: FrameSplit,
    feature_columns: list[str],
    *,
    model_type: str,
    model_seed: int,
    device: str,
) -> dict[str, Any]:
    prepared = prepare_split_data(
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        feature_columns=feature_columns,
        num_time_bins=12,
    )
    started = time.perf_counter()
    model, val_metrics, test_metrics = train_tabular_cox(
        split=prepared,
        model_type=model_type,
        hidden_dim=64,
        dropout=0.20,
        lr=0.001 if model_type == "mlp" else 0.01,
        weight_decay=0.0001,
        epochs=500 if model_type == "mlp" else 800,
        patience=50 if model_type == "mlp" else 80,
        min_delta=0.0001,
        seed=model_seed,
        device=device,
    )
    model.eval()
    model_device = next(model.parameters()).device
    with torch.no_grad():
        train_risk = model(
            torch.as_tensor(prepared.X_train, dtype=torch.float32, device=model_device)
        ).detach().cpu().numpy()
    history = val_metrics["history"]
    return {
        "train_c_index": float(
            concordance_index(prepared.time_train, prepared.event_train, train_risk)
        ),
        "validation_c_index": float(val_metrics["c_index"]),
        "test_c_index": float(test_metrics["c_index"]),
        "best_epoch": int(max(history, key=lambda row: row["val_c_index"])["epoch"]),
        "epochs_run": int(len(history)),
        "test_loss": float(test_metrics["loss"]),
        "seconds": float(time.perf_counter() - started),
    }


def _aft_run(
    split: FrameSplit,
    feature_columns: list[str],
    *,
    model_seed: int,
) -> dict[str, Any]:
    train_x, val_x, test_x, _ = _impute_from_train(
        split.train,
        split.val,
        split.test,
        feature_columns,
    )
    labels = {
        "train": (
            split.train["time"].to_numpy(float),
            split.train["event"].to_numpy(float),
        ),
        "val": (
            split.val["time"].to_numpy(float),
            split.val["event"].to_numpy(float),
        ),
        "test": (
            split.test["time"].to_numpy(float),
            split.test["event"].to_numpy(float),
        ),
    }
    matrices = {
        name: _make_aft_dmatrix(values, *labels[name], feature_columns)
        for name, values in {"train": train_x, "val": val_x, "test": test_x}.items()
    }
    started = time.perf_counter()
    _, result, _ = _train_aft_candidate(
        name="balanced_normal",
        preset=AFT_PRESETS["balanced_normal"],
        matrices=matrices,
        labels=labels,
        seed=model_seed,
        num_boost_round=600,
        early_stopping_rounds=40,
        nthread=6,
    )
    metrics = result["metrics"]
    return {
        "train_c_index": float(metrics["train"]["c_index"]),
        "validation_c_index": float(metrics["val"]["c_index"]),
        "test_c_index": float(metrics["test"]["c_index"]),
        "best_iteration": int(result["best_iteration"]),
        "validation_aft_nloglik": float(metrics["val"]["aft_nloglik"]),
        "test_aft_nloglik": float(metrics["test"]["aft_nloglik"]),
        "seconds": float(time.perf_counter() - started),
    }


def _domain_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    train_values = train[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    test_values = test[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    combined = pd.concat([train_values, test_values], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train_values)), np.ones(len(test_values))])
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = cross_val_predict(pipeline, combined, domain, cv=folds, method="predict_proba")[:, 1]

    medians = train_values.median(axis=0).fillna(0.0)
    train_imputed = train_values.fillna(medians)
    test_imputed = test_values.fillna(medians)
    shifts = []
    for column in feature_columns:
        left = train_imputed[column].to_numpy(float)
        right = test_imputed[column].to_numpy(float)
        pooled = math.sqrt((float(np.var(left)) + float(np.var(right))) / 2.0)
        smd = abs(float(np.mean(left) - np.mean(right))) / max(pooled, 1e-12)
        wd = float(wasserstein_distance(left, right)) / max(float(np.std(left)), 1e-12)
        shifts.append({"feature": column, "absolute_smd": smd, "normalized_wasserstein": wd})
    shifts.sort(key=lambda row: row["absolute_smd"], reverse=True)
    return {
        "domain_classifier_auc": float(roc_auc_score(domain, probability)),
        "top_feature_shifts": shifts[:10],
    }


def _generator_oracle_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the v7 outcome generator without changing any dataset file."""
    manifest_path = Path(config["paths"]["manifest_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generator_version") == "topology_v7_hybrid_generator_v2":
        return _generator_v2_oracle_audit(config, manifest)
    generator_seed = int(manifest["seed"]) + 200
    v6_root = Path("archive/datasets/topology_v6")
    paths = {
        "v6_graph": v6_root / "topology_v6_sample_graph_table.csv",
        "v6_clinical": v6_root / "topology_v6_sample_clinical_table.csv",
        "v6_metabolite": v6_root / "topology_v6_sample_metabolite_table.csv",
        "v6_label": v6_root / "topology_v6_sample_label_table.csv",
        "v7_graph": Path(config["paths"]["graph_csv"]),
        "v7_clinical": Path(config["paths"]["clinical_csv"]),
        "v7_metabolite": Path(config["paths"]["metabolite_csv"]),
        "v7_label": Path(config["paths"]["label_csv"]),
    }
    missing = [str(path.as_posix()) for path in paths.values() if not path.exists()]
    if missing:
        return {"available": False, "missing_paths": missing}

    v6_graph = pd.read_csv(paths["v6_graph"])
    v7_graph = pd.read_csv(paths["v7_graph"])
    v6_abundance, v6_function, v6_edges = _pivot_v6_graph(v6_graph)
    v7_abundance, v7_function, v7_edges = _pivot_v6_graph(v7_graph)
    v6_ids = v6_abundance.index.astype(str)
    clinical_columns = ["age", "bmi", "smoking", "family_history"]
    metabolite_columns = ["bile_acids", "scfa", "tryptophan_metabolism"]
    edge_columns = [f"{src} -> {dst}" for src, dst in EDGE_PAIRS]

    def indexed(path: Path) -> pd.DataFrame:
        value = pd.read_csv(path)
        value["sample_id"] = value["sample_id"].astype(str)
        return value.set_index("sample_id")

    v6_clinical = indexed(paths["v6_clinical"])
    v6_metabolite = indexed(paths["v6_metabolite"])
    v6_label = indexed(paths["v6_label"])
    v7_clinical = indexed(paths["v7_clinical"])
    v7_metabolite = indexed(paths["v7_metabolite"])
    v7_label = indexed(paths["v7_label"])
    v7_ids = v7_clinical.index.astype(str)
    v7_abundance = v7_abundance.reindex(v7_ids)
    v7_function = v7_function.reindex(v7_ids)
    v7_edges = v7_edges.reindex(v7_ids)

    x_v6 = np.column_stack(
        [
            v6_abundance.to_numpy(float),
            v6_clinical.loc[v6_ids, clinical_columns].to_numpy(float),
            v6_metabolite.loc[v6_ids, metabolite_columns].to_numpy(float),
            v6_function.to_numpy(float),
            v6_edges.to_numpy(float),
        ]
    )
    x_v7 = np.column_stack(
        [
            v7_abundance.to_numpy(float),
            v7_clinical.loc[v7_ids, clinical_columns].to_numpy(float),
            v7_metabolite.loc[v7_ids, metabolite_columns].to_numpy(float),
            v7_function.to_numpy(float),
            v7_edges.loc[v7_ids, edge_columns].to_numpy(float),
        ]
    )
    feature_names = [
        *[f"abundance:{name}" for name in PANEL_TAXA],
        *[f"clinical:{name}" for name in clinical_columns],
        *[f"metabolite:{name}" for name in metabolite_columns],
        *[f"function:{name}" for name in PANEL_TAXA],
        *[f"edge:{name}" for name in edge_columns],
    ]
    labels = v6_label.loc[v6_ids]
    event_mask = labels["event"].astype(int).to_numpy() == 1
    censor_mask = ~event_mask
    event_model = _fit_random_forest_regressor(
        x_v6[event_mask],
        np.log(labels.loc[event_mask, "time"].to_numpy(float)),
        seed=generator_seed + 1,
        min_samples_leaf=10,
    )
    censor_model = _fit_random_forest_regressor(
        x_v6[censor_mask],
        np.log(labels.loc[censor_mask, "time"].to_numpy(float)),
        seed=generator_seed + 2,
        min_samples_leaf=10,
    )
    event_residual = (
        np.log(labels.loc[event_mask, "time"].to_numpy(float))
        - np.asarray(event_model.oob_prediction_, dtype=float)
    )
    censor_residual = (
        np.log(labels.loc[censor_mask, "time"].to_numpy(float))
        - np.asarray(censor_model.oob_prediction_, dtype=float)
    )
    event_prediction = np.asarray(event_model.predict(x_v7), dtype=float)
    censor_prediction = np.asarray(censor_model.predict(x_v7), dtype=float)
    event_noise_sd = max(float(np.std(event_residual)), 1e-3) * 0.80
    censor_noise_sd = max(float(np.std(censor_residual)), 1e-3) * 0.80

    rng = np.random.default_rng(generator_seed)
    event_noise = rng.normal(0.0, event_noise_sd, size=len(x_v7))
    censor_noise = rng.normal(0.0, censor_noise_sd, size=len(x_v7))
    event_time = np.exp(event_prediction + event_noise)
    censor_time = np.exp(censor_prediction + censor_noise)
    target_event_count = int(round(float(labels["event"].mean()) * len(x_v7)))
    ratios = event_time / np.maximum(censor_time, 1e-8)
    threshold = float(np.partition(ratios, target_event_count - 1)[target_event_count - 1])
    event_time *= 1.0 / max(threshold, 1e-8)
    reconstructed_event = (event_time <= censor_time).astype(int)
    if int(reconstructed_event.sum()) != target_event_count:
        order = np.argsort(ratios, kind="stable")
        reconstructed_event[:] = 0
        reconstructed_event[order[:target_event_count]] = 1
    reconstructed_time = np.clip(
        np.rint(np.where(reconstructed_event == 1, event_time, censor_time)), 6.0, 132.0
    ).astype(int)

    saved = v7_label.loc[v7_ids]
    saved_time = saved["time"].to_numpy(float)
    saved_event = saved["event"].to_numpy(float)
    v6_min = np.min(x_v6, axis=0)
    v6_max = np.max(x_v6, axis=0)
    outside = (x_v7 < v6_min) | (x_v7 > v6_max)
    outside_by_feature = sorted(
        [
            {"feature": name, "outside_v6_range_fraction": float(outside[:, index].mean())}
            for index, name in enumerate(feature_names)
        ],
        key=lambda row: row["outside_v6_range_fraction"],
        reverse=True,
    )
    event_signal_sd = float(np.std(event_prediction))
    censor_signal_sd = float(np.std(censor_prediction))
    return {
        "available": True,
        "generator_seed": generator_seed,
        "v6_training_samples": int(len(x_v6)),
        "v7_generated_samples": int(len(x_v7)),
        "deterministic_event_time_risk_c_index": float(
            concordance_index(saved_time, saved_event, -event_prediction)
        ),
        "deterministic_event_vs_censor_margin_c_index": float(
            concordance_index(saved_time, saved_event, censor_prediction - event_prediction)
        ),
        "realized_hidden_event_time_oracle_c_index": float(
            concordance_index(saved_time, saved_event, -event_time)
        ),
        "event_prediction_log_time_sd": event_signal_sd,
        "event_random_residual_log_time_sd": event_noise_sd,
        "event_noise_to_signal_sd_ratio": event_noise_sd / max(event_signal_sd, 1e-12),
        "censor_prediction_log_time_sd": censor_signal_sd,
        "censor_random_residual_log_time_sd": censor_noise_sd,
        "censor_noise_to_signal_sd_ratio": censor_noise_sd / max(censor_signal_sd, 1e-12),
        "reconstructed_event_agreement": float(
            np.mean(reconstructed_event == saved_event.astype(int))
        ),
        "reconstructed_time_exact_agreement": float(np.mean(reconstructed_time == saved_time)),
        "reconstructed_time_mean_absolute_error": float(
            np.mean(np.abs(reconstructed_time - saved_time))
        ),
        "v7_values_outside_v6_feature_range_fraction": float(outside.mean()),
        "top_outside_v6_range_features": outside_by_feature[:10],
        "interpretation": (
            "The deterministic risk is available to a model; the realized hidden-event oracle also "
            "contains the generator's random residual and is not available from input features."
        ),
    }


def _generator_v2_oracle_audit(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    provenance_path = Path(config["paths"]["provenance_csv"])
    label_path = Path(config["paths"]["label_csv"])
    provenance = pd.read_csv(provenance_path)
    labels = pd.read_csv(label_path)
    required = {
        "sample_id",
        "generation_group_id",
        "survival_latent_risk",
        "survival_event_noise",
        "survival_event_time",
        "survival_censor_time",
    }
    missing = sorted(required.difference(provenance.columns))
    if missing:
        return {"available": False, "missing_provenance_columns": missing}
    combined = labels.merge(
        provenance[list(required)],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    time_values = combined["time"].to_numpy(float)
    event_values = combined["event"].to_numpy(float)
    latent_risk = combined["survival_latent_risk"].to_numpy(float)
    event_noise = combined["survival_event_noise"].to_numpy(float)
    event_time = combined["survival_event_time"].to_numpy(float)
    censor_time = combined["survival_censor_time"].to_numpy(float)
    reconstructed_event = (event_time <= censor_time).astype(int)
    reconstructed_time = np.clip(
        np.rint(np.minimum(event_time, censor_time)), 6.0, 132.0
    ).astype(int)
    survival = manifest["generation"]["survival"]
    group_c_indices = {
        str(int(group_id)): float(
            concordance_index(
                time_values[combined[GROUP_COLUMN].to_numpy(int) == group_id],
                event_values[combined[GROUP_COLUMN].to_numpy(int) == group_id],
                latent_risk[combined[GROUP_COLUMN].to_numpy(int) == group_id],
            )
        )
        for group_id in sorted(combined[GROUP_COLUMN].astype(int).unique().tolist())
    }
    signal_sd = float(survival["event_signal_log_time_sd"])
    noise_sd = float(np.std(event_noise, ddof=0))
    return {
        "available": True,
        "generator_version": manifest["generator_version"],
        "generator_seed": int(manifest["seed"]) + 200,
        "v7_generated_samples": int(len(combined)),
        "deterministic_event_time_risk_c_index": float(
            concordance_index(time_values, event_values, latent_risk)
        ),
        "realized_hidden_event_time_oracle_c_index": float(
            concordance_index(time_values, event_values, -event_time)
        ),
        "event_prediction_log_time_sd": signal_sd,
        "event_random_residual_log_time_sd": noise_sd,
        "event_noise_to_signal_sd_ratio": noise_sd / max(signal_sd, 1e-12),
        "generation_group_latent_risk_c_index": group_c_indices,
        "minimum_generation_group_latent_risk_c_index": min(group_c_indices.values()),
        "censor_log_time_latent_risk_correlation": float(
            np.corrcoef(latent_risk, np.log(censor_time))[0, 1]
        ),
        "reconstructed_event_agreement": float(
            np.mean(reconstructed_event == event_values.astype(int))
        ),
        "reconstructed_time_exact_agreement": float(
            np.mean(reconstructed_time == time_values.astype(int))
        ),
        "reconstructed_time_mean_absolute_error": float(
            np.mean(np.abs(reconstructed_time - time_values))
        ),
        "interpretation": (
            "Generator_v2 stores audit-only latent risk and hidden times in provenance. These "
            "columns are prohibited model features and are used here only to verify label generation."
        ),
    }


def _archived_v6_reference() -> dict[str, Any]:
    root = Path("archive/model_releases/temporal_topology_v6/artifacts/current_mainline_v2")
    repeat_path = root / "research_repeat_runs_summary.json"
    final_path = root / "full_risk_oof_v2_final_summary.json"
    result: dict[str, Any] = {
        "available": repeat_path.exists(),
        "five_seed_summary_path": str(repeat_path.as_posix()),
        "final_summary_path": str(final_path.as_posix()),
    }
    if repeat_path.exists():
        repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        result["same_gnn_five_seed_mean_test_c_index"] = float(repeat["mean_test_c_index"])
        result["same_gnn_five_seed_std_test_c_index"] = float(repeat["std_test_c_index"])
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        result["v6_final_two_split_mean_test_c_index"] = float(
            final["aggregate"]["mean_final_test_c_index"]
        )
    return result


def _evaluate_saved_gnn(
    config: dict[str, Any],
    *,
    split_seed: int,
    checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    local_config = copy.deepcopy(config)
    graph_preprocess = local_config.get("graph_preprocess", {})
    tabular_preprocess = local_config.get("tabular_preprocess", {})
    dataset = build_dataset_from_csv(
        graph_csv=local_config["paths"]["graph_csv"],
        clinical_csv=local_config["paths"]["clinical_csv"],
        metabolite_csv=local_config["paths"]["metabolite_csv"],
        label_csv=local_config["paths"]["label_csv"],
        node_feature_columns=local_config["model"]["node_feature_columns"],
        clinical_columns=local_config["model"]["clinical_columns"],
        metabolite_columns=local_config["model"]["metabolite_columns"],
        seed=int(local_config["seed"]),
        split_seed=split_seed,
        keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
        min_edge_weight=graph_preprocess.get("min_edge_weight"),
        standardize_tabular=bool(tabular_preprocess.get("standardize", False)),
        val_ratio=float(local_config["train"]["val_ratio"]),
        test_ratio=float(local_config["train"]["test_ratio"]),
    )
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(local_config, dataset, torch_device)
    model.load_state_dict(torch.load(checkpoint, map_location=torch_device))
    model.eval()
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint.as_posix()),
        "tabular_standardization": bool(tabular_preprocess.get("standardize", False)),
    }
    split_sets = {"train": dataset.train_set, "val": dataset.val_set, "test": dataset.test_set}
    with torch.no_grad():
        for name, values in split_sets.items():
            loader = DataLoader(values, batch_size=256, shuffle=False)
            time_values: list[float] = []
            event_values: list[float] = []
            risk_values: list[float] = []
            for batch in loader:
                batch = batch.to(torch_device)
                output = model(batch, compute_contrastive=False)
                time_values.extend(batch.time.detach().cpu().numpy().tolist())
                event_values.extend(batch.event.detach().cpu().numpy().tolist())
                risk_values.extend(output["risk"].detach().cpu().numpy().tolist())
            result[f"{name}_c_index"] = float(
                concordance_index(time_values, event_values, risk_values)
            )
    return result


def _existing_gnn_summary(model_output_root: Path, split_seed: int) -> dict[str, Any]:
    path = (
        model_output_root
        / f"split{split_seed}_five_seed"
        / "research_repeat_runs_summary.json"
    )
    if not path.exists():
        return {"available": False, "path": str(path.as_posix())}
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "path": str(path.as_posix()),
        "mean_test_c_index": float(summary["mean_test_c_index"]),
        "std_test_c_index": float(summary["std_test_c_index"]),
        "runs": [
            {"seed": int(row["seed"]), "test_c_index": float(row["test_c_index"])}
            for row in summary["runs"]
        ],
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(runs)
    rows = []
    keys = ["split_strategy", "split_seed", "model_name", "feature_set"]
    for values, group in frame.groupby(keys, sort=True):
        test_values = group["test_c_index"].astype(float).tolist()
        train_values = group["train_c_index"].dropna().astype(float).tolist()
        rows.append(
            {
                **dict(zip(keys, values)),
                "num_model_seeds": int(len(group)),
                "mean_train_c_index": statistics.mean(train_values) if train_values else None,
                "mean_validation_c_index": float(group["validation_c_index"].mean()),
                "mean_test_c_index": statistics.mean(test_values),
                "std_test_c_index": statistics.stdev(test_values) if len(test_values) > 1 else 0.0,
                "min_test_c_index": min(test_values),
                "max_test_c_index": max(test_values),
            }
        )
    return sorted(rows, key=lambda row: (row["split_strategy"], row["split_seed"], -row["mean_test_c_index"]))


def _aggregate_across_splits(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(runs)
    rows = []
    keys = ["split_strategy", "model_name", "feature_set"]
    for values, group in frame.groupby(keys, sort=True):
        split_means = group.groupby("split_seed")["test_c_index"].mean().astype(float)
        test_values = group["test_c_index"].astype(float)
        train_values = group["train_c_index"].dropna().astype(float)
        rows.append(
            {
                **dict(zip(keys, values)),
                "num_split_seeds": int(group["split_seed"].nunique()),
                "num_model_runs": int(len(group)),
                "mean_train_c_index": float(train_values.mean()) if len(train_values) else None,
                "mean_validation_c_index": float(group["validation_c_index"].mean()),
                "mean_test_c_index": float(test_values.mean()),
                "between_split_std_test_c_index": (
                    float(split_means.std(ddof=1)) if len(split_means) > 1 else 0.0
                ),
                "min_split_mean_test_c_index": float(split_means.min()),
                "max_split_mean_test_c_index": float(split_means.max()),
            }
        )
    return sorted(rows, key=lambda row: (row["split_strategy"], -row["mean_test_c_index"]))


def _diagnostic_interpretation(
    cross_split_aggregates: list[dict[str, Any]],
    gnn: dict[str, Any],
    drift: dict[str, Any],
    generator_oracle: dict[str, Any],
) -> dict[str, Any]:
    def best(strategy: str) -> dict[str, Any]:
        candidates = [
            row for row in cross_split_aggregates if row["split_strategy"] == strategy
        ]
        return max(candidates, key=lambda row: row["mean_test_c_index"])

    random_best = best("random")
    group_best = best("group_disjoint")
    gnn_values = [
        value["five_seed"]["mean_test_c_index"]
        for value in gnn.values()
        if value["five_seed"].get("available")
    ]
    mean_gnn = statistics.mean(gnn_values) if gnn_values else None
    random_group_gap = random_best["mean_test_c_index"] - group_best["mean_test_c_index"]
    group_model_gap = (
        group_best["mean_test_c_index"] - mean_gnn if mean_gnn is not None else None
    )
    replay_train = [
        value["seed42_checkpoint_replay"]["train_c_index"]
        for value in gnn.values()
        if "train_c_index" in value.get("seed42_checkpoint_replay", {})
    ]
    mean_replay_train = statistics.mean(replay_train) if replay_train else None
    domain_auc = max(value["domain_classifier_auc"] for value in drift.values())
    deterministic_generator_c = generator_oracle.get("deterministic_event_time_risk_c_index")

    evidence = {
        "best_random_split_model": random_best,
        "best_group_disjoint_model": group_best,
        "mean_existing_gnn_test_c_index": mean_gnn,
        "random_minus_group_best_c_index": random_group_gap,
        "group_best_minus_gnn_c_index": group_model_gap,
        "mean_saved_gnn_train_c_index": mean_replay_train,
        "maximum_generation_group_domain_auc": domain_auc,
        "deterministic_generator_risk_c_index": deterministic_generator_c,
    }
    flags = {
        "weak_label_signal_likely": random_best["mean_test_c_index"] < 0.58,
        "generator_deterministic_signal_weak": (
            deterministic_generator_c is not None and deterministic_generator_c < 0.58
        ),
        "generation_group_covariate_shift_present": domain_auc >= 0.70,
        "generation_group_shift_harms_c_index": random_group_gap > 0.03,
        "saved_gnn_underfits_training_ranking": (
            mean_replay_train is not None and mean_replay_train < 0.60
        ),
        "stronger_non_gnn_consistently_outperforms_saved_gnn": (
            group_model_gap is not None and group_model_gap > 0.015
        ),
    }
    model_issue_present = flags["saved_gnn_underfits_training_ranking"]
    model_issue_material = flags["stronger_non_gnn_consistently_outperforms_saved_gnn"]
    flags["current_gnn_optimization_issue_present"] = model_issue_present
    flags["model_change_alone_likely_to_resolve_low_c_index"] = model_issue_material
    if flags["weak_label_signal_likely"] and model_issue_present:
        primary = "data_signal_primary_model_training_secondary"
    elif model_issue_present and not flags["weak_label_signal_likely"]:
        primary = "current_gnn_or_feature_representation"
    elif flags["generation_group_shift_harms_c_index"]:
        primary = "generation_group_distribution_shift"
    elif flags["weak_label_signal_likely"]:
        primary = "weak_or_noisy_survival_signal"
    elif deterministic_generator_c is not None and deterministic_generator_c >= 0.70:
        primary = "controlled_signal_recovered_pending_full_gnn_retraining"
    else:
        primary = "mixed_or_unresolved"
    return {"primary_diagnosis": primary, "flags": flags, "evidence": evidence}


def _write_report(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# topology_v7 diagnostic report",
        "",
        f"Primary diagnosis: **{summary['interpretation']['primary_diagnosis']}**",
        "",
        "## Cross-split benchmark",
        "",
        "| Split | Model | Features | Train C-index | Test C-index | Split SD |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["cross_split_aggregates"]:
        train_value = row["mean_train_c_index"]
        train_text = "NA" if train_value is None else f"{train_value:.4f}"
        lines.append(
            f"| {row['split_strategy']} | {row['model_name']} | {row['feature_set']} | "
            f"{train_text} | {row['mean_test_c_index']:.4f} | "
            f"{row['between_split_std_test_c_index']:.4f} |"
        )
    lines.extend(
        [
        "",
        "## Per-split benchmark",
        "",
        "| Split | Seed | Model | Features | Train C-index | Test C-index |",
        "|---|---:|---|---|---:|---:|",
        ]
    )
    for row in summary["aggregates"]:
        train_value = row["mean_train_c_index"]
        train_text = "NA" if train_value is None else f"{train_value:.4f}"
        lines.append(
            f"| {row['split_strategy']} | {row['split_seed']} | {row['model_name']} | "
            f"{row['feature_set']} | {train_text} | {row['mean_test_c_index']:.4f} |"
        )
    oracle = summary["generator_oracle_audit"]
    archived = summary["archived_v6_reference"]
    lines.extend(
        [
            "",
            "## Generator signal audit",
            "",
            (
                f"- Deterministic generator risk C-index: "
                f"`{oracle.get('deterministic_event_time_risk_c_index', float('nan')):.4f}`"
            ),
            (
                f"- Hidden realized-event oracle C-index: "
                f"`{oracle.get('realized_hidden_event_time_oracle_c_index', float('nan')):.4f}`"
            ),
            (
                f"- Event residual-to-signal SD ratio: "
                f"`{oracle.get('event_noise_to_signal_sd_ratio', float('nan')):.2f}`"
            ),
            (
                f"- Reconstructed label event agreement: "
                f"`{oracle.get('reconstructed_event_agreement', float('nan')):.4f}`"
            ),
            "",
            "## Historical control",
            "",
            (
                f"- Archived v6 same-GNN five-seed mean C-index: "
                f"`{archived.get('same_gnn_five_seed_mean_test_c_index', float('nan')):.4f}`"
            ),
            "",
            "## Interpretation flags",
            "",
            *[
                f"- {name}: `{str(value).lower()}`"
                for name, value in summary["interpretation"]["flags"].items()
            ],
            "",
            "Random-split scores are diagnostic only and must not be reported as external validation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps(_as_builtin(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(summary["benchmark_runs"]).to_csv(
        output_dir / "benchmark_runs.csv", index=False
    )
    _write_report(summary, output_dir / "diagnosis_report.md")


def refresh_existing_diagnosis(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "diagnosis_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cross_split = _aggregate_across_splits(summary["benchmark_runs"])
    summary["cross_split_aggregates"] = cross_split
    summary["interpretation"] = _diagnostic_interpretation(
        cross_split,
        summary["existing_gnn"],
        summary["domain_shift"],
        summary["generator_oracle_audit"],
    )
    _write_outputs(summary, output_dir)
    return summary


def run_diagnosis(
    *,
    config_path: Path,
    output_dir: Path,
    split_seeds: Sequence[int],
    model_seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_output_root = Path(config["paths"]["output_dir"]).parent
    frame, feature_sets, feature_metadata = _feature_frame(config)
    required = {"sample_id", "time", "event", GROUP_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Diagnostic frame is missing columns: {missing}")

    benchmark_runs: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    drift: dict[str, Any] = {}
    gnn: dict[str, Any] = {}

    model_specs = [
        ("linear_cox", "edge_identity"),
        ("mlp_cox", "edge_identity"),
        ("xgb_aft", "edge_identity"),
        ("xgb_aft", "full_topology"),
    ]

    for split_seed in split_seeds:
        controlled_frame = frame[
            ["sample_id", "time", "event", GROUP_COLUMN, *feature_sets["full_topology"]]
        ].copy()
        splits = {
            "group_disjoint": _group_split(
                controlled_frame,
                seed=int(split_seed),
                val_ratio=float(config["train"]["val_ratio"]),
                test_ratio=float(config["train"]["test_ratio"]),
            ),
            "random": _random_split(
                controlled_frame,
                seed=int(split_seed),
                val_ratio=float(config["train"]["val_ratio"]),
                test_ratio=float(config["train"]["test_ratio"]),
            ),
        }
        split_summaries[str(split_seed)] = {
            name: value.summary for name, value in splits.items()
        }
        drift[str(split_seed)] = _domain_shift(
            splits["group_disjoint"].train,
            splits["group_disjoint"].test,
            feature_sets["full_topology"],
            seed=int(split_seed),
        )

        checkpoint = (
            model_output_root
            / f"split{split_seed}_five_seed"
            / "research_seed42"
            / "best_model.pt"
        )
        gnn[str(split_seed)] = {
            "five_seed": _existing_gnn_summary(model_output_root, int(split_seed))
        }
        if checkpoint.exists():
            gnn[str(split_seed)]["seed42_checkpoint_replay"] = _evaluate_saved_gnn(
                config,
                split_seed=int(split_seed),
                checkpoint=checkpoint,
                device=device,
            )

        for split_strategy, split in splits.items():
            for model_name, feature_set_name in model_specs:
                for model_seed in model_seeds:
                    if model_name == "linear_cox":
                        metrics = _cox_run(
                            split,
                            feature_sets[feature_set_name],
                            model_type="linear",
                            model_seed=int(model_seed),
                            device=device,
                        )
                    elif model_name == "mlp_cox":
                        metrics = _cox_run(
                            split,
                            feature_sets[feature_set_name],
                            model_type="mlp",
                            model_seed=int(model_seed),
                            device=device,
                        )
                    else:
                        metrics = _aft_run(
                            split,
                            feature_sets[feature_set_name],
                            model_seed=int(model_seed),
                        )
                    benchmark_runs.append(
                        {
                            "split_strategy": split_strategy,
                            "split_seed": int(split_seed),
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "num_features": int(len(feature_sets[feature_set_name])),
                            "model_seed": int(model_seed),
                            **metrics,
                        }
                    )
                    print(
                        f"{split_strategy} split={split_seed} {model_name}/{feature_set_name} "
                        f"seed={model_seed} test_c={metrics['test_c_index']:.4f}",
                        flush=True,
                    )

    aggregates = _aggregate_runs(benchmark_runs)
    cross_split_aggregates = _aggregate_across_splits(benchmark_runs)
    summary = {
        "schema_version": 1,
        "config_path": str(config_path.as_posix()),
        "dataset_version": "topology_v7",
        "diagnostic_scope": (
            "Controlled learnability and model-fit diagnosis. Random splits are diagnostic only."
        ),
        "feature_safety": {
            "generation_group_id_used_only_for_splitting": True,
            "generation_group_id_in_features": False,
            "label_columns_in_features": False,
        },
        "feature_sets": {name: len(columns) for name, columns in feature_sets.items()},
        "feature_metadata": feature_metadata,
        "split_summaries": split_summaries,
        "domain_shift": drift,
        "existing_gnn": gnn,
        "benchmark_runs": benchmark_runs,
        "aggregates": aggregates,
        "cross_split_aggregates": cross_split_aggregates,
    }
    generator_oracle = _generator_oracle_audit(config)
    archived_v6 = _archived_v6_reference()
    summary["generator_oracle_audit"] = generator_oracle
    summary["archived_v6_reference"] = archived_v6
    summary["interpretation"] = _diagnostic_interpretation(
        cross_split_aggregates,
        gnn,
        drift,
        generator_oracle,
    )

    _write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose topology_v7 data versus model failure.")
    parser.add_argument("--config", type=Path, default=Path("research_config_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topology_v7_diagnosis"))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[7, 42, 2026])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Rebuild aggregation and reports from an existing diagnosis_summary.json.",
    )
    args = parser.parse_args()
    if args.refresh_existing:
        result = refresh_existing_diagnosis(args.output_dir)
    else:
        result = run_diagnosis(
            config_path=args.config,
            output_dir=args.output_dir,
            split_seeds=args.split_seeds,
            model_seeds=args.model_seeds,
            device=args.device,
        )
    print(json.dumps(_as_builtin(result["interpretation"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
