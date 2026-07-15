from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from research.data import build_sample_table, load_research_tables
from research.ensemble_stack_v2 import _cohort_cox_loss, _fit_cox_risk_scale
from research.metrics import concordance_index
from research.task import summarize_survival_labels


AFT_PRESETS: dict[str, dict[str, Any]] = {
    "shallow_normal": {
        "max_depth": 2,
        "min_child_weight": 12.0,
        "eta": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_lambda": 4.0,
        "reg_alpha": 0.0,
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
    },
    "balanced_normal": {
        "max_depth": 3,
        "min_child_weight": 8.0,
        "eta": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 5.0,
        "reg_alpha": 0.05,
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
    },
    "shallow_logistic": {
        "max_depth": 2,
        "min_child_weight": 12.0,
        "eta": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_lambda": 4.0,
        "reg_alpha": 0.0,
        "aft_loss_distribution": "logistic",
        "aft_loss_distribution_scale": 1.0,
    },
    "balanced_logistic": {
        "max_depth": 3,
        "min_child_weight": 8.0,
        "eta": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 5.0,
        "reg_alpha": 0.05,
        "aft_loss_distribution": "logistic",
        "aft_loss_distribution_scale": 1.0,
    },
}

FEATURE_SETS = ("full", "legacy_summary", "edge_identity", "topology_only")


def _safe_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return token.lower() or "unknown"


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = frame.copy()
    flattened: list[str] = []
    for column in frame.columns:
        if isinstance(column, tuple):
            parts = [_safe_token(part) for part in column if str(part)]
        else:
            parts = [_safe_token(column)]
        flattened.append("__".join([prefix, *parts]))
    frame.columns = flattened
    return frame


def _pivot_metric(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    prefix: str,
) -> pd.DataFrame:
    pivot = frame.pivot(index=index, columns=columns, values=values).sort_index(axis=1)
    pivot.columns = [f"{prefix}__{_safe_token(column)}" for column in pivot.columns]
    return pivot


def build_topology_fingerprint_dataframe(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build label-free edge, node, and topology features for each sample."""
    graph_df, clinical_df, metabolite_df, label_df, data_summary = load_research_tables(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
    )
    sample_df = build_sample_table(
        clinical_df=clinical_df,
        metabolite_df=metabolite_df,
        label_df=label_df,
    )
    graph = graph_df.copy()
    graph["sample_id"] = graph["sample_id"].astype(str)
    graph["edge_name"] = graph["src"].map(_safe_token) + "__to__" + graph["dst"].map(_safe_token)
    duplicate_edges = graph.duplicated(["sample_id", "edge_name"], keep=False)
    if duplicate_edges.any():
        examples = graph.loc[duplicate_edges, ["sample_id", "src", "dst"]].head(5).to_dict("records")
        raise ValueError(f"Topology fingerprint requires one row per sample/edge; duplicates: {examples}")

    node_rows = graph.drop_duplicates(["sample_id", "node_name"])[
        ["sample_id", "node_name", "abundance", "function_score"]
    ].copy()
    node_parts = [
        _pivot_metric(
            node_rows,
            index="sample_id",
            columns="node_name",
            values="abundance",
            prefix="node_abundance",
        ),
        _pivot_metric(
            node_rows,
            index="sample_id",
            columns="node_name",
            values="function_score",
            prefix="node_function",
        ),
    ]

    src_lookup = node_rows.rename(
        columns={
            "node_name": "src",
            "abundance": "src_abundance",
            "function_score": "src_function",
        }
    )
    dst_lookup = node_rows.rename(
        columns={
            "node_name": "dst",
            "abundance": "dst_abundance",
            "function_score": "dst_function",
        }
    )
    enriched = graph.merge(src_lookup, on=["sample_id", "src"], how="left", validate="many_to_one")
    enriched = enriched.merge(dst_lookup, on=["sample_id", "dst"], how="left", validate="many_to_one")
    enriched["src_abundance_flow"] = enriched["edge_weight"] * enriched["src_abundance"]
    enriched["dst_abundance_flow"] = enriched["edge_weight"] * enriched["dst_abundance"]
    enriched["src_function_flow"] = enriched["edge_weight"] * enriched["src_function"]
    enriched["dst_function_flow"] = enriched["edge_weight"] * enriched["dst_function"]
    enriched["abundance_gap_flow"] = enriched["edge_weight"] * (
        enriched["src_abundance"] - enriched["dst_abundance"]
    ).abs()
    enriched["function_gap_flow"] = enriched["edge_weight"] * (
        enriched["src_function"] - enriched["dst_function"]
    ).abs()

    edge_parts = []
    for metric in (
        "edge_weight",
        "src_abundance_flow",
        "dst_abundance_flow",
        "src_function_flow",
        "dst_function_flow",
        "abundance_gap_flow",
        "function_gap_flow",
    ):
        edge_parts.append(
            _pivot_metric(
                enriched,
                index="sample_id",
                columns="edge_name",
                values=metric,
                prefix=f"edge_{metric}",
            )
        )

    degree_parts: list[pd.DataFrame] = []
    for direction, node_column in (("out", "src"), ("in", "dst")):
        degree = enriched.groupby(["sample_id", node_column])["edge_weight"].agg(["sum", "mean", "max"])
        degree = degree.unstack(node_column).sort_index(axis=1)
        degree_parts.append(_flatten_columns(degree, f"degree_{direction}"))

    edge_summary = enriched.groupby("sample_id")["edge_weight"].agg(
        ["mean", "std", "min", "max", "median", "sum"]
    )
    edge_summary["q25"] = enriched.groupby("sample_id")["edge_weight"].quantile(0.25)
    edge_summary["q75"] = enriched.groupby("sample_id")["edge_weight"].quantile(0.75)
    edge_summary["iqr"] = edge_summary["q75"] - edge_summary["q25"]
    normalized_weight = enriched["edge_weight"] / enriched.groupby("sample_id")["edge_weight"].transform("sum")
    entropy_terms = -normalized_weight * np.log(np.clip(normalized_weight, 1e-12, None))
    edge_summary["entropy"] = entropy_terms.groupby(enriched["sample_id"]).sum()
    edge_summary = edge_summary.add_prefix("topology_edge_")

    threshold_parts: list[pd.DataFrame] = []
    for threshold in (0.2, 0.4, 0.6, 0.8):
        active = enriched["edge_weight"] >= threshold
        grouped = enriched.assign(
            active_count=active.astype(float),
            active_weight=enriched["edge_weight"].where(active, 0.0),
        ).groupby("sample_id")[["active_count", "active_weight"]].sum()
        suffix = str(threshold).replace(".", "p")
        grouped.columns = [f"threshold_{suffix}__{column}" for column in grouped.columns]
        threshold_parts.append(grouped)

    graph_features = pd.concat(
        [*node_parts, *edge_parts, *degree_parts, edge_summary, *threshold_parts],
        axis=1,
    ).reset_index()
    merged = sample_df.merge(graph_features, on="sample_id", how="inner", validate="one_to_one")

    pathogen_nodes = ["fusobacterium", "porphyromonas", "prevotella"]
    pathogen_abundance = [f"node_abundance__{node}" for node in pathogen_nodes]
    pathogen_function = [f"node_function__{node}" for node in pathogen_nodes]
    protective_abundance = "node_abundance__lactobacillus"
    protective_function = "node_function__lactobacillus"
    if all(column in merged for column in [*pathogen_abundance, protective_abundance]):
        merged["axis_pathogen_abundance_mean"] = merged[pathogen_abundance].mean(axis=1)
        merged["axis_dysbiosis_abundance"] = (
            merged["axis_pathogen_abundance_mean"] - merged[protective_abundance]
        )
    if all(column in merged for column in [*pathogen_function, protective_function]):
        merged["axis_pathogen_function_mean"] = merged[pathogen_function].mean(axis=1)
        merged["axis_dysbiosis_function"] = merged["axis_pathogen_function_mean"] - merged[protective_function]

    excluded = {"sample_id", "time", "event"}
    feature_columns = [column for column in merged.columns if column not in excluded]
    numeric = merged[feature_columns].apply(pd.to_numeric, errors="coerce")
    merged.loc[:, feature_columns] = numeric.replace([np.inf, -np.inf], np.nan)
    merged = merged.sort_values("sample_id").reset_index(drop=True)

    metadata = {
        "num_samples": int(len(merged)),
        "num_features": int(len(feature_columns)),
        "num_node_features": int(sum(column.startswith("node_") for column in feature_columns)),
        "num_edge_identity_features": int(sum(column.startswith("edge_") for column in feature_columns)),
        "num_degree_features": int(sum(column.startswith("degree_") for column in feature_columns)),
        "num_threshold_features": int(sum(column.startswith("threshold_") for column in feature_columns)),
        "feature_columns": feature_columns,
        "dataset_origin": data_summary.get("dataset_origin", {}),
        "design_note": (
            "Every edge keeps its source/destination identity. Features use only input modalities and never time/event."
        ),
    }
    return merged, feature_columns, metadata


def select_feature_set(
    feature_columns: Sequence[str],
    config: dict[str, Any],
    feature_set: str,
) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}. Available: {list(FEATURE_SETS)}")
    columns = list(feature_columns)
    if feature_set == "full":
        return columns

    clinical_metabolite = set(config["model"]["clinical_columns"] + config["model"]["metabolite_columns"])
    node_columns = {column for column in columns if column.startswith("node_")}
    raw_edge_columns = {column for column in columns if column.startswith("edge_edge_weight__")}
    legacy_edge_summary = {
        column
        for column in columns
        if column in {
            "topology_edge_mean",
            "topology_edge_std",
            "topology_edge_min",
            "topology_edge_max",
        }
    }
    if feature_set == "legacy_summary":
        selected = clinical_metabolite | node_columns | legacy_edge_summary
    elif feature_set == "edge_identity":
        selected = clinical_metabolite | node_columns | raw_edge_columns
    else:
        selected = set(columns).difference(clinical_metabolite)
    return [column for column in columns if column in selected]


def _load_mainline_predictions(path: str | Path) -> dict[str, np.ndarray]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Mainline prediction artifact not found: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key]) for key in payload.files}
    required = {
        f"{split}_{suffix}"
        for split in ("train", "val", "test")
        for suffix in ("sample_ids", "time", "event", "selected_risk")
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Mainline prediction artifact is missing keys: {missing}")
    return values


def _align_split(
    frame: pd.DataFrame,
    sample_ids: np.ndarray,
    expected_time: np.ndarray,
    expected_event: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    indexed = frame.assign(sample_id=frame["sample_id"].astype(str)).set_index("sample_id", drop=False)
    ids = np.asarray(sample_ids).astype(str)
    missing = sorted(set(ids).difference(indexed.index))
    if missing:
        raise ValueError(f"{split_name} is missing sample IDs: {missing[:5]}")
    aligned = indexed.loc[ids].reset_index(drop=True)
    if not np.allclose(aligned["time"].to_numpy(float), np.asarray(expected_time, dtype=float)):
        raise ValueError(f"{split_name} time labels do not match the mainline artifact.")
    if not np.allclose(aligned["event"].to_numpy(float), np.asarray(expected_event, dtype=float)):
        raise ValueError(f"{split_name} event labels do not match the mainline artifact.")
    return aligned


def _impute_from_train(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train_values = train.loc[:, feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    val_values = val.loc[:, feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    test_values = test.loc[:, feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = train_values.median(axis=0).fillna(0.0)
    arrays = [values.fillna(medians).to_numpy(dtype=np.float32) for values in (train_values, val_values, test_values)]
    for split_name, array in zip(("train", "val", "test"), arrays):
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite topology features remain in {split_name} after train-only imputation.")
    return arrays[0], arrays[1], arrays[2], {str(key): float(value) for key, value in medians.items()}


def _make_aft_dmatrix(
    features: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    feature_columns: Sequence[str],
) -> xgb.DMatrix:
    matrix = xgb.DMatrix(features, feature_names=list(feature_columns))
    lower = np.asarray(time, dtype=np.float32)
    upper = np.where(np.asarray(event, dtype=float) > 0.0, lower, np.inf).astype(np.float32)
    matrix.set_float_info("label_lower_bound", lower)
    matrix.set_float_info("label_upper_bound", upper)
    return matrix


def _parse_eval_metric(eval_text: str) -> float:
    match = re.search(r":([-+0-9.eE]+)\s*$", eval_text)
    if not match:
        raise ValueError(f"Could not parse XGBoost evaluation output: {eval_text}")
    return float(match.group(1))


def _train_aft_candidate(
    *,
    name: str,
    preset: dict[str, Any],
    matrices: dict[str, xgb.DMatrix],
    labels: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    nthread: int,
) -> tuple[xgb.Booster, dict[str, Any], dict[str, np.ndarray]]:
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "tree_method": "hist",
        "seed": int(seed),
        "nthread": int(nthread),
        **preset,
    }
    booster = xgb.train(
        params=params,
        dtrain=matrices["train"],
        num_boost_round=int(num_boost_round),
        evals=[(matrices["train"], "train"), (matrices["val"], "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", num_boost_round - 1))
    best_model = booster[: best_iteration + 1]
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        predicted_log_time = np.asarray(best_model.predict(matrices[split_name]), dtype=float)
        risk = -predicted_log_time
        time, event = labels[split_name]
        predictions[split_name] = risk
        metrics[split_name] = {
            "c_index": float(concordance_index(time, event, risk)),
            "aft_nloglik": _parse_eval_metric(best_model.eval_set([(matrices[split_name], split_name)])),
        }
    result = {
        "name": name,
        "params": params,
        "best_iteration": best_iteration,
        "best_validation_aft_nloglik": float(getattr(booster, "best_score", metrics["val"]["aft_nloglik"])),
        "metrics": metrics,
    }
    return best_model, result, predictions


def _standardize_from_train(train: np.ndarray, *others: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
    train_values = np.asarray(train, dtype=float)
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    transformed = [(np.asarray(values, dtype=float) - mean) / std for values in (train, *others)]
    return transformed, {"train_mean": mean, "train_std": std}


def select_blend_alpha(
    reference_val_risk: np.ndarray,
    expert_val_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    alpha_grid: Iterable[float] | None = None,
    maximum_alpha: float = 1.0,
    alpha_step: float = 0.01,
    minimum_c_index_delta: float = 0.0003,
) -> dict[str, Any]:
    reference = np.asarray(reference_val_risk, dtype=float)
    expert = np.asarray(expert_val_risk, dtype=float)
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    if not (reference.shape == expert.shape == time_values.shape == event_values.shape):
        raise ValueError("Reference, expert, time, and event arrays must have the same shape.")
    if alpha_grid is None:
        if maximum_alpha <= 0.0 or maximum_alpha > 1.0:
            raise ValueError("maximum_alpha must be in (0, 1].")
        if alpha_step <= 0.0:
            raise ValueError("alpha_step must be positive.")
        num_steps = int(round(float(maximum_alpha) / float(alpha_step)))
        alpha_grid = np.linspace(0.0, float(maximum_alpha), num_steps + 1)

    rows = []
    for alpha_value in alpha_grid:
        alpha = float(alpha_value)
        risk = (1.0 - alpha) * reference + alpha * expert
        calibration = _fit_cox_risk_scale(risk, time_values, event_values)
        rows.append(
            {
                "alpha": alpha,
                "validation_c_index": float(concordance_index(time_values, event_values, risk)),
                "calibrated_validation_cox_loss": float(calibration["calibrated_validation_cox_loss"]),
            }
        )
    reference_row = min(rows, key=lambda row: abs(row["alpha"]))
    best_row = sorted(
        rows,
        key=lambda row: (
            -row["validation_c_index"],
            row["calibrated_validation_cox_loss"],
            row["alpha"],
        ),
    )[0]
    delta = float(best_row["validation_c_index"] - reference_row["validation_c_index"])
    selected = best_row if delta >= float(minimum_c_index_delta) else reference_row
    return {
        "reference": reference_row,
        "best_candidate": {**best_row, "validation_c_index_delta": delta},
        "selected": {
            **selected,
            "validation_c_index_delta": float(
                selected["validation_c_index"] - reference_row["validation_c_index"]
            ),
        },
        "minimum_c_index_delta": float(minimum_c_index_delta),
        "maximum_alpha": float(maximum_alpha),
        "alpha_step": float(alpha_step),
        "selected_expert": bool(selected["alpha"] > 0.0),
        "grid": rows,
    }


def _risk_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    pearson = float(np.corrcoef(left_values, right_values)[0, 1])
    left_rank = pd.Series(left_values).rank(method="average").to_numpy(float)
    right_rank = pd.Series(right_values).rank(method="average").to_numpy(float)
    spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def _pair_correction_diagnostics(
    time: np.ndarray,
    event: np.ndarray,
    reference_risk: np.ndarray,
    candidate_risk: np.ndarray,
) -> dict[str, Any]:
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    reference = np.asarray(reference_risk, dtype=float)
    candidate = np.asarray(candidate_risk, dtype=float)
    comparable = (event_values[:, None] > 0.0) & (time_values[:, None] < time_values[None, :])
    reference_correct = reference[:, None] > reference[None, :]
    candidate_correct = candidate[:, None] > candidate[None, :]
    corrected = comparable & ~reference_correct & candidate_correct
    damaged = comparable & reference_correct & ~candidate_correct
    count = int(comparable.sum())
    corrected_count = int(corrected.sum())
    damaged_count = int(damaged.sum())
    return {
        "num_comparable_pairs": count,
        "corrected_pairs": corrected_count,
        "damaged_pairs": damaged_count,
        "net_corrected_pairs": corrected_count - damaged_count,
        "net_correction_rate": float((corrected_count - damaged_count) / max(count, 1)),
    }


def run_experiment(
    *,
    config_path: str | Path,
    mainline_predictions_path: str | Path,
    output_dir: str | Path,
    split_seed: int,
    seed: int,
    preset_names: Sequence[str],
    num_boost_round: int,
    early_stopping_rounds: int,
    nthread: int,
    minimum_c_index_delta: float,
    maximum_alpha: float,
    emit_json: bool = True,
    feature_set: str = "full",
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_path = Path(output_dir)
    model_dir = output_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    unknown = sorted(set(preset_names).difference(AFT_PRESETS))
    if unknown:
        raise ValueError(f"Unknown AFT presets: {unknown}. Available: {sorted(AFT_PRESETS)}")

    frame, all_feature_columns, feature_metadata = build_topology_fingerprint_dataframe(config)
    feature_columns = select_feature_set(all_feature_columns, config, feature_set)
    feature_metadata = {
        **feature_metadata,
        "feature_set": feature_set,
        "num_all_available_features": len(all_feature_columns),
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
    }
    mainline = _load_mainline_predictions(mainline_predictions_path)

    aligned: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        aligned[split_name] = _align_split(
            frame,
            mainline[f"{split_name}_sample_ids"],
            mainline[f"{split_name}_time"],
            mainline[f"{split_name}_event"],
            split_name,
        )
    split_summary = {
        "split_seed": int(split_seed),
        "split_strategy": "replayed_exactly_from_mainline_prediction_artifact",
        "num_total_samples": int(sum(len(aligned[name]) for name in ("train", "val", "test"))),
        **{
            split_name: summarize_survival_labels(
                aligned[split_name][["sample_id", "time", "event"]]
            )
            for split_name in ("train", "val", "test")
        },
    }

    train_x, val_x, test_x, imputation_medians = _impute_from_train(
        aligned["train"], aligned["val"], aligned["test"], feature_columns
    )
    feature_arrays = {"train": train_x, "val": val_x, "test": test_x}
    labels = {
        split_name: (
            np.asarray(mainline[f"{split_name}_time"], dtype=float),
            np.asarray(mainline[f"{split_name}_event"], dtype=float),
        )
        for split_name in ("train", "val", "test")
    }
    matrices = {
        split_name: _make_aft_dmatrix(
            feature_arrays[split_name],
            labels[split_name][0],
            labels[split_name][1],
            feature_columns,
        )
        for split_name in ("train", "val", "test")
    }

    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[str, dict[str, np.ndarray]] = {}
    for name in preset_names:
        model, row, predictions = _train_aft_candidate(
            name=name,
            preset=AFT_PRESETS[name],
            matrices=matrices,
            labels=labels,
            seed=int(seed),
            num_boost_round=int(num_boost_round),
            early_stopping_rounds=int(early_stopping_rounds),
            nthread=int(nthread),
        )
        model_path = model_dir / f"{name}.json"
        model.save_model(model_path)
        row["model_path"] = str(model_path.as_posix())
        candidate_rows.append(row)
        candidate_predictions[name] = predictions

    selected_candidate = sorted(
        candidate_rows,
        key=lambda row: (
            -row["metrics"]["val"]["c_index"],
            row["metrics"]["val"]["aft_nloglik"],
            row["best_iteration"],
        ),
    )[0]
    selected_name = str(selected_candidate["name"])
    expert_raw = candidate_predictions[selected_name]
    main_raw = {
        split_name: np.asarray(mainline[f"{split_name}_selected_risk"], dtype=float)
        for split_name in ("train", "val", "test")
    }

    main_standardized, main_scaler = _standardize_from_train(
        main_raw["train"], main_raw["val"], main_raw["test"]
    )
    expert_standardized, expert_scaler = _standardize_from_train(
        expert_raw["train"], expert_raw["val"], expert_raw["test"]
    )
    main_z = dict(zip(("train", "val", "test"), main_standardized))
    expert_z = dict(zip(("train", "val", "test"), expert_standardized))

    blend_selection = select_blend_alpha(
        main_z["val"],
        expert_z["val"],
        labels["val"][0],
        labels["val"][1],
        maximum_alpha=float(maximum_alpha),
        minimum_c_index_delta=float(minimum_c_index_delta),
    )
    selected_alpha = float(blend_selection["selected"]["alpha"])
    blended = {
        split_name: (1.0 - selected_alpha) * main_z[split_name] + selected_alpha * expert_z[split_name]
        for split_name in ("train", "val", "test")
    }

    reference_calibration = _fit_cox_risk_scale(main_z["val"], labels["val"][0], labels["val"][1])
    selected_calibration = _fit_cox_risk_scale(blended["val"], labels["val"][0], labels["val"][1])
    reference_test_c_index = float(
        concordance_index(labels["test"][0], labels["test"][1], main_z["test"])
    )
    expert_test_c_index = float(
        concordance_index(labels["test"][0], labels["test"][1], expert_z["test"])
    )
    selected_test_c_index = float(
        concordance_index(labels["test"][0], labels["test"][1], blended["test"])
    )
    reference_test_loss = _cohort_cox_loss(
        main_z["test"] * float(reference_calibration["scale"]),
        labels["test"][0],
        labels["test"][1],
    )
    selected_test_loss = _cohort_cox_loss(
        blended["test"] * float(selected_calibration["scale"]),
        labels["test"][0],
        labels["test"][1],
    )

    prediction_path = output_path / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        selected_expert_name=np.asarray(selected_name),
        selected_alpha=np.asarray(selected_alpha, dtype=float),
        **{
            f"{split_name}_{key}": value
            for split_name in ("train", "val", "test")
            for key, value in {
                "sample_ids": np.asarray(mainline[f"{split_name}_sample_ids"]),
                "time": labels[split_name][0],
                "event": labels[split_name][1],
                "mainline_risk": main_raw[split_name],
                "expert_risk": expert_raw[split_name],
                "selected_risk": blended[split_name],
            }.items()
        },
    )

    result = {
        "protocol": "independent_topology_fingerprint_xgboost_aft_with_validation_safe_fusion",
        "config_path": str(Path(config_path).as_posix()),
        "mainline_predictions_path": str(Path(mainline_predictions_path).as_posix()),
        "output_dir": str(output_path.as_posix()),
        "split_seed": int(split_seed),
        "model_seed": int(seed),
        "feature_metadata": feature_metadata,
        "imputation": {
            "strategy": "train_only_median",
            "num_imputation_values": len(imputation_medians),
        },
        "split_summary": split_summary,
        "aft_training": {
            "num_boost_round": int(num_boost_round),
            "early_stopping_rounds": int(early_stopping_rounds),
            "nthread": int(nthread),
            "candidate_selection": "highest validation c-index; AFT nloglik and complexity break ties",
        },
        "candidates": candidate_rows,
        "selected_expert": selected_candidate,
        "risk_standardization": {
            "mainline": main_scaler,
            "expert": expert_scaler,
            "policy": "both risks use train-only mean and standard deviation before convex blending",
        },
        "blend_selection": blend_selection,
        "validation": {
            "reference_c_index": float(
                concordance_index(labels["val"][0], labels["val"][1], main_z["val"])
            ),
            "expert_c_index": float(
                concordance_index(labels["val"][0], labels["val"][1], expert_z["val"])
            ),
            "selected_c_index": float(
                concordance_index(labels["val"][0], labels["val"][1], blended["val"])
            ),
            "main_expert_correlation": _risk_correlation(main_z["val"], expert_z["val"]),
            "selected_pair_corrections": _pair_correction_diagnostics(
                labels["val"][0], labels["val"][1], main_z["val"], blended["val"]
            ),
        },
        "test": {
            "reference_c_index": reference_test_c_index,
            "expert_c_index": expert_test_c_index,
            "selected_c_index": selected_test_c_index,
            "selected_c_index_delta": selected_test_c_index - reference_test_c_index,
            "reference_calibrated_cox_loss": reference_test_loss,
            "selected_calibrated_cox_loss": selected_test_loss,
            "calibrated_cox_loss_delta": selected_test_loss - reference_test_loss,
            "main_expert_correlation": _risk_correlation(main_z["test"], expert_z["test"]),
            "selected_pair_corrections": _pair_correction_diagnostics(
                labels["test"][0], labels["test"][1], main_z["test"], blended["test"]
            ),
        },
        "artifacts": {
            "predictions": str(prediction_path.as_posix()),
            "models": [row["model_path"] for row in candidate_rows],
        },
        "decision": (
            "accept_temporal_topology_expert"
            if blend_selection["selected_expert"]
            else "reject_temporal_topology_expert_and_keep_mainline"
        ),
        "interpretation": (
            "The AFT expert sees edge identities and full censored survival time. It is independent from the current "
            "mainline and receives zero weight unless validation C-index clears the configured minimum gain."
        ),
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if emit_json:
        print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--mainline-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--presets", nargs="+", default=list(AFT_PRESETS))
    parser.add_argument("--num-boost-round", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--nthread", type=int, default=6)
    parser.add_argument("--minimum-c-index-delta", type=float, default=0.0003)
    parser.add_argument("--maximum-alpha", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full")
    args = parser.parse_args()
    run_experiment(
        config_path=args.config,
        mainline_predictions_path=args.mainline_predictions,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        seed=args.seed,
        preset_names=args.presets,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        nthread=args.nthread,
        minimum_c_index_delta=args.minimum_c_index_delta,
        maximum_alpha=args.maximum_alpha,
        emit_json=not args.quiet,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()
