from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from research.task import (
    get_survival_task_definition,
    infer_dataset_origin,
    load_and_validate_survival_labels,
    summarize_survival_labels,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data
else:
    Data = Any


@dataclass
class DatasetBundle:
    train_set: List[Data]
    val_set: List[Data]
    test_set: List[Data]
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int
    num_node_types: int
    node_type_names: List[str]
    task_definition: dict[str, Any]
    data_summary: dict[str, Any]
    split_summary: dict[str, Any]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_sample_graph(
    sample_graph: pd.DataFrame,
    keep_top_k_edges: int | None = None,
    min_edge_weight: float | None = None,
) -> pd.DataFrame:
    edge_graph = sample_graph.copy()
    edge_graph["edge_weight"] = edge_graph["edge_weight"].astype(float)

    if min_edge_weight is not None:
        edge_graph = edge_graph.loc[edge_graph["edge_weight"] >= float(min_edge_weight)].copy()

    if keep_top_k_edges is not None and keep_top_k_edges > 0 and len(edge_graph) > keep_top_k_edges:
        edge_graph = edge_graph.nlargest(keep_top_k_edges, "edge_weight").copy()

    if edge_graph.empty:
        edge_graph = sample_graph.copy()
        edge_graph["edge_weight"] = edge_graph["edge_weight"].astype(float)
        edge_graph = edge_graph.nlargest(1, "edge_weight").copy()

    return edge_graph.reset_index(drop=True)


def _validate_unique_sample_ids(df: pd.DataFrame, table_name: str) -> None:
    if "sample_id" not in df.columns:
        raise ValueError(f"{table_name} must contain a sample_id column.")
    if df["sample_id"].duplicated().any():
        duplicated = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()[:10]
        raise ValueError(f"{table_name} contains duplicate sample_id values: {duplicated}")


def _row_examples(df: pd.DataFrame, mask: pd.Series, limit: int = 5) -> list[str]:
    selected = df.loc[mask]
    if "sample_id" in selected.columns:
        return selected["sample_id"].astype(str).head(limit).tolist()
    return [str(index) for index in selected.index[:limit]]


def _coerce_finite_column(df: pd.DataFrame, column: str, table_name: str) -> None:
    numeric = pd.to_numeric(df[column], errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid.any():
        examples = _row_examples(df, pd.Series(invalid, index=df.index))
        raise ValueError(
            f"{table_name}.{column} contains non-numeric, NaN, or infinite values; "
            f"sample_id examples: {examples}"
        )
    df[column] = numeric.astype(float)


def _validate_column_range(
    df: pd.DataFrame,
    column: str,
    table_name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    values = df[column].astype(float)
    invalid = values < float(minimum)
    if maximum is not None:
        invalid = invalid | (values > float(maximum))
    if invalid.any():
        examples = _row_examples(df, invalid)
        upper_text = "unbounded" if maximum is None else f"{maximum:g}"
        raise ValueError(
            f"{table_name}.{column} must be within [{minimum:g}, {upper_text}]; "
            f"sample_id examples: {examples}"
        )


def validate_research_feature_tables(
    graph_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    metabolite_df: pd.DataFrame,
) -> dict[str, Any]:
    numeric_columns = {
        "graph table": [
            column
            for column in ("abundance", "function_score", "edge_weight")
            if column in graph_df.columns
        ],
        "clinical table": [column for column in clinical_df.columns if column != "sample_id"],
        "metabolite table": [column for column in metabolite_df.columns if column != "sample_id"],
    }
    table_map = {
        "graph table": graph_df,
        "clinical table": clinical_df,
        "metabolite table": metabolite_df,
    }
    for table_name, columns in numeric_columns.items():
        for column in columns:
            _coerce_finite_column(table_map[table_name], column, table_name)

    if "abundance" in graph_df.columns:
        _validate_column_range(graph_df, "abundance", "graph table", minimum=0.0, maximum=1.0)
        dedup = graph_df.drop_duplicates(subset=["sample_id", "node_name"])
        totals = dedup.groupby("sample_id")["abundance"].sum()
        empty_samples = totals.index[totals <= 0.0].astype(str).tolist()[:5]
        if empty_samples:
            raise ValueError(
                "graph table must contain at least one positive microbial abundance per sample; "
                f"sample_id examples: {empty_samples}"
            )
    if "function_score" in graph_df.columns:
        _validate_column_range(graph_df, "function_score", "graph table", minimum=0.0, maximum=1.0)

    known_clinical_ranges = {
        "age": (1.0, 120.0),
        "bmi": (5.0, 100.0),
        "smoking": (0.0, 1.0),
        "family_history": (0.0, 1.0),
    }
    for column, (minimum, maximum) in known_clinical_ranges.items():
        if column in clinical_df.columns:
            _validate_column_range(
                clinical_df,
                column,
                "clinical table",
                minimum=minimum,
                maximum=maximum,
            )
    for column in ("smoking", "family_history"):
        if column in clinical_df.columns:
            invalid = ~clinical_df[column].isin([0.0, 1.0])
            if invalid.any():
                examples = _row_examples(clinical_df, invalid)
                raise ValueError(
                    f"clinical table.{column} must contain only 0 or 1; sample_id examples: {examples}"
                )

    for column in numeric_columns["metabolite table"]:
        _validate_column_range(
            metabolite_df,
            column,
            "metabolite table",
            minimum=0.0,
            maximum=1.0,
        )

    ranges = {
        table_name: {
            column: {
                "min": float(table_map[table_name][column].min()),
                "max": float(table_map[table_name][column].max()),
            }
            for column in columns
        }
        for table_name, columns in numeric_columns.items()
    }
    return {
        "validated": True,
        "numeric_columns": numeric_columns,
        "observed_ranges": ranges,
    }


def _build_edges(sample_graph: pd.DataFrame, node_order: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    node_to_idx = {name: i for i, name in enumerate(node_order)}
    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_attr: List[List[float]] = []

    for _, row in sample_graph.iterrows():
        src = node_to_idx[row["src"]]
        dst = node_to_idx[row["dst"]]
        weight = float(row["edge_weight"])
        edges_src.extend([src, dst])
        edges_dst.extend([dst, src])
        edge_attr.extend([[weight], [weight]])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    return edge_index, edge_attr_tensor


def load_research_tables(
    graph_csv: str,
    clinical_csv: str,
    metabolite_csv: str,
    label_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    graph_df = pd.read_csv(graph_csv)
    clinical_df = pd.read_csv(clinical_csv)
    metabolite_df = pd.read_csv(metabolite_csv)
    label_df = load_and_validate_survival_labels(label_csv)

    _validate_unique_sample_ids(clinical_df, "clinical table")
    _validate_unique_sample_ids(metabolite_df, "metabolite table")

    graph_required = {"sample_id", "node_name", "src", "dst", "edge_weight"}
    missing_graph_cols = sorted(graph_required.difference(graph_df.columns))
    if missing_graph_cols:
        raise ValueError(f"Graph table is missing required columns: {missing_graph_cols}")

    data_quality = validate_research_feature_tables(graph_df, clinical_df, metabolite_df)

    graph_sample_ids = set(graph_df["sample_id"].astype(str).unique().tolist())
    clinical_sample_ids = set(clinical_df["sample_id"].astype(str).tolist())
    metabolite_sample_ids = set(metabolite_df["sample_id"].astype(str).tolist())
    label_sample_ids = set(label_df["sample_id"].astype(str).tolist())
    shared_sample_ids = graph_sample_ids & clinical_sample_ids & metabolite_sample_ids & label_sample_ids

    if not shared_sample_ids:
        raise ValueError("No overlapping sample_id values exist across graph/clinical/metabolite/label tables.")

    mismatches = {
        "graph_missing_from_shared": sorted(shared_sample_ids.difference(graph_sample_ids)),
        "clinical_missing_from_shared": sorted(shared_sample_ids.difference(clinical_sample_ids)),
        "metabolite_missing_from_shared": sorted(shared_sample_ids.difference(metabolite_sample_ids)),
        "label_missing_from_shared": sorted(shared_sample_ids.difference(label_sample_ids)),
    }
    dropped = {
        "graph_only": sorted(graph_sample_ids.difference(shared_sample_ids))[:20],
        "clinical_only": sorted(clinical_sample_ids.difference(shared_sample_ids))[:20],
        "metabolite_only": sorted(metabolite_sample_ids.difference(shared_sample_ids))[:20],
        "label_only": sorted(label_sample_ids.difference(shared_sample_ids))[:20],
    }

    graph_df = graph_df.loc[graph_df["sample_id"].astype(str).isin(shared_sample_ids)].copy()
    clinical_df = clinical_df.loc[clinical_df["sample_id"].astype(str).isin(shared_sample_ids)].copy()
    metabolite_df = metabolite_df.loc[metabolite_df["sample_id"].astype(str).isin(shared_sample_ids)].copy()
    label_df = label_df.loc[label_df["sample_id"].astype(str).isin(shared_sample_ids)].copy()

    dataset_origin = infer_dataset_origin(graph_csv, clinical_csv, metabolite_csv, label_csv)
    strict_assumptions = [
        "All modalities are inner-joined on sample_id before splitting.",
        "Evaluation remains survival-style and uses the same time/event semantics across deep models and baselines.",
    ]
    if dataset_origin["dataset_version"] == "topology_v7":
        strict_assumptions.extend(
            [
                "topology_v7 contains model-generated development samples anchored to a 42-patient public paired oral-gut cohort.",
                "Clinical variables, metabolites, topology targets, and survival labels are generated proxies rather than observed patient measurements.",
                "generation_group_id must remain disjoint across train, validation, and test splits.",
                "Performance on topology_v7 measures generator recovery and is not external clinical validation.",
            ]
        )
    else:
        strict_assumptions.append(
            "topology_v6 is synthetic/noisy augmented research data and is not an external clinical benchmark."
        )

    data_summary = {
        "task_definition": get_survival_task_definition(),
        "label_summary": summarize_survival_labels(label_df),
        "dataset_origin": dataset_origin,
        "modalities": {
            "graph_num_rows": int(len(graph_df)),
            "graph_num_samples": int(graph_df["sample_id"].nunique()),
            "clinical_num_samples": int(clinical_df["sample_id"].nunique()),
            "metabolite_num_samples": int(metabolite_df["sample_id"].nunique()),
            "label_num_samples": int(label_df["sample_id"].nunique()),
        },
        "dropped_sample_id_examples": dropped,
        "data_quality": data_quality,
        "strict_assumptions": strict_assumptions,
    }
    return graph_df, clinical_df, metabolite_df, label_df, data_summary


def build_sample_table(
    clinical_df: pd.DataFrame,
    metabolite_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = clinical_df.merge(metabolite_df, on="sample_id", how="inner").merge(label_df, on="sample_id", how="inner")
    _validate_unique_sample_ids(merged, "merged sample table")
    return merged


def fit_tabular_standardizer(
    train_df: pd.DataFrame,
    clinical_columns: List[str],
    metabolite_columns: List[str],
) -> dict[str, Any]:
    feature_groups = {
        "clinical": list(clinical_columns),
        "metabolite": list(metabolite_columns),
    }
    columns = feature_groups["clinical"] + feature_groups["metabolite"]
    if len(columns) != len(set(columns)):
        raise ValueError("Clinical and metabolite feature names must be unique for tabular standardization.")

    missing_columns = [column for column in columns if column not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Cannot standardize missing tabular columns: {missing_columns}")

    train_values = train_df[columns].astype(float)
    means = train_values.mean(axis=0)
    raw_scales = train_values.std(axis=0, ddof=0)
    constant_mask = raw_scales <= 1e-12
    scales = raw_scales.mask(constant_mask, 1.0)

    return {
        "enabled": True,
        "method": "z_score",
        "fit_split": "train",
        "feature_groups": feature_groups,
        "features": {
            column: {
                "mean": float(means[column]),
                "scale": float(scales[column]),
                "zero_variance": bool(constant_mask[column]),
            }
            for column in columns
        },
    }


def apply_tabular_standardizer(
    sample_df: pd.DataFrame,
    standardizer: dict[str, Any],
) -> pd.DataFrame:
    if standardizer.get("method") != "z_score":
        raise ValueError(f"Unsupported tabular standardization method: {standardizer.get('method')}")

    transformed = sample_df.copy()
    for column, stats in standardizer["features"].items():
        if column not in transformed.columns:
            raise ValueError(f"Cannot standardize missing tabular column: {column}")
        transformed[column] = (transformed[column].astype(float) - float(stats["mean"])) / float(stats["scale"])
    return transformed


def _event_stratify_labels(sample_df: pd.DataFrame) -> np.ndarray | None:
    event_values = sample_df["event"].astype(int).to_numpy()
    unique, counts = np.unique(event_values, return_counts=True)
    if len(unique) < 2:
        return None
    if np.min(counts) < 2:
        return None
    return event_values


def _canonical_group_value(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("generation_group_id cannot contain missing values.")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value)


def split_sample_table(
    sample_df: pd.DataFrame,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    validation_group: str | int | None = None,
    test_group: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1.")

    total = len(sample_df)
    if total < 5:
        raise ValueError("At least 5 samples are required for reproducible train/val/test splitting.")

    group_column = "generation_group_id" if "generation_group_id" in sample_df.columns else None
    effective_val_ratio = val_ratio / (1.0 - test_ratio)
    explicit_group_split = validation_group is not None or test_group is not None
    if explicit_group_split and (validation_group is None or test_group is None):
        raise ValueError("validation_group and test_group must be provided together.")
    if explicit_group_split and group_column is None:
        raise ValueError("Explicit LOGO splitting requires generation_group_id.")

    if explicit_group_split:
        group_values = sample_df[group_column].map(_canonical_group_value)
        requested_validation_group = _canonical_group_value(validation_group)
        requested_test_group = _canonical_group_value(test_group)
        if requested_validation_group == requested_test_group:
            raise ValueError("validation_group and test_group must be distinct.")
        available_groups = set(group_values.unique().tolist())
        missing_groups = sorted(
            {requested_validation_group, requested_test_group}.difference(available_groups)
        )
        if missing_groups:
            raise ValueError(f"Explicit LOGO groups are absent from the dataset: {missing_groups}")

        val_df = sample_df.loc[group_values == requested_validation_group].copy()
        test_df = sample_df.loc[group_values == requested_test_group].copy()
        train_df = sample_df.loc[
            ~group_values.isin([requested_validation_group, requested_test_group])
        ].copy()
        split_strategy = "generation_group_explicit_logo_train_val_test_split"
    elif group_column is not None:
        group_values = sample_df[group_column].map(_canonical_group_value).to_numpy()
        if len(np.unique(group_values)) < 3:
            raise ValueError("Grouped splitting requires at least three distinct generation groups.")
        outer = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_indices, test_indices = next(outer.split(sample_df, groups=group_values))
        train_val_df = sample_df.iloc[train_val_indices].copy()
        test_df = sample_df.iloc[test_indices].copy()

        train_val_groups = train_val_df[group_column].map(_canonical_group_value).to_numpy()
        inner = GroupShuffleSplit(n_splits=1, test_size=effective_val_ratio, random_state=seed + 1)
        train_indices, val_indices = next(inner.split(train_val_df, groups=train_val_groups))
        train_df = train_val_df.iloc[train_indices].copy()
        val_df = train_val_df.iloc[val_indices].copy()
        split_strategy = "generation_group_disjoint_train_val_test_split"
    else:
        stratify = _event_stratify_labels(sample_df)
        train_val_df, test_df = train_test_split(
            sample_df,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )

        stratify_train_val = _event_stratify_labels(train_val_df)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=effective_val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_train_val,
        )
        split_strategy = "event_stratified_train_val_test_split"

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df.empty:
            raise ValueError(f"{split_name} split is empty after splitting.")

    train_ids = set(train_df["sample_id"].astype(str).tolist())
    val_ids = set(val_df["sample_id"].astype(str).tolist())
    test_ids = set(test_df["sample_id"].astype(str).tolist())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Sample leakage detected: train/val/test splits overlap.")

    split_summary = {
        "split_seed": int(seed),
        "split_strategy": split_strategy,
        "num_total_samples": int(total),
        "train": summarize_survival_labels(train_df[["sample_id", "time", "event"]]),
        "val": summarize_survival_labels(val_df[["sample_id", "time", "event"]]),
        "test": summarize_survival_labels(test_df[["sample_id", "time", "event"]]),
        "train_sample_ids_preview": train_df["sample_id"].astype(str).head(10).tolist(),
        "val_sample_ids_preview": val_df["sample_id"].astype(str).head(10).tolist(),
        "test_sample_ids_preview": test_df["sample_id"].astype(str).head(10).tolist(),
    }
    if group_column is not None:
        split_summary["group_column"] = group_column
        split_summary["train_groups"] = sorted(
            train_df[group_column].map(_canonical_group_value).unique().tolist()
        )
        split_summary["val_groups"] = sorted(
            val_df[group_column].map(_canonical_group_value).unique().tolist()
        )
        split_summary["test_groups"] = sorted(
            test_df[group_column].map(_canonical_group_value).unique().tolist()
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_summary,
    )


def build_dataset_from_csv(
    graph_csv: str,
    clinical_csv: str,
    metabolite_csv: str,
    label_csv: str,
    node_feature_columns: List[str],
    clinical_columns: List[str],
    metabolite_columns: List[str],
    seed: int = 42,
    split_seed: int | None = None,
    keep_top_k_edges: int | None = None,
    min_edge_weight: float | None = None,
    standardize_tabular: bool = False,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    validation_group: str | int | None = None,
    test_group: str | int | None = None,
) -> DatasetBundle:
    from torch_geometric.data import Data
    from research.model_v2 import compute_single_graph_structure

    graph_df, clinical_df, metabolite_df, label_df, data_summary = load_research_tables(
        graph_csv=graph_csv,
        clinical_csv=clinical_csv,
        metabolite_csv=metabolite_csv,
        label_csv=label_csv,
    )
    sample_table = build_sample_table(clinical_df=clinical_df, metabolite_df=metabolite_df, label_df=label_df)

    effective_split_seed = seed if split_seed is None else split_seed
    train_df, val_df, test_df, split_summary = split_sample_table(
        sample_df=sample_table,
        seed=effective_split_seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        validation_group=validation_group,
        test_group=test_group,
    )

    if standardize_tabular:
        tabular_preprocess = fit_tabular_standardizer(
            train_df,
            clinical_columns=clinical_columns,
            metabolite_columns=metabolite_columns,
        )
        sample_table = apply_tabular_standardizer(sample_table, tabular_preprocess)
    else:
        tabular_preprocess = {
            "enabled": False,
            "method": "none",
            "fit_split": None,
            "feature_groups": {
                "clinical": list(clinical_columns),
                "metabolite": list(metabolite_columns),
            },
            "features": {},
        }

    split_map = {
        "train": set(train_df["sample_id"].astype(str).tolist()),
        "val": set(val_df["sample_id"].astype(str).tolist()),
        "test": set(test_df["sample_id"].astype(str).tolist()),
    }
    merged_by_id = sample_table.set_index("sample_id")
    node_type_names = sorted(graph_df["node_name"].astype(str).unique().tolist())
    node_type_to_index = {name: index for index, name in enumerate(node_type_names)}

    data_list_by_split: dict[str, List[Data]] = {"train": [], "val": [], "test": []}

    for sample_id, sample_graph in graph_df.groupby("sample_id"):
        sample_id = str(sample_id)
        if sample_id not in merged_by_id.index:
            raise ValueError(f"Sample {sample_id} exists in graph table but not in merged sample table.")

        sample_meta = merged_by_id.loc[sample_id]
        node_order = sample_graph["node_name"].drop_duplicates().tolist()
        node_feature_rows = (
            sample_graph.drop_duplicates(subset=["node_name"])
            .set_index("node_name")
            .loc[node_order, node_feature_columns]
        )
        if node_feature_rows.isna().any().any():
            raise ValueError(f"Node features contain NaN values for sample {sample_id}.")
        x = torch.tensor(node_feature_rows.to_numpy(dtype=float), dtype=torch.float32)
        node_type = torch.tensor(
            [node_type_to_index[str(name)] for name in node_order],
            dtype=torch.long,
        )

        processed_graph = preprocess_sample_graph(
            sample_graph,
            keep_top_k_edges=keep_top_k_edges,
            min_edge_weight=min_edge_weight,
        )
        edge_index, edge_attr = _build_edges(processed_graph, node_order)
        node_struct, node_targets, graph_targets, graph_cluster_targets = compute_single_graph_structure(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        clinical = torch.tensor(sample_meta[clinical_columns].to_numpy(dtype=float), dtype=torch.float32)
        metabolites = torch.tensor(sample_meta[metabolite_columns].to_numpy(dtype=float), dtype=torch.float32)
        time = torch.tensor(float(sample_meta["time"]), dtype=torch.float32)
        event = torch.tensor(float(sample_meta["event"]), dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_type,
            node_struct=node_struct,
            node_targets=node_targets,
            graph_targets=graph_targets,
            graph_cluster_targets=graph_cluster_targets,
            clinical=clinical,
            metabolites=metabolites,
            time=time,
            event=event,
            sample_id=sample_id,
        )

        if sample_id in split_map["train"]:
            data_list_by_split["train"].append(data)
        elif sample_id in split_map["val"]:
            data_list_by_split["val"].append(data)
        elif sample_id in split_map["test"]:
            data_list_by_split["test"].append(data)
        else:
            raise ValueError(f"Sample {sample_id} was not assigned to train/val/test split.")

    data_summary["feature_dimensions"] = {
        "node_feature_dim": int(len(node_feature_columns)),
        "clinical_dim": int(len(clinical_columns)),
        "metabolite_dim": int(len(metabolite_columns)),
        "num_node_types": int(len(node_type_names)),
        "node_type_names": node_type_names,
        "precomputed_structure_features": True,
    }
    data_summary["graph_preprocess"] = {
        "keep_top_k_edges": keep_top_k_edges,
        "min_edge_weight": min_edge_weight,
    }
    data_summary["tabular_preprocess"] = tabular_preprocess
    data_summary["split_summary"] = split_summary

    return DatasetBundle(
        train_set=data_list_by_split["train"],
        val_set=data_list_by_split["val"],
        test_set=data_list_by_split["test"],
        node_feature_dim=len(node_feature_columns),
        clinical_dim=len(clinical_columns),
        metabolite_dim=len(metabolite_columns),
        num_node_types=len(node_type_names),
        node_type_names=node_type_names,
        task_definition=get_survival_task_definition(),
        data_summary=data_summary,
        split_summary=split_summary,
    )
