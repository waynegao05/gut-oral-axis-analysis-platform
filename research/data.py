from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

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

    data_summary = {
        "task_definition": get_survival_task_definition(),
        "label_summary": summarize_survival_labels(label_df),
        "dataset_origin": infer_dataset_origin(graph_csv, clinical_csv, metabolite_csv, label_csv),
        "modalities": {
            "graph_num_rows": int(len(graph_df)),
            "graph_num_samples": int(graph_df["sample_id"].nunique()),
            "clinical_num_samples": int(clinical_df["sample_id"].nunique()),
            "metabolite_num_samples": int(metabolite_df["sample_id"].nunique()),
            "label_num_samples": int(label_df["sample_id"].nunique()),
        },
        "dropped_sample_id_examples": dropped,
        "strict_assumptions": [
            "All modalities are inner-joined on sample_id before splitting.",
            "Current repository dataset version is inferred from file names; topology_v6 is treated as synthetic/noisy augmented research data.",
            "Evaluation remains survival-style and uses the same time/event semantics across deep models and baselines.",
        ],
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


def _event_stratify_labels(sample_df: pd.DataFrame) -> np.ndarray | None:
    event_values = sample_df["event"].astype(int).to_numpy()
    unique, counts = np.unique(event_values, return_counts=True)
    if len(unique) < 2:
        return None
    if np.min(counts) < 2:
        return None
    return event_values


def split_sample_table(
    sample_df: pd.DataFrame,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1.")

    total = len(sample_df)
    if total < 5:
        raise ValueError("At least 5 samples are required for reproducible train/val/test splitting.")

    effective_test_ratio = test_ratio
    stratify = _event_stratify_labels(sample_df)
    train_val_df, test_df = train_test_split(
        sample_df,
        test_size=effective_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    effective_val_ratio = val_ratio / (1.0 - test_ratio)
    stratify_train_val = _event_stratify_labels(train_val_df)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=effective_val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_train_val,
    )

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
        "split_strategy": "event_stratified_train_val_test_split",
        "num_total_samples": int(total),
        "train": summarize_survival_labels(train_df[["sample_id", "time", "event"]]),
        "val": summarize_survival_labels(val_df[["sample_id", "time", "event"]]),
        "test": summarize_survival_labels(test_df[["sample_id", "time", "event"]]),
        "train_sample_ids_preview": train_df["sample_id"].astype(str).head(10).tolist(),
        "val_sample_ids_preview": val_df["sample_id"].astype(str).head(10).tolist(),
        "test_sample_ids_preview": test_df["sample_id"].astype(str).head(10).tolist(),
    }

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
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> DatasetBundle:
    from torch_geometric.data import Data

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
    )

    split_map = {
        "train": set(train_df["sample_id"].astype(str).tolist()),
        "val": set(val_df["sample_id"].astype(str).tolist()),
        "test": set(test_df["sample_id"].astype(str).tolist()),
    }
    merged_by_id = sample_table.set_index("sample_id")

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

        processed_graph = preprocess_sample_graph(
            sample_graph,
            keep_top_k_edges=keep_top_k_edges,
            min_edge_weight=min_edge_weight,
        )
        edge_index, edge_attr = _build_edges(processed_graph, node_order)
        clinical = torch.tensor(sample_meta[clinical_columns].to_numpy(dtype=float), dtype=torch.float32)
        metabolites = torch.tensor(sample_meta[metabolite_columns].to_numpy(dtype=float), dtype=torch.float32)
        time = torch.tensor(float(sample_meta["time"]), dtype=torch.float32)
        event = torch.tensor(float(sample_meta["event"]), dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
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
    }
    data_summary["graph_preprocess"] = {
        "keep_top_k_edges": keep_top_k_edges,
        "min_edge_weight": min_edge_weight,
    }
    data_summary["split_summary"] = split_summary

    return DatasetBundle(
        train_set=data_list_by_split["train"],
        val_set=data_list_by_split["val"],
        test_set=data_list_by_split["test"],
        node_feature_dim=len(node_feature_columns),
        clinical_dim=len(clinical_columns),
        metabolite_dim=len(metabolite_columns),
        task_definition=get_survival_task_definition(),
        data_summary=data_summary,
        split_summary=split_summary,
    )
