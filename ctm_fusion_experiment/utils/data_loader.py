from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


class FrozenFeatureScaler:
    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def fit(self, features: np.ndarray) -> "FrozenFeatureScaler":
        self._scaler.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self._scaler.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(features)


def make_cv_splits(
    sample_table: pd.DataFrame,
    folds: int,
    seed: int,
    val_ratio: float,
) -> list[FoldSplit]:
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between zero and one.")

    sample_ids = sample_table["sample_id"].astype(str).to_numpy()
    events = sample_table["event"].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = []

    for fold, (train_val_index, test_index) in enumerate(splitter.split(sample_ids, events), start=1):
        train_val_ids = sample_ids[train_val_index]
        train_val_events = events[train_val_index]
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio,
            random_state=seed + fold,
            shuffle=True,
            stratify=train_val_events,
        )
        splits.append(
            FoldSplit(
                fold=fold,
                train_ids=tuple(sorted(train_ids.tolist())),
                val_ids=tuple(sorted(val_ids.tolist())),
                test_ids=tuple(sorted(sample_ids[test_index].tolist())),
            )
        )

    return splits


@dataclass(frozen=True)
class GraphDatasetBundle:
    graphs_by_id: dict[str, Any]
    sample_table: pd.DataFrame
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int


@dataclass(frozen=True)
class FusionArraySet:
    sample_ids: tuple[str, ...]
    graph: np.ndarray
    clinical: np.ndarray
    metabolite: np.ndarray
    time: np.ndarray
    event: np.ndarray


@dataclass(frozen=True)
class FusionArrays:
    train: FusionArraySet
    val: FusionArraySet
    test: FusionArraySet


def load_graph_dataset(config: dict, sample_limit: int | None = None) -> GraphDatasetBundle:
    import torch
    from torch_geometric.data import Data

    from research.data import _build_edges, build_sample_table, load_research_tables, preprocess_sample_graph

    graph_df, clinical_df, metabolite_df, label_df, _ = load_research_tables(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
    )
    sample_table = build_sample_table(clinical_df, metabolite_df, label_df)
    if sample_limit is not None and sample_limit < len(sample_table):
        sample_table, _ = train_test_split(
            sample_table,
            train_size=sample_limit,
            random_state=int(config["seed"]),
            shuffle=True,
            stratify=sample_table["event"].astype(int),
        )
        sample_table = sample_table.reset_index(drop=True)
        selected_ids = set(sample_table["sample_id"].astype(str))
        graph_df = graph_df.loc[graph_df["sample_id"].astype(str).isin(selected_ids)].copy()

    node_columns = list(config["model"]["node_feature_columns"])
    clinical_columns = list(config["model"]["clinical_columns"])
    metabolite_columns = list(config["model"]["metabolite_columns"])
    sample_meta = sample_table.set_index("sample_id")
    graph_preprocess = config.get("graph_preprocess", {})
    graphs_by_id = {}

    for sample_id, sample_graph in graph_df.groupby("sample_id"):
        sample_id = str(sample_id)
        node_order = sample_graph["node_name"].drop_duplicates().tolist()
        node_features = (
            sample_graph.drop_duplicates(subset=["node_name"])
            .set_index("node_name")
            .loc[node_order, node_columns]
            .to_numpy(dtype=float)
        )
        processed = preprocess_sample_graph(
            sample_graph,
            keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
            min_edge_weight=graph_preprocess.get("min_edge_weight"),
        )
        edge_index, edge_attr = _build_edges(processed, node_order)
        meta = sample_meta.loc[sample_id]
        graphs_by_id[sample_id] = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            clinical=torch.tensor(meta[clinical_columns].to_numpy(dtype=float), dtype=torch.float32),
            metabolites=torch.tensor(meta[metabolite_columns].to_numpy(dtype=float), dtype=torch.float32),
            time=torch.tensor(float(meta["time"]), dtype=torch.float32),
            event=torch.tensor(float(meta["event"]), dtype=torch.float32),
            sample_id=sample_id,
        )

    return GraphDatasetBundle(
        graphs_by_id=graphs_by_id,
        sample_table=sample_table.reset_index(drop=True),
        node_feature_dim=len(node_columns),
        clinical_dim=len(clinical_columns),
        metabolite_dim=len(metabolite_columns),
    )


def subset_graphs(graphs_by_id: dict[str, Any], sample_ids: Sequence[str]) -> list[Any]:
    return [graphs_by_id[str(sample_id)] for sample_id in sample_ids]


def prepare_fusion_arrays(
    sample_table: pd.DataFrame,
    split: FoldSplit,
    graph_embeddings: dict[str, np.ndarray],
    clinical_columns: Sequence[str],
    metabolite_columns: Sequence[str],
) -> FusionArrays:
    indexed = sample_table.copy()
    indexed["sample_id"] = indexed["sample_id"].astype(str)
    indexed = indexed.set_index("sample_id")

    def raw(sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = [str(sample_id) for sample_id in sample_ids]
        graph = np.stack([graph_embeddings[sample_id] for sample_id in ids])
        clinical = indexed.loc[ids, list(clinical_columns)].to_numpy(dtype=float)
        metabolite = indexed.loc[ids, list(metabolite_columns)].to_numpy(dtype=float)
        return graph, clinical, metabolite

    train_raw = raw(split.train_ids)
    val_raw = raw(split.val_ids)
    test_raw = raw(split.test_ids)
    graph_scaler = FrozenFeatureScaler().fit(train_raw[0])
    clinical_scaler = FrozenFeatureScaler().fit(train_raw[1])
    metabolite_scaler = FrozenFeatureScaler().fit(train_raw[2])

    def scaled(sample_ids: tuple[str, ...], values: tuple[np.ndarray, np.ndarray, np.ndarray]) -> FusionArraySet:
        return FusionArraySet(
            sample_ids=sample_ids,
            graph=graph_scaler.transform(values[0]).astype(np.float32),
            clinical=clinical_scaler.transform(values[1]).astype(np.float32),
            metabolite=metabolite_scaler.transform(values[2]).astype(np.float32),
            time=indexed.loc[list(sample_ids), "time"].to_numpy(dtype=np.float32),
            event=indexed.loc[list(sample_ids), "event"].to_numpy(dtype=np.float32),
        )

    return FusionArrays(
        train=scaled(split.train_ids, train_raw),
        val=scaled(split.val_ids, val_raw),
        test=scaled(split.test_ids, test_raw),
    )
