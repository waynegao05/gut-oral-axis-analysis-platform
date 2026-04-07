from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data


@dataclass
class DatasetBundle:
    train_set: List[Data]
    val_set: List[Data]
    test_set: List[Data]
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_dataset_from_csv(
    graph_csv: str,
    clinical_csv: str,
    metabolite_csv: str,
    label_csv: str,
    node_feature_columns: List[str],
    clinical_columns: List[str],
    metabolite_columns: List[str],
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> DatasetBundle:
    set_seed(seed)
    graph_df = pd.read_csv(graph_csv)
    clinical_df = pd.read_csv(clinical_csv)
    metabolite_df = pd.read_csv(metabolite_csv)
    label_df = pd.read_csv(label_csv)

    merged = clinical_df.merge(metabolite_df, on="sample_id").merge(label_df, on="sample_id")
    data_list: List[Data] = []

    for sample_id, sample_graph in graph_df.groupby("sample_id"):
        sample_meta = merged.loc[merged["sample_id"] == sample_id]
        if sample_meta.empty:
            continue
        sample_meta = sample_meta.iloc[0]
        node_order = sample_graph["node_name"].drop_duplicates().tolist()
        node_features = (
            sample_graph.drop_duplicates(subset=["node_name"])
            .set_index("node_name")
            .loc[node_order, node_feature_columns]
            .to_numpy()
        )
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index, edge_attr = _build_edges(sample_graph, node_order)
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
            sample_id=str(sample_id),
        )
        data_list.append(data)

    total = len(data_list)
    test_size = max(1, int(total * test_ratio)) if total >= 3 else 0
    val_size = max(1, int(total * val_ratio)) if total - test_size >= 3 else 0
    train_size = total - val_size - test_size
    if train_size <= 0:
        train_size = max(1, total - 2)
        val_size = 1 if total >= 2 else 0
        test_size = total - train_size - val_size

    splits = random_split(data_list, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
    train_set, val_set, test_set = [list(split) for split in splits]

    return DatasetBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        node_feature_dim=len(node_feature_columns),
        clinical_dim=len(clinical_columns),
        metabolite_dim=len(metabolite_columns),
    )
