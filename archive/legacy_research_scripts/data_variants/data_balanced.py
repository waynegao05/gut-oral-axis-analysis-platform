from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
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
    random.seed(seed)
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


def _allocate_group(indices: List[int], val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    n = len(indices)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    if n >= 3 and n_test == 0 and test_ratio > 0:
        n_test = 1
    if n - n_test >= 2 and n_val == 0 and val_ratio > 0:
        n_val = 1

    if n_test + n_val >= n:
        if n >= 3:
            n_test = 1
            n_val = 1
        elif n == 2:
            n_test = 1
            n_val = 0
        else:
            n_test = 0
            n_val = 0

    test_part = indices[:n_test]
    val_part = indices[n_test:n_test + n_val]
    train_part = indices[n_test + n_val:]
    return train_part, val_part, test_part


def _balanced_split_indices(meta_df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    df = meta_df.copy().reset_index(drop=True)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for event_value in sorted(df["event"].unique().tolist()):
        group = df[df["event"] == event_value].copy()
        group = group.sort_values("time").reset_index()
        group_indices = group["index"].tolist()

        blocks: List[List[int]] = []
        block_size = 3
        for start in range(0, len(group_indices), block_size):
            block = group_indices[start:start + block_size]
            rng.shuffle(block)
            blocks.append(block)

        shuffled_indices = [idx for block in blocks for idx in block]
        group_train, group_val, group_test = _allocate_group(shuffled_indices, val_ratio, test_ratio)

        train_idx.extend(group_train)
        val_idx.extend(group_val)
        test_idx.extend(group_test)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


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
    test_ratio: float = 0.2,
) -> DatasetBundle:
    set_seed(seed)
    graph_df = pd.read_csv(graph_csv)
    clinical_df = pd.read_csv(clinical_csv)
    metabolite_df = pd.read_csv(metabolite_csv)
    label_df = pd.read_csv(label_csv)

    merged = clinical_df.merge(metabolite_df, on="sample_id").merge(label_df, on="sample_id")
    data_list: List[Data] = []
    sample_order: List[str] = []

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
        sample_order.append(str(sample_id))

    meta_for_split = merged[merged["sample_id"].isin(sample_order)].copy()
    meta_for_split = meta_for_split.set_index("sample_id").loc[sample_order].reset_index()

    train_idx, val_idx, test_idx = _balanced_split_indices(
        meta_df=meta_for_split,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_set = [data_list[i] for i in train_idx]
    val_set = [data_list[i] for i in val_idx]
    test_set = [data_list[i] for i in test_idx]

    return DatasetBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        node_feature_dim=len(node_feature_columns),
        clinical_dim=len(clinical_columns),
        metabolite_dim=len(metabolite_columns),
    )
