from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from research.data import build_dataset_from_csv


REMOVED_PRECOMPUTED_ATTRIBUTES = (
    "edge_attr",
    "node_struct",
    "node_targets",
    "graph_targets",
    "graph_cluster_targets",
)


@dataclass
class InternalEdgeDatasetBundle:
    train_set: list[Any]
    val_set: list[Any]
    test_set: list[Any]
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int
    num_node_types: int
    node_type_names: list[str]
    task_definition: dict[str, Any]
    data_summary: dict[str, Any]
    split_summary: dict[str, Any]


def validate_complete_undirected_topology(
    data: Any,
    *,
    num_node_types: int,
) -> None:
    if num_node_types < 2:
        raise ValueError("num_node_types must be at least two.")
    if data.x.ndim != 2 or int(data.x.size(0)) != num_node_types:
        raise ValueError(
            "Each internal-edge graph must contain exactly one node per type."
        )
    node_type = data.node_type.long().view(-1)
    if sorted(node_type.tolist()) != list(range(num_node_types)):
        raise ValueError("Each node type must occur exactly once per graph.")
    edge_index = data.edge_index.long()
    if edge_index.ndim != 2 or int(edge_index.size(0)) != 2:
        raise ValueError("edge_index must have shape [2, E].")
    source = edge_index[0]
    target = edge_index[1]
    if (
        (source < 0).any()
        or (target < 0).any()
        or (source >= num_node_types).any()
        or (target >= num_node_types).any()
    ):
        raise ValueError("edge_index contains an out-of-range node index.")
    source_type = node_type[source]
    target_type = node_type[target]
    if torch.any(source_type == target_type):
        raise ValueError("Self edges are prohibited.")
    counts = torch.zeros(
        (num_node_types, num_node_types), dtype=torch.long
    )
    counts.index_put_(
        (source_type.cpu(), target_type.cpu()),
        torch.ones_like(source_type.cpu()),
        accumulate=True,
    )
    expected = torch.ones_like(counts)
    expected.fill_diagonal_(0)
    if not torch.equal(counts, expected):
        raise ValueError(
            "The topology must contain every directed half of the complete "
            "undirected graph exactly once."
        )


def sanitize_graph_item(
    data: Any,
    *,
    num_node_types: int,
) -> Any:
    validate_complete_undirected_topology(
        data, num_node_types=num_node_types
    )
    sanitized = data.clone()
    for attribute in REMOVED_PRECOMPUTED_ATTRIBUTES:
        if attribute in sanitized:
            del sanitized[attribute]
    sanitized.internal_edge_policy = "computed_inside_forward"
    return sanitized


def sanitize_graph_sequence(
    data_set: Sequence[Any],
    *,
    num_node_types: int,
) -> list[Any]:
    return [
        sanitize_graph_item(item, num_node_types=num_node_types)
        for item in data_set
    ]


def build_internal_edge_dataset_from_csv(
    *,
    graph_csv: str,
    clinical_csv: str,
    metabolite_csv: str,
    label_csv: str,
    node_feature_columns: list[str],
    clinical_columns: list[str],
    metabolite_columns: list[str],
    seed: int = 42,
    split_seed: int | None = None,
    standardize_tabular: bool = True,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    validation_group: str | int | None = None,
    test_group: str | int | None = None,
) -> InternalEdgeDatasetBundle:
    bundle = build_dataset_from_csv(
        graph_csv=graph_csv,
        clinical_csv=clinical_csv,
        metabolite_csv=metabolite_csv,
        label_csv=label_csv,
        node_feature_columns=node_feature_columns,
        clinical_columns=clinical_columns,
        metabolite_columns=metabolite_columns,
        seed=seed,
        split_seed=split_seed,
        keep_top_k_edges=None,
        min_edge_weight=None,
        standardize_tabular=standardize_tabular,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        validation_group=validation_group,
        test_group=test_group,
    )
    train_set = sanitize_graph_sequence(
        bundle.train_set, num_node_types=bundle.num_node_types
    )
    val_set = sanitize_graph_sequence(
        bundle.val_set, num_node_types=bundle.num_node_types
    )
    test_set = sanitize_graph_sequence(
        bundle.test_set, num_node_types=bundle.num_node_types
    )
    summary = dict(bundle.data_summary)
    summary["feature_dimensions"] = dict(
        summary.get("feature_dimensions", {})
    )
    summary["feature_dimensions"]["precomputed_structure_features"] = False
    summary["internal_edge_adapter"] = {
        "complete_topology_only": True,
        "removed_attributes": list(REMOVED_PRECOMPUTED_ATTRIBUTES),
        "precomputed_edge_weight_used": False,
        "edge_filtering_used": False,
    }
    return InternalEdgeDatasetBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        node_feature_dim=bundle.node_feature_dim,
        clinical_dim=bundle.clinical_dim,
        metabolite_dim=bundle.metabolite_dim,
        num_node_types=bundle.num_node_types,
        node_type_names=list(bundle.node_type_names),
        task_definition=dict(bundle.task_definition),
        data_summary=summary,
        split_summary=dict(bundle.split_summary),
    )
