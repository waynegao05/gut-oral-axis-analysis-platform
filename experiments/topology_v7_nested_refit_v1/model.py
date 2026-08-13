from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


def canonical_edge_vector(
    batch: Any,
    *,
    num_node_types: int,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    if num_node_types < 2:
        raise ValueError("num_node_types must be at least two.")
    num_graphs = int(batch.num_graphs)
    device = batch.edge_attr.device
    values = torch.zeros(
        (num_graphs, num_node_types, num_node_types),
        dtype=batch.edge_attr.dtype,
        device=device,
    )
    counts = torch.zeros(
        (num_graphs, num_node_types, num_node_types),
        dtype=torch.long,
        device=device,
    )
    edge_graph = batch.batch[batch.edge_index[0]].long()
    source_type = batch.node_type[batch.edge_index[0]].long()
    target_type = batch.node_type[batch.edge_index[1]].long()
    edge_weight = batch.edge_attr.view(-1)
    values.index_put_(
        (edge_graph, source_type, target_type),
        edge_weight,
        accumulate=True,
    )
    counts.index_put_(
        (edge_graph, source_type, target_type),
        torch.ones_like(source_type),
        accumulate=True,
    )
    upper = torch.triu_indices(
        num_node_types,
        num_node_types,
        offset=1,
        device=device,
    )
    upper_counts = counts[:, upper[0], upper[1]]
    lower_counts = counts[:, upper[1], upper[0]]
    if not torch.all(upper_counts == 1) or not torch.all(lower_counts == 1):
        raise ValueError(
            "Every canonical undirected edge must occur exactly once in each direction."
        )
    upper_values = values[:, upper[0], upper[1]]
    lower_values = values[:, upper[1], upper[0]]
    if not torch.allclose(upper_values, lower_values, atol=tolerance, rtol=0.0):
        raise ValueError("Opposite directions of an undirected edge must have equal weight.")
    return upper_values


def fit_edge_standardizer(
    data_set: Sequence[Any],
    *,
    num_node_types: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch_geometric.loader import DataLoader

    loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)
    batch = next(iter(loader))
    values = canonical_edge_vector(batch, num_node_types=num_node_types)
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    scale = torch.where(scale <= 1e-12, torch.ones_like(scale), scale)
    return mean.detach().cpu(), scale.detach().cpu()


class FixedEdgeResidualModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_node_types: int,
        edge_mean: torch.Tensor,
        edge_scale: torch.Tensor,
        hidden_dim: int = 16,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        expected_edges = num_node_types * (num_node_types - 1) // 2
        if edge_mean.numel() != expected_edges or edge_scale.numel() != expected_edges:
            raise ValueError("Edge standardizer dimension does not match node types.")
        self.base_model = base_model
        self.num_node_types = int(num_node_types)
        self.residual_scale = float(residual_scale)
        self.register_buffer("edge_mean", edge_mean.float().view(1, -1))
        self.register_buffer("edge_scale", edge_scale.float().view(1, -1))
        self.edge_network = nn.Sequential(
            nn.Linear(expected_edges, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max(4, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(4, hidden_dim // 2), 1),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, batch: Any, compute_contrastive: bool = False) -> dict[str, torch.Tensor]:
        output = self.base_model(batch, compute_contrastive=compute_contrastive)
        edge_values = canonical_edge_vector(
            batch,
            num_node_types=self.num_node_types,
        )
        standardized = (edge_values - self.edge_mean) / self.edge_scale
        edge_delta = self.edge_network(standardized).squeeze(-1)
        gate_value = torch.tanh(self.gate)
        output["risk"] = (
            output["risk"]
            + self.residual_scale * gate_value * edge_delta
        )
        output["fixed_edge_delta"] = edge_delta
        output["fixed_edge_gate"] = gate_value
        return output
