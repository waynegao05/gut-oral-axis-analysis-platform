from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation


@dataclass(frozen=True)
class EdgeState:
    pair_logits: torch.Tensor
    pair_weights: torch.Tensor
    directed_weights: torch.Tensor
    edge_matrix: torch.Tensor
    delta_regularization: torch.Tensor
    saturation_regularization: torch.Tensor


def _num_graphs(batch: Any) -> int:
    value = getattr(batch, "num_graphs", None)
    if value is not None:
        return int(value)
    batch_index = batch.batch.long()
    return int(batch_index.max().item()) + 1 if batch_index.numel() else 0


def canonical_node_tensor(
    batch: Any,
    *,
    num_node_types: int,
) -> torch.Tensor:
    if not torch.isfinite(batch.x).all():
        raise ValueError("Node features contain NaN or infinite values.")
    num_graphs = _num_graphs(batch)
    if num_graphs <= 0:
        raise ValueError("At least one graph is required.")
    node_type = batch.node_type.long().view(-1)
    graph_index = batch.batch.long().view(-1)
    if (
        node_type.numel() != batch.x.size(0)
        or graph_index.numel() != batch.x.size(0)
    ):
        raise ValueError("Node metadata does not align with node features.")
    if (
        (node_type < 0).any()
        or (node_type >= num_node_types).any()
        or (graph_index < 0).any()
        or (graph_index >= num_graphs).any()
    ):
        raise ValueError("Node type or graph index is out of range.")
    values = batch.x.new_zeros(
        (num_graphs, num_node_types, batch.x.size(1))
    )
    counts = torch.zeros(
        (num_graphs, num_node_types),
        dtype=torch.long,
        device=batch.x.device,
    )
    values.index_put_(
        (graph_index, node_type), batch.x, accumulate=True
    )
    counts.index_put_(
        (graph_index, node_type),
        torch.ones_like(node_type),
        accumulate=True,
    )
    if not torch.all(counts == 1):
        raise ValueError("Every graph must contain each node type exactly once.")
    return values


class SymmetricSampleEdgeGenerator(nn.Module):
    MODES = {"equal", "global", "node", "node_context"}

    def __init__(
        self,
        *,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        num_node_types: int,
        hidden_dim: int = 24,
        node_identity_dim: int = 8,
        mode: str = "node_context",
        minimum_weight: float = 0.02,
        residual_logit_scale: float = 0.75,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported edge mode: {mode}")
        if num_node_types < 2:
            raise ValueError("num_node_types must be at least two.")
        if not 0.0 <= minimum_weight < 1.0:
            raise ValueError("minimum_weight must be in [0, 1).")
        self.node_feature_dim = int(node_feature_dim)
        self.clinical_dim = int(clinical_dim)
        self.metabolite_dim = int(metabolite_dim)
        self.num_node_types = int(num_node_types)
        self.mode = str(mode)
        self.minimum_weight = float(minimum_weight)
        self.residual_logit_scale = float(residual_logit_scale)
        upper = torch.triu_indices(
            num_node_types, num_node_types, offset=1
        )
        self.register_buffer("pair_source", upper[0], persistent=False)
        self.register_buffer("pair_target", upper[1], persistent=False)
        num_pairs = int(upper.size(1))
        pair_lookup = torch.full(
            (num_node_types, num_node_types), -1, dtype=torch.long
        )
        for pair_id, (source, target) in enumerate(upper.t().tolist()):
            pair_lookup[source, target] = pair_id
            pair_lookup[target, source] = pair_id
        self.register_buffer("pair_lookup", pair_lookup, persistent=False)

        self.node_identity = nn.Embedding(
            num_node_types, node_identity_dim
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim + node_identity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        context_input_dim = clinical_dim + metabolite_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pair_identity = nn.Embedding(num_pairs, hidden_dim)
        pair_input_dim = hidden_dim * 5
        self.pair_network = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_bias = nn.Parameter(torch.zeros(num_pairs))
        nn.init.zeros_(self.pair_network[-1].weight)
        nn.init.zeros_(self.pair_network[-1].bias)

    @property
    def num_pairs(self) -> int:
        return int(self.pair_source.numel())

    def _tabular_context(
        self, batch: Any, *, num_graphs: int
    ) -> torch.Tensor:
        clinical = batch.clinical.view(num_graphs, self.clinical_dim)
        metabolites = batch.metabolites.view(
            num_graphs, self.metabolite_dim
        )
        tabular = torch.cat([clinical, metabolites], dim=1)
        if not torch.isfinite(tabular).all():
            raise ValueError("Clinical or metabolite inputs are not finite.")
        return self.context_encoder(tabular)

    def _validate_and_expand(
        self,
        batch: Any,
        pair_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = batch.edge_index.long()
        source_node = edge_index[0]
        target_node = edge_index[1]
        if source_node.numel() == 0:
            raise ValueError("The graph cannot be edge-free.")
        source_graph = batch.batch[source_node].long()
        target_graph = batch.batch[target_node].long()
        if not torch.equal(source_graph, target_graph):
            raise ValueError("Cross-graph edges are prohibited.")
        source_type = batch.node_type[source_node].long()
        target_type = batch.node_type[target_node].long()
        if torch.any(source_type == target_type):
            raise ValueError("Self edges are prohibited.")
        pair_id = self.pair_lookup[source_type, target_type]
        if torch.any(pair_id < 0):
            raise ValueError("An edge has no canonical undirected pair.")
        num_graphs = int(pair_weights.size(0))
        directed_counts = torch.zeros(
            (num_graphs, self.num_node_types, self.num_node_types),
            dtype=torch.long,
            device=edge_index.device,
        )
        directed_counts.index_put_(
            (source_graph, source_type, target_type),
            torch.ones_like(source_type),
            accumulate=True,
        )
        expected = torch.ones_like(directed_counts)
        diagonal = torch.arange(
            self.num_node_types, device=edge_index.device
        )
        expected[:, diagonal, diagonal] = 0
        if not torch.equal(directed_counts, expected):
            raise ValueError(
                "Every canonical edge direction must occur exactly once."
            )
        directed_weights = pair_weights[source_graph, pair_id]
        edge_matrix = pair_weights.new_zeros(
            (num_graphs, self.num_node_types, self.num_node_types)
        )
        edge_matrix[
            :, self.pair_source, self.pair_target
        ] = pair_weights
        edge_matrix[
            :, self.pair_target, self.pair_source
        ] = pair_weights
        return directed_weights, edge_matrix

    def forward(self, batch: Any) -> EdgeState:
        canonical = canonical_node_tensor(
            batch, num_node_types=self.num_node_types
        )
        num_graphs = int(canonical.size(0))
        pair_bias = self.pair_bias.view(1, -1).expand(num_graphs, -1)
        if self.mode == "equal":
            logits = torch.zeros_like(pair_bias)
        elif self.mode == "global":
            logits = pair_bias
        else:
            identity = self.node_identity.weight.view(
                1, self.num_node_types, -1
            ).expand(num_graphs, -1, -1)
            node_state = self.node_encoder(
                torch.cat([canonical, identity], dim=2)
            )
            source = node_state[:, self.pair_source, :]
            target = node_state[:, self.pair_target, :]
            if self.mode == "node_context":
                context = self._tabular_context(
                    batch, num_graphs=num_graphs
                )
            else:
                context = torch.zeros_like(source[:, 0, :])
            context = context.unsqueeze(1).expand(
                -1, self.num_pairs, -1
            )
            pair_identity = self.pair_identity.weight.unsqueeze(0).expand(
                num_graphs, -1, -1
            )
            pair_features = torch.cat(
                [
                    source + target,
                    torch.abs(source - target),
                    source * target,
                    context,
                    pair_identity,
                ],
                dim=2,
            )
            residual = self.pair_network(pair_features).squeeze(-1)
            logits = pair_bias + self.residual_logit_scale * torch.tanh(
                residual
            )
        weights = self.minimum_weight + (
            1.0 - self.minimum_weight
        ) * torch.sigmoid(logits)
        directed, edge_matrix = self._validate_and_expand(batch, weights)
        pair_mean = weights.mean(dim=0, keepdim=True)
        delta_regularization = torch.mean((weights - pair_mean) ** 2)
        saturation_regularization = torch.mean(
            F.relu(torch.abs(logits) - 5.0) ** 2
        )
        return EdgeState(
            pair_logits=logits,
            pair_weights=weights,
            directed_weights=directed,
            edge_matrix=edge_matrix,
            delta_regularization=delta_regularization,
            saturation_regularization=saturation_regularization,
        )


class InternalEdgeGATDualSurvivalModel(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        num_node_types: int,
        hidden_dim: int = 48,
        heads: int = 2,
        dropout: float = 0.1,
        edge_hidden_dim: int = 16,
        node_identity_dim: int = 8,
        edge_mode: str = "node_context",
        num_time_bins: int = 12,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.clinical_dim = int(clinical_dim)
        self.metabolite_dim = int(metabolite_dim)
        self.num_node_types = int(num_node_types)
        self.num_time_bins = int(num_time_bins)
        self.dropout = nn.Dropout(dropout)
        self.edge_generator = SymmetricSampleEdgeGenerator(
            node_feature_dim=node_feature_dim,
            clinical_dim=clinical_dim,
            metabolite_dim=metabolite_dim,
            num_node_types=num_node_types,
            hidden_dim=max(16, hidden_dim // 2),
            node_identity_dim=node_identity_dim,
            mode=edge_mode,
        )
        self.node_identity = nn.Embedding(
            num_node_types, node_identity_dim
        )
        structure_dim = 5
        self.node_projection = nn.Sequential(
            nn.Linear(
                node_feature_dim + node_identity_dim + structure_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
        )
        first_dim = hidden_dim * heads
        self.conv1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_hidden_dim,
        )
        self.residual1 = nn.Linear(hidden_dim, first_dim)
        self.norm1 = nn.LayerNorm(first_dim)
        self.conv2 = GATConv(
            first_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_hidden_dim,
        )
        self.residual2 = nn.Linear(first_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
                nn.GELU(),
                nn.Linear(max(16, hidden_dim // 2), 1),
            )
        )
        graph_dim = hidden_dim * 3
        pair_count = num_node_types * (num_node_types - 1) // 2
        self.identity_projection = nn.Sequential(
            nn.Linear(
                num_node_types * node_feature_dim + pair_count,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        composition_dim = (
            num_node_types
            + pair_count
            + 2
            + num_node_types
            + num_node_types
        )
        self.composition_projection = nn.Sequential(
            nn.Linear(composition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        tabular_dim = clinical_dim + metabolite_dim
        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        fusion_dim = graph_dim + hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cox_head = nn.Linear(hidden_dim, 1)
        self.discrete_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_time_bins),
        )

    def _structure_features(
        self, batch: Any, edge_matrix: torch.Tensor
    ) -> torch.Tensor:
        incident = edge_matrix
        mask = (
            1.0
            - torch.eye(
                self.num_node_types,
                device=incident.device,
                dtype=incident.dtype,
            )
        ).unsqueeze(0)
        count = float(self.num_node_types - 1)
        mean = (incident * mask).sum(dim=2) / count
        variance = (
            ((incident - mean.unsqueeze(2)) * mask) ** 2
        ).sum(dim=2) / count
        masked_max = incident.masked_fill(mask == 0, float("-inf")).max(
            dim=2
        ).values
        masked_min = incident.masked_fill(mask == 0, float("inf")).min(
            dim=2
        ).values
        normalized = incident / torch.clamp(
            incident.sum(dim=2, keepdim=True), min=1e-8
        )
        entropy = -(
            normalized
            * torch.log(torch.clamp(normalized, min=1e-8))
            * mask
        ).sum(dim=2) / torch.log(
            incident.new_tensor(float(self.num_node_types - 1))
        )
        canonical = torch.stack(
            [mean, torch.sqrt(variance + 1e-8), masked_max, masked_min, entropy],
            dim=2,
        )
        return canonical[
            batch.batch.long(), batch.node_type.long()
        ]

    def _composition_features(
        self, canonical_nodes: torch.Tensor
    ) -> torch.Tensor:
        abundance = torch.clamp(canonical_nodes[:, :, 0], min=0.0)
        closure = (abundance + 1e-4) / torch.clamp(
            (abundance + 1e-4).sum(dim=1, keepdim=True), min=1e-8
        )
        log_abundance = torch.log(closure)
        clr = log_abundance - log_abundance.mean(dim=1, keepdim=True)
        source = self.edge_generator.pair_source
        target = self.edge_generator.pair_target
        log_ratios = log_abundance[:, source] - log_abundance[:, target]
        entropy = -(
            closure * torch.log(torch.clamp(closure, min=1e-8))
        ).sum(dim=1, keepdim=True)
        simpson = (
            1.0 - torch.sum(closure**2, dim=1, keepdim=True)
        )
        if self.node_feature_dim > 1:
            function = canonical_nodes[:, :, 1]
        else:
            function = torch.zeros_like(abundance)
        interactions = closure * function
        return torch.cat(
            [clr, log_ratios, entropy, simpson, function, interactions],
            dim=1,
        )

    def forward(
        self, batch: Any, compute_contrastive: bool = False
    ) -> dict[str, torch.Tensor]:
        del compute_contrastive
        canonical_nodes = canonical_node_tensor(
            batch, num_node_types=self.num_node_types
        )
        edge_state = self.edge_generator(batch)
        structure = self._structure_features(
            batch, edge_state.edge_matrix
        )
        identity = self.node_identity(batch.node_type.long())
        x = self.node_projection(
            torch.cat([batch.x, identity, structure], dim=1)
        )
        weight = edge_state.directed_weights.view(-1, 1)
        edge_features = self.edge_encoder(
            torch.cat(
                [weight, weight**2, torch.log1p(weight)],
                dim=1,
            )
        )
        first = self.conv1(x, batch.edge_index, edge_features)
        first = self.norm1(first + self.residual1(x))
        first = self.dropout(F.gelu(first))
        second = self.conv2(first, batch.edge_index, edge_features)
        second = self.norm2(second + self.residual2(first))
        second = self.dropout(F.gelu(second))
        batch_index = batch.batch.long()
        graph_embedding = torch.cat(
            [
                global_mean_pool(second, batch_index),
                global_max_pool(second, batch_index),
                self.attention_pool(second, index=batch_index),
            ],
            dim=1,
        )
        identity_input = torch.cat(
            [
                canonical_nodes.flatten(start_dim=1),
                edge_state.pair_weights,
            ],
            dim=1,
        )
        identity_embedding = self.identity_projection(identity_input)
        composition_embedding = self.composition_projection(
            self._composition_features(canonical_nodes)
        )
        num_graphs = int(canonical_nodes.size(0))
        clinical = batch.clinical.view(num_graphs, self.clinical_dim)
        metabolites = batch.metabolites.view(
            num_graphs, self.metabolite_dim
        )
        tabular = torch.cat([clinical, metabolites], dim=1)
        if not torch.isfinite(tabular).all():
            raise ValueError("Tabular model inputs are not finite.")
        tabular_embedding = self.tabular_projection(tabular)
        latent = self.fusion(
            torch.cat(
                [
                    graph_embedding,
                    identity_embedding,
                    composition_embedding,
                    tabular_embedding,
                ],
                dim=1,
            )
        )
        risk = self.cox_head(latent).squeeze(-1)
        time_logits = self.discrete_head(latent)
        hazard = torch.sigmoid(time_logits).clamp(
            min=1e-6, max=1.0 - 1e-6
        )
        survival = torch.cumprod(1.0 - hazard, dim=1)
        discrete_risk = -survival.sum(dim=1)
        return {
            "risk": risk,
            "time_logits": time_logits,
            "discrete_risk": discrete_risk,
            "latent": latent,
            "graph_embedding": graph_embedding,
            "pair_edge_logits": edge_state.pair_logits,
            "pair_edge_weights": edge_state.pair_weights,
            "directed_edge_weights": edge_state.directed_weights,
            "edge_delta_regularization": edge_state.delta_regularization,
            "edge_saturation_regularization": (
                edge_state.saturation_regularization
            ),
            "contrastive_loss": risk.new_zeros(()),
            "graph_aux_loss": risk.new_zeros(()),
            "node_aux_loss": risk.new_zeros(()),
            "aux_loss": risk.new_zeros(()),
        }
