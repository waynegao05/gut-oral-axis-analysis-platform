from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.topology_v7_compositional_temporal_v1.model import (
    EdgeState,
    canonical_node_tensor,
)
from experiments.topology_v7_internalized_edge_v2.model import (
    internal_complete_graph_structure,
)


class SiteConditionedEdgeGenerator(nn.Module):
    MODES = {"equal", "site_context"}

    def __init__(
        self,
        *,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        site_feature_dim: int,
        num_node_types: int,
        hidden_dim: int = 32,
        node_identity_dim: int = 8,
        mode: str = "site_context",
        maximum_log_weight_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported internal edge mode: {mode}")
        if num_node_types < 2 or site_feature_dim <= 0:
            raise ValueError("Node and site feature dimensions must be positive.")
        self.node_feature_dim = int(node_feature_dim)
        self.clinical_dim = int(clinical_dim)
        self.metabolite_dim = int(metabolite_dim)
        self.site_feature_dim = int(site_feature_dim)
        self.num_node_types = int(num_node_types)
        self.mode = str(mode)
        self.maximum_log_weight_delta = float(maximum_log_weight_delta)

        upper = torch.triu_indices(
            self.num_node_types,
            self.num_node_types,
            offset=1,
        )
        self.register_buffer("pair_source", upper[0], persistent=False)
        self.register_buffer("pair_target", upper[1], persistent=False)
        pair_lookup = torch.full(
            (self.num_node_types, self.num_node_types),
            -1,
            dtype=torch.long,
        )
        for pair_id, (source, target) in enumerate(upper.t().tolist()):
            pair_lookup[source, target] = pair_id
            pair_lookup[target, source] = pair_id
        self.register_buffer("pair_lookup", pair_lookup, persistent=False)

        num_pairs = int(upper.size(1))
        self.node_identity = nn.Embedding(
            self.num_node_types,
            int(node_identity_dim),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(
                self.node_feature_dim + int(node_identity_dim),
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        context_dim = (
            self.clinical_dim
            + self.metabolite_dim
            + self.site_feature_dim
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pair_identity = nn.Embedding(num_pairs, hidden_dim)
        self.pair_network = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.pair_network[-1].weight)
        nn.init.zeros_(self.pair_network[-1].bias)

    @property
    def num_pairs(self) -> int:
        return int(self.pair_source.numel())

    def _expand(
        self,
        batch: Any,
        pair_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_node = batch.edge_index[0].long()
        target_node = batch.edge_index[1].long()
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
            raise ValueError("An edge has no canonical pair.")
        num_graphs = int(pair_weights.size(0))
        counts = torch.zeros(
            (num_graphs, self.num_node_types, self.num_node_types),
            dtype=torch.long,
            device=source_node.device,
        )
        counts.index_put_(
            (source_graph, source_type, target_type),
            torch.ones_like(source_type),
            accumulate=True,
        )
        expected = torch.ones_like(counts)
        diagonal = torch.arange(self.num_node_types, device=counts.device)
        expected[:, diagonal, diagonal] = 0
        if not torch.equal(counts, expected):
            raise ValueError("Every complete-graph edge direction is required.")
        directed = pair_weights[source_graph, pair_id]
        matrix = pair_weights.new_zeros(
            (num_graphs, self.num_node_types, self.num_node_types)
        )
        matrix[:, self.pair_source, self.pair_target] = pair_weights
        matrix[:, self.pair_target, self.pair_source] = pair_weights
        return directed, matrix

    def forward(self, batch: Any) -> EdgeState:
        canonical = canonical_node_tensor(
            batch,
            num_node_types=self.num_node_types,
        )
        num_graphs = int(canonical.size(0))
        if self.mode == "equal":
            logits = canonical.new_zeros((num_graphs, self.num_pairs))
        else:
            identity = self.node_identity.weight.unsqueeze(0).expand(
                num_graphs,
                -1,
                -1,
            )
            node_state = self.node_encoder(
                torch.cat([canonical, identity], dim=2)
            )
            source = node_state[:, self.pair_source, :]
            target = node_state[:, self.pair_target, :]
            site = batch.site_features.view(
                num_graphs,
                self.site_feature_dim,
            )
            clinical = batch.clinical.view(num_graphs, self.clinical_dim)
            metabolites = batch.metabolites.view(
                num_graphs,
                self.metabolite_dim,
            )
            context_values = torch.cat(
                [clinical, metabolites, site],
                dim=1,
            )
            if not torch.isfinite(context_values).all():
                raise ValueError("Internal edge context is not finite.")
            context = self.context_encoder(context_values)
            context = context.unsqueeze(1).expand(-1, self.num_pairs, -1)
            pair_identity = self.pair_identity.weight.unsqueeze(0).expand(
                num_graphs,
                -1,
                -1,
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
            logits = self.pair_network(pair_features).squeeze(-1)

        weights = torch.exp(
            self.maximum_log_weight_delta * torch.tanh(logits)
        )
        directed, matrix = self._expand(batch, weights)
        return EdgeState(
            pair_logits=logits,
            pair_weights=weights,
            directed_weights=directed,
            edge_matrix=matrix,
            delta_regularization=torch.mean((weights - 1.0) ** 2),
            saturation_regularization=torch.mean(
                F.relu(torch.abs(logits) - 5.0) ** 2
            ),
        )


class InternalRelationSiteModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        site_feature_dim: int,
        num_node_types: int,
        edge_mode: str,
        use_site_residual: bool,
        edge_hidden_dim: int = 32,
        site_hidden_dim: int = 32,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_node_types = int(num_node_types)
        self.site_feature_dim = int(site_feature_dim)
        self.use_site_residual = bool(use_site_residual)
        self.residual_scale = float(residual_scale)
        self.edge_generator = SiteConditionedEdgeGenerator(
            node_feature_dim=node_feature_dim,
            clinical_dim=clinical_dim,
            metabolite_dim=metabolite_dim,
            site_feature_dim=site_feature_dim,
            num_node_types=num_node_types,
            hidden_dim=edge_hidden_dim,
            mode=edge_mode,
        )
        latent_dim = int(base_model.risk_head.in_features)
        self.site_encoder = nn.Sequential(
            nn.Linear(site_feature_dim, site_hidden_dim),
            nn.LayerNorm(site_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(site_hidden_dim, site_hidden_dim),
            nn.GELU(),
        )
        self.site_residual_head = nn.Sequential(
            nn.Linear(latent_dim + site_hidden_dim, site_hidden_dim),
            nn.GELU(),
            nn.Linear(site_hidden_dim, 1),
        )
        nn.init.zeros_(self.site_residual_head[-1].weight)
        nn.init.zeros_(self.site_residual_head[-1].bias)
        self.aft_location_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )
        self.aft_log_scale = nn.Parameter(torch.tensor(-0.35))

    def initialize_aft_location(self, mean_log_time: float) -> None:
        nn.init.zeros_(self.aft_location_head[-1].weight)
        nn.init.constant_(
            self.aft_location_head[-1].bias,
            float(mean_log_time),
        )

    def forward(
        self,
        batch: Any,
        compute_contrastive: bool = False,
    ) -> dict[str, torch.Tensor]:
        edge_state = self.edge_generator(batch)
        internal_batch = batch.clone()
        internal_batch.edge_attr = edge_state.directed_weights.view(-1, 1)
        (
            internal_batch.node_struct,
            internal_batch.node_targets,
            internal_batch.graph_targets,
            internal_batch.graph_cluster_targets,
        ) = internal_complete_graph_structure(
            internal_batch,
            edge_state,
            num_node_types=self.num_node_types,
        )
        output = self.base_model(
            internal_batch,
            compute_contrastive=compute_contrastive,
        )
        if self.use_site_residual:
            site = batch.site_features.view(-1, self.site_feature_dim)
            site_embedding = self.site_encoder(site)
            raw_residual = self.site_residual_head(
                torch.cat([output["latent"], site_embedding], dim=1)
            ).squeeze(-1)
            residual = self.residual_scale * torch.tanh(raw_residual)
        else:
            residual = torch.zeros_like(output["risk"])
        aft_location = self.aft_location_head(output["latent"]).squeeze(-1)
        aft_log_scale = torch.clamp(
            self.aft_log_scale,
            min=-2.0,
            max=1.0,
        )
        output["base_risk"] = output["risk"]
        output["risk"] = output["risk"] + residual
        output.update(
            {
                "site_risk_residual": residual,
                "pair_edge_logits": edge_state.pair_logits,
                "pair_edge_weights": edge_state.pair_weights,
                "directed_edge_weights": edge_state.directed_weights,
                "edge_delta_regularization": (
                    edge_state.delta_regularization
                ),
                "edge_saturation_regularization": (
                    edge_state.saturation_regularization
                ),
                "aft_location": aft_location,
                "aft_log_scale": aft_log_scale,
            }
        )
        return output


def internal_relation_regularization(
    output: dict[str, torch.Tensor],
    *,
    edge_weight: float,
    site_residual_weight: float,
) -> torch.Tensor:
    return (
        float(edge_weight) * output["edge_delta_regularization"]
        + 1e-4 * output["edge_saturation_regularization"]
        + float(site_residual_weight)
        * torch.mean(output["site_risk_residual"] ** 2)
    )
