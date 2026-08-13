from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.topology_v7_compositional_temporal_v1.model import (
    EdgeState,
    SymmetricSampleEdgeGenerator,
)


def internal_complete_graph_structure(
    batch: Any,
    edge_state: EdgeState,
    *,
    num_node_types: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weighted_degree = edge_state.edge_matrix.sum(dim=2)
    maximum = torch.clamp(
        weighted_degree.max(dim=1, keepdim=True).values, min=1.0
    )
    normalized_weighted_degree = weighted_degree / maximum
    num_graphs = int(weighted_degree.size(0))
    ones = torch.ones_like(weighted_degree)
    zeros = torch.zeros_like(weighted_degree)
    triangle_score = weighted_degree.new_full(
        weighted_degree.shape,
        float((num_node_types - 1) * (num_node_types - 2))
        / float(2 * num_node_types),
    )
    canonical_node_struct = torch.stack(
        [
            ones,
            normalized_weighted_degree,
            ones,
            triangle_score,
            zeros,
        ],
        dim=2,
    )
    canonical_node_targets = torch.stack(
        [ones, normalized_weighted_degree, ones, zeros], dim=2
    )
    graph_targets = (
        0.55
        + 0.25
        * weighted_degree.mean(dim=1, keepdim=True)
        / float(num_node_types)
    )
    graph_cluster_targets = weighted_degree.new_full(
        (num_graphs, 1), 0.6
    )
    graph_index = batch.batch.long()
    node_type = batch.node_type.long()
    return (
        canonical_node_struct[graph_index, node_type],
        canonical_node_targets[graph_index, node_type],
        graph_targets,
        graph_cluster_targets,
    )


class InternalizedEdgeDropInModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        num_node_types: int,
        edge_mode: str,
        edge_hidden_dim: int = 32,
        node_identity_dim: int = 8,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_node_types = int(num_node_types)
        self.edge_generator = SymmetricSampleEdgeGenerator(
            node_feature_dim=node_feature_dim,
            clinical_dim=clinical_dim,
            metabolite_dim=metabolite_dim,
            num_node_types=num_node_types,
            hidden_dim=edge_hidden_dim,
            node_identity_dim=node_identity_dim,
            mode=edge_mode,
        )
        latent_dim = int(base_model.risk_head.in_features)
        self.aft_location_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )
        self.aft_log_scale = nn.Parameter(torch.tensor(-0.35))

    def initialize_aft_location(self, mean_log_time: float) -> None:
        nn.init.zeros_(self.aft_location_head[-1].weight)
        nn.init.constant_(
            self.aft_location_head[-1].bias, float(mean_log_time)
        )

    def forward(
        self, batch: Any, compute_contrastive: bool = False
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
            internal_batch, compute_contrastive=compute_contrastive
        )
        aft_location = self.aft_location_head(output["latent"]).squeeze(-1)
        aft_log_scale = torch.clamp(
            self.aft_log_scale, min=-2.0, max=1.0
        )
        output.update(
            {
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


def lognormal_aft_nll(
    location: torch.Tensor,
    log_scale: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    location = location.float().view(-1)
    time = time.float().view(-1)
    event = event.float().view(-1)
    if not (location.shape == time.shape == event.shape):
        raise ValueError("AFT predictions and labels must align.")
    if torch.any(time <= 0) or not torch.isfinite(time).all():
        raise ValueError("AFT survival times must be finite and positive.")
    scale = torch.exp(log_scale).clamp(min=1e-3, max=10.0)
    log_time = torch.log(time)
    standardized = (log_time - location) / scale
    event_nll = (
        log_time
        + log_scale
        + 0.5 * standardized**2
        + 0.5 * torch.log(time.new_tensor(2.0 * torch.pi))
    )
    log_survival = torch.special.log_ndtr(-standardized)
    censored_nll = -log_survival
    return torch.where(event > 0.5, event_nll, censored_nll).mean()


def internalized_edge_objective(
    output: dict[str, torch.Tensor],
    *,
    time: torch.Tensor,
    event: torch.Tensor,
    edge_target: torch.Tensor,
    edge_reconstruction_weight: float,
    aft_weight: float,
    cox_loss: torch.Tensor,
    edge_delta_weight: float = 1e-3,
    edge_saturation_weight: float = 1e-4,
) -> dict[str, torch.Tensor]:
    prediction = output["pair_edge_weights"]
    target = edge_target.view_as(prediction).to(prediction)
    edge_reconstruction = F.mse_loss(prediction, target)
    aft = lognormal_aft_nll(
        output["aft_location"],
        output["aft_log_scale"],
        time,
        event,
    )
    total = (
        cox_loss
        + float(edge_reconstruction_weight) * edge_reconstruction
        + float(aft_weight) * aft
        + float(edge_delta_weight)
        * output["edge_delta_regularization"]
        + float(edge_saturation_weight)
        * output["edge_saturation_regularization"]
    )
    return {
        "total": total,
        "cox": cox_loss,
        "edge_reconstruction": edge_reconstruction,
        "aft": aft,
    }
