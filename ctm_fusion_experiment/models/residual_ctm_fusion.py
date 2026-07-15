from __future__ import annotations

import torch
import torch.nn as nn

from ctm_fusion_experiment.models.cox_head import CoxHead
from ctm_fusion_experiment.models.ctm import CTM


class _GatedModalityProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.network(features)


class ResidualCTMFusionModel(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        d_input: int,
        d_model: int,
        iterations: int,
        memory_length: int,
        nlm_hidden_dim: int,
        n_heads: int,
        n_synch_action: int,
        n_synch_out: int,
        n_self_pairs: int,
        synapse_depth: int,
        dropout: float,
        max_residual_gate: float,
        initial_gate_logit: float,
    ) -> None:
        super().__init__()
        self.graph_projection = _GatedModalityProjection(graph_dim, d_input)
        self.clinical_projection = _GatedModalityProjection(clinical_dim, d_input)
        self.metabolite_projection = _GatedModalityProjection(metabolite_dim, d_input)
        self.modality_embedding = nn.Parameter(torch.zeros(3, d_input))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.ctm = CTM(
            d_model=d_model,
            d_input=d_input,
            iterations=iterations,
            memory_length=memory_length,
            nlm_hidden_dim=nlm_hidden_dim,
            n_heads=n_heads,
            n_synch_action=n_synch_action,
            n_synch_out=n_synch_out,
            n_self_pairs=n_self_pairs,
            synapse_depth=synapse_depth,
            dropout=dropout,
        )
        self.delta_head = CoxHead(n_synch_out)
        nn.init.zeros_(self.delta_head.projection.weight)
        nn.init.zeros_(self.delta_head.projection.bias)
        self.residual_gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))
        self.max_residual_gate = float(max_residual_gate)

    def forward(
        self,
        graph_features: torch.Tensor,
        clinical_features: torch.Tensor,
        metabolite_features: torch.Tensor,
        baseline_risk: torch.Tensor,
        track_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokens = torch.stack(
            [
                self.graph_projection(graph_features),
                self.clinical_projection(clinical_features),
                self.metabolite_projection(metabolite_features),
            ],
            dim=1,
        )
        tokens = tokens + self.modality_embedding.unsqueeze(0)
        ctm_output = self.ctm(tokens, track_attention=track_attention)
        delta_per_tick = self.delta_head(ctm_output.representations)
        residual_gate = self.max_residual_gate * torch.sigmoid(self.residual_gate_logit)
        risk_per_tick = baseline_risk.detach().unsqueeze(1) + residual_gate * delta_per_tick
        return {
            "risk_per_tick": risk_per_tick,
            "delta_per_tick": delta_per_tick,
            "residual_gate": residual_gate,
            "representations": ctm_output.representations,
            "attention_weights": ctm_output.attention_weights,
        }
