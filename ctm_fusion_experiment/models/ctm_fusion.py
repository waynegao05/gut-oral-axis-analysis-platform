from __future__ import annotations

import torch
import torch.nn as nn

from ctm_fusion_experiment.models.cox_head import CoxHead
from ctm_fusion_experiment.models.ctm import CTM


class _ModalityProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CTMFusionModel(nn.Module):
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
    ) -> None:
        super().__init__()
        self.graph_projection = _ModalityProjection(graph_dim, d_input)
        self.clinical_projection = _ModalityProjection(clinical_dim, d_input)
        self.metabolite_projection = _ModalityProjection(metabolite_dim, d_input)
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
        self.risk_head = CoxHead(n_synch_out)

    def forward(
        self,
        graph_features: torch.Tensor,
        clinical_features: torch.Tensor,
        metabolite_features: torch.Tensor,
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
        ctm_output = self.ctm(tokens, track_attention=track_attention)
        risk_per_tick = self.risk_head(ctm_output.representations)
        return {
            "risk_per_tick": risk_per_tick,
            "representations": ctm_output.representations,
            "attention_weights": ctm_output.attention_weights,
        }
