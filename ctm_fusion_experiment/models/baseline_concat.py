from __future__ import annotations

import torch
import torch.nn as nn

from ctm_fusion_experiment.models.cox_head import CoxHead


class BaselineConcatModel(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        input_dim = graph_dim + clinical_dim + metabolite_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = CoxHead(hidden_dim)

    def forward(
        self,
        graph_features: torch.Tensor,
        clinical_features: torch.Tensor,
        metabolite_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        concatenated = torch.cat([graph_features, clinical_features, metabolite_features], dim=1)
        latent = self.fusion(concatenated)
        return {"risk": self.risk_head(latent), "latent": latent}
