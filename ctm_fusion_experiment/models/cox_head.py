from __future__ import annotations

import torch
import torch.nn as nn


class CoxHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features).squeeze(-1)
