from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ctm_fusion_experiment.utils.metrics import concordance_index


@dataclass(frozen=True)
class ResidualAlphaResult:
    alpha: float
    c_index: float
    candidates: list[dict[str, float]]


def apply_residual_alpha(
    baseline_risk: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    return baseline_risk + float(alpha) * delta


def choose_residual_alpha(
    baseline_risk: torch.Tensor,
    delta: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    alpha_grid: Sequence[float],
) -> ResidualAlphaResult:
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one candidate.")

    candidates = []
    best_alpha = float(alpha_grid[0])
    best_c_index = float("-inf")
    time_np = times.detach().cpu().numpy()
    event_np = events.detach().cpu().numpy()

    for alpha in alpha_grid:
        risk = apply_residual_alpha(baseline_risk, delta, float(alpha))
        c_index = concordance_index(time_np, event_np, risk.detach().cpu().numpy())
        candidates.append({"alpha": float(alpha), "c_index": float(c_index)})
        if c_index > best_c_index:
            best_c_index = float(c_index)
            best_alpha = float(alpha)

    return ResidualAlphaResult(alpha=best_alpha, c_index=best_c_index, candidates=candidates)
