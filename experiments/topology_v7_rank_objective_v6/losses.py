from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def comparable_pair_logistic_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Smooth Harrell-pair loss for observed earlier events."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)
    comparable = (
        (event[:, None] > 0.5)
        & (time[:, None] < time[None, :])
    )
    if not torch.any(comparable):
        return risk.sum() * 0.0
    difference = (
        risk[:, None] - risk[None, :]
    ) / float(temperature)
    return F.softplus(-difference[comparable]).mean()


def horizon_pair_logistic_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    horizons: Sequence[float],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Smooth cumulative/dynamic AUC loss with ambiguous censoring excluded."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)
    losses: list[torch.Tensor] = []
    for horizon in horizons:
        cases = (event > 0.5) & (time <= float(horizon))
        controls = time > float(horizon)
        if not torch.any(cases) or not torch.any(controls):
            continue
        difference = (
            risk[cases, None] - risk[None, controls]
        ) / float(temperature)
        losses.append(F.softplus(-difference).mean())
    if not losses:
        return risk.sum() * 0.0
    return torch.stack(losses).mean()
