from __future__ import annotations

import torch


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    log_likelihood = risk - log_cumsum
    observed = log_likelihood * event
    denom = torch.clamp(event.sum(), min=1.0)
    return -observed.sum() / denom
