from __future__ import annotations

import torch
import torch.nn.functional as F

from research.losses import cox_ph_loss


def fit_discrete_time_cutpoints(
    train_time: torch.Tensor,
    train_event: torch.Tensor,
    *,
    num_bins: int,
) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be at least two.")
    time = train_time.detach().float().view(-1).cpu()
    event = train_event.detach().float().view(-1).cpu()
    if time.numel() != event.numel() or time.numel() < num_bins:
        raise ValueError("Training labels are too small for the requested bins.")
    if not torch.isfinite(time).all() or torch.any(time <= 0):
        raise ValueError("Survival times must be finite and positive.")
    observed = time[event > 0.5]
    if observed.numel() < num_bins:
        observed = time
    quantiles = torch.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    cutpoints = torch.quantile(observed, quantiles)
    if cutpoints.numel() != num_bins - 1:
        raise RuntimeError("Unexpected discrete-time cutpoint count.")
    epsilon = torch.finfo(cutpoints.dtype).eps * 100
    for index in range(1, cutpoints.numel()):
        cutpoints[index] = torch.maximum(
            cutpoints[index], cutpoints[index - 1] + epsilon
        )
    return cutpoints


def discrete_time_nll(
    time_logits: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    cutpoints: torch.Tensor,
) -> torch.Tensor:
    if time_logits.ndim != 2:
        raise ValueError("time_logits must have shape [N, num_bins].")
    time = time.float().view(-1)
    event = event.float().view(-1)
    if (
        time.numel() != time_logits.size(0)
        or event.numel() != time_logits.size(0)
    ):
        raise ValueError("Discrete-time labels do not align with logits.")
    if cutpoints.numel() != time_logits.size(1) - 1:
        raise ValueError("Cutpoint count must equal num_bins - 1.")
    interval = torch.bucketize(
        time, cutpoints.to(device=time.device, dtype=time.dtype)
    )
    log_hazard = F.logsigmoid(time_logits)
    log_survival = F.logsigmoid(-time_logits)
    bins = torch.arange(
        time_logits.size(1), device=time_logits.device
    ).view(1, -1)
    before = bins < interval.view(-1, 1)
    through = bins <= interval.view(-1, 1)
    event_log_likelihood = (
        (log_survival * before).sum(dim=1)
        + log_hazard.gather(1, interval.view(-1, 1)).squeeze(1)
    )
    censored_log_likelihood = (log_survival * through).sum(dim=1)
    log_likelihood = torch.where(
        event > 0.5, event_log_likelihood, censored_log_likelihood
    )
    return -log_likelihood.mean()


def dual_survival_objective(
    output: dict[str, torch.Tensor],
    *,
    time: torch.Tensor,
    event: torch.Tensor,
    cutpoints: torch.Tensor,
    discrete_weight: float,
    edge_delta_weight: float = 1e-3,
    edge_saturation_weight: float = 1e-4,
) -> dict[str, torch.Tensor]:
    cox = cox_ph_loss(
        output["risk"],
        time,
        event,
        ties_method="breslow",
    )
    discrete = discrete_time_nll(
        output["time_logits"],
        time,
        event,
        cutpoints,
    )
    total = (
        cox
        + float(discrete_weight) * discrete
        + float(edge_delta_weight)
        * output["edge_delta_regularization"]
        + float(edge_saturation_weight)
        * output["edge_saturation_regularization"]
    )
    return {
        "total": total,
        "cox": cox,
        "discrete": discrete,
        "edge_delta": output["edge_delta_regularization"],
        "edge_saturation": output["edge_saturation_regularization"],
    }
