from __future__ import annotations

import torch
import torch.nn.functional as F


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    log_likelihood = risk - log_cumsum
    observed = log_likelihood * event
    denom = torch.clamp(event.sum(), min=1.0)
    return -observed.sum() / denom


def pairwise_ranking_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    losses = []
    num_samples = risk.size(0)

    for i in range(num_samples):
        if event[i] <= 0:
            continue
        later_mask = time > time[i]
        if not torch.any(later_mask):
            continue
        pair_losses = torch.relu(margin - (risk[i] - risk[later_mask]))
        losses.append(pair_losses.mean())

    if not losses:
        return torch.zeros((), device=risk.device, dtype=risk.dtype)

    return torch.stack(losses).mean()


def combined_survival_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    cox_loss = cox_ph_loss(risk, time, event)
    ranking_loss = pairwise_ranking_loss(
        risk=risk,
        time=time,
        event=event,
        margin=ranking_margin,
    )
    total_loss = cox_loss + ranking_weight * ranking_loss
    return {
        "total": total_loss,
        "cox": cox_loss,
        "ranking": ranking_loss,
    }


def build_time_bin_edges(
    time: torch.Tensor,
    num_bins: int,
    min_bin_width: float = 1e-3,
) -> torch.Tensor:
    if num_bins <= 1:
        return torch.tensor([float(time.max().item())], dtype=time.dtype)

    quantiles = torch.linspace(
        1.0 / float(num_bins),
        1.0,
        steps=num_bins,
        dtype=time.dtype,
        device=time.device,
    )
    edges = torch.quantile(time, quantiles)
    for idx in range(1, edges.numel()):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + min_bin_width
    return edges


def discrete_time_nll_loss(
    time_logits: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    time_bin_edges: torch.Tensor,
) -> torch.Tensor:
    num_bins = time_logits.size(1)
    bin_index = torch.bucketize(time, time_bin_edges)
    bin_index = torch.clamp(bin_index, max=num_bins - 1)

    bin_range = torch.arange(num_bins, device=time_logits.device).unsqueeze(0)
    expanded_index = bin_index.unsqueeze(1)
    mask = bin_range <= expanded_index
    targets = (bin_range == expanded_index) & (event.unsqueeze(1) > 0)

    losses = F.binary_cross_entropy_with_logits(time_logits, targets.float(), reduction="none")
    denom = torch.clamp(mask.sum(), min=1).to(time_logits.dtype)
    return (losses * mask.float()).sum() / denom
