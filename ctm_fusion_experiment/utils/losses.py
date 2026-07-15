from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CTMLossResult:
    loss: torch.Tensor
    losses_per_tick: torch.Tensor
    best_loss_tick: int
    stable_tick: int


@dataclass(frozen=True)
class ResidualCTMLossResult:
    loss: torch.Tensor
    losses_per_tick: torch.Tensor
    best_loss_tick: int
    components: dict[str, torch.Tensor]


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    ordered_risk = risk_scores[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_risk, dim=0)
    observed = (ordered_risk - log_risk) * ordered_events
    return -observed.sum() / torch.clamp(ordered_events.sum(), min=1.0)


def pairwise_ranking_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    if risk_scores.ndim != 1:
        raise ValueError("risk_scores must have shape (batch,).")
    if not (risk_scores.shape == times.shape == events.shape):
        raise ValueError("risk_scores, times, and events must have the same shape.")

    earlier_event = events.unsqueeze(1) > 0.0
    comparable = earlier_event & (times.unsqueeze(1) < times.unsqueeze(0))
    if not torch.any(comparable):
        return risk_scores.sum() * 0.0

    risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
    return F.softplus(float(margin) - risk_diff[comparable]).mean()


def baseline_discordant_pairwise_loss(
    risk_scores: torch.Tensor,
    baseline_risk: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    if not (risk_scores.shape == baseline_risk.shape == times.shape == events.shape):
        raise ValueError("risk_scores, baseline_risk, times, and events must have the same shape.")

    earlier_event = events.unsqueeze(1) > 0.0
    comparable = earlier_event & (times.unsqueeze(1) < times.unsqueeze(0))
    if not torch.any(comparable):
        return risk_scores.sum() * 0.0

    baseline_diff = baseline_risk.detach().unsqueeze(1) - baseline_risk.detach().unsqueeze(0)
    hard_pairs = comparable & (baseline_diff <= 0.0)
    if not torch.any(hard_pairs):
        return risk_scores.sum() * 0.0

    risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
    return F.softplus(float(margin) - risk_diff[hard_pairs]).mean()


def select_stable_ticks(risk_scores_per_tick: torch.Tensor) -> torch.Tensor:
    if risk_scores_per_tick.ndim != 2:
        raise ValueError("risk_scores_per_tick must have shape (batch, ticks).")
    if risk_scores_per_tick.size(1) == 1:
        return torch.zeros(risk_scores_per_tick.size(0), dtype=torch.long, device=risk_scores_per_tick.device)
    changes = torch.abs(risk_scores_per_tick[:, 1:] - risk_scores_per_tick[:, :-1])
    return torch.argmin(changes, dim=1) + 1


def select_batch_stable_tick(risk_scores_per_tick: torch.Tensor) -> int:
    if risk_scores_per_tick.size(1) == 1:
        return 0
    mean_changes = torch.mean(
        torch.abs(risk_scores_per_tick[:, 1:] - risk_scores_per_tick[:, :-1]),
        dim=0,
    )
    return int(torch.argmin(mean_changes).item()) + 1


def gather_stable_risk(risk_scores_per_tick: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stable_ticks = select_stable_ticks(risk_scores_per_tick)
    stable_risk = risk_scores_per_tick.gather(1, stable_ticks.unsqueeze(1)).squeeze(1)
    return stable_risk, stable_ticks


def ctm_cox_loss(
    risk_scores_per_tick: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> CTMLossResult:
    losses_per_tick = torch.stack(
        [
            cox_partial_likelihood_loss(risk_scores_per_tick[:, tick], times, events)
            for tick in range(risk_scores_per_tick.size(1))
        ]
    )
    best_loss_tick = int(torch.argmin(losses_per_tick).item())
    stable_tick = select_batch_stable_tick(risk_scores_per_tick)
    loss = 0.5 * (losses_per_tick[best_loss_tick] + losses_per_tick[stable_tick])
    return CTMLossResult(
        loss=loss,
        losses_per_tick=losses_per_tick,
        best_loss_tick=best_loss_tick,
        stable_tick=stable_tick,
    )


def _zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = values - values.mean()
    return centered / torch.clamp(centered.std(unbiased=False), min=eps)


def residual_ctm_cox_loss(
    risk_scores_per_tick: torch.Tensor,
    baseline_risk: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    mean_weight: float,
    final_weight: float,
    best_weight: float,
    distillation_weight: float,
    separation_weight: float,
    min_risk_std: float,
) -> ResidualCTMLossResult:
    losses_per_tick = torch.stack(
        [
            cox_partial_likelihood_loss(risk_scores_per_tick[:, tick], times, events)
            for tick in range(risk_scores_per_tick.size(1))
        ]
    )
    best_loss_tick = int(torch.argmin(losses_per_tick).item())
    final_risk = risk_scores_per_tick[:, -1]
    mean_tick_cox = losses_per_tick.mean()
    final_tick_cox = losses_per_tick[-1]
    best_tick_cox = losses_per_tick[best_loss_tick]
    distillation = torch.mean((_zscore(final_risk) - _zscore(baseline_risk.detach())) ** 2)
    separation = torch.relu(
        torch.tensor(float(min_risk_std), device=final_risk.device, dtype=final_risk.dtype)
        - final_risk.std(unbiased=False)
    )
    loss = (
        float(mean_weight) * mean_tick_cox
        + float(final_weight) * final_tick_cox
        + float(best_weight) * best_tick_cox
        + float(distillation_weight) * distillation
        + float(separation_weight) * separation
    )
    return ResidualCTMLossResult(
        loss=loss,
        losses_per_tick=losses_per_tick,
        best_loss_tick=best_loss_tick,
        components={
            "mean_tick_cox": mean_tick_cox.detach(),
            "final_tick_cox": final_tick_cox.detach(),
            "best_tick_cox": best_tick_cox.detach(),
            "distillation": distillation.detach(),
            "separation": separation.detach(),
        },
    )
