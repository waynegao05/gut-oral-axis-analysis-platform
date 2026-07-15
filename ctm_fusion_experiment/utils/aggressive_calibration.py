from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ctm_fusion_experiment.utils.calibration import apply_residual_alpha
from ctm_fusion_experiment.utils.utility_metrics import risk_utility_metrics


@dataclass(frozen=True)
class AggressiveResidualAlphaResult:
    alpha: float
    objective: float
    c_index: float
    utility: dict[str, float]
    baseline_utility: dict[str, float]
    candidates: list[dict[str, float | bool]]


def choose_aggressive_residual_alpha(
    *,
    baseline_risk: torch.Tensor,
    delta: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    alpha_grid: Sequence[float],
    top_fraction: float = 0.1,
    c_index_weight: float = 1.0,
    top_event_lift_weight: float = 0.003,
    high_low_gap_weight: float = 0.002,
    risk_spread_weight: float = 0.0005,
    min_c_index_delta: float = -0.001,
    min_objective_delta: float = 0.0,
) -> AggressiveResidualAlphaResult:
    """Select alpha with a validation objective beyond pure c-index.

    The objective is expressed as improvement over the fold's baseline:

    c-index delta
    + top event lift delta
    + high/low event-rate gap delta
    + relative risk-spread delta

    A small negative c-index floor is allowed so v3 can choose useful risk
    stratification improvements instead of being identical to v2.
    """

    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one candidate.")

    time_np = times.detach().cpu().numpy()
    event_np = events.detach().cpu().numpy()
    baseline_np = baseline_risk.detach().cpu().numpy()
    baseline_utility = risk_utility_metrics(time_np, event_np, baseline_np, top_fraction=top_fraction)
    baseline_std = max(float(baseline_utility["risk_std"]), 1e-8)

    candidates: list[dict[str, float | bool]] = []
    best: dict[str, float | bool] | None = None
    best_utility: dict[str, float] | None = None

    for alpha in alpha_grid:
        risk = apply_residual_alpha(baseline_risk, delta, float(alpha)).detach().cpu().numpy()
        utility = risk_utility_metrics(time_np, event_np, risk, top_fraction=top_fraction)
        c_index_delta = float(utility["c_index"] - baseline_utility["c_index"])
        top_lift_delta = float(utility["top_event_lift"] - baseline_utility["top_event_lift"])
        gap_delta = float(utility["high_low_event_gap"] - baseline_utility["high_low_event_gap"])
        raw_spread_delta = float((utility["risk_std"] - baseline_utility["risk_std"]) / baseline_std)
        spread_delta = float(np.clip(raw_spread_delta, -2.0, 2.0))
        objective = (
            float(c_index_weight) * c_index_delta
            + float(top_event_lift_weight) * top_lift_delta
            + float(high_low_gap_weight) * gap_delta
            + float(risk_spread_weight) * spread_delta
        )
        eligible = c_index_delta >= float(min_c_index_delta)
        candidate = {
            "alpha": float(alpha),
            "objective": float(objective),
            "c_index": float(utility["c_index"]),
            "c_index_delta": c_index_delta,
            "top_event_lift": float(utility["top_event_lift"]),
            "top_event_lift_delta": top_lift_delta,
            "high_low_event_gap": float(utility["high_low_event_gap"]),
            "high_low_event_gap_delta": gap_delta,
            "risk_std": float(utility["risk_std"]),
            "risk_spread_delta": spread_delta,
            "eligible": bool(eligible),
        }
        candidates.append(candidate)
        if not eligible:
            continue
        if _is_better_candidate(candidate, best):
            best = candidate
            best_utility = utility

    if best is None:
        best_index = int(np.argmax([float(candidate["c_index"]) for candidate in candidates]))
        best = candidates[best_index]
        best_alpha = float(best["alpha"])
        best_risk = apply_residual_alpha(baseline_risk, delta, best_alpha).detach().cpu().numpy()
        best_utility = risk_utility_metrics(time_np, event_np, best_risk, top_fraction=top_fraction)

    if best is not None and float(best["objective"]) < float(min_objective_delta):
        baseline_candidate = _baseline_candidate(candidates)
        if baseline_candidate is not None:
            best = baseline_candidate
            best_utility = baseline_utility

    return AggressiveResidualAlphaResult(
        alpha=float(best["alpha"]),
        objective=float(best["objective"]),
        c_index=float(best["c_index"]),
        utility=best_utility or baseline_utility,
        baseline_utility=baseline_utility,
        candidates=candidates,
    )


def _is_better_candidate(
    candidate: dict[str, float | bool],
    current: dict[str, float | bool] | None,
) -> bool:
    if current is None:
        return True
    candidate_objective = float(candidate["objective"])
    current_objective = float(current["objective"])
    if candidate_objective > current_objective + 1e-12:
        return True
    if not np.isclose(candidate_objective, current_objective):
        return False
    candidate_c_index = float(candidate["c_index"])
    current_c_index = float(current["c_index"])
    if candidate_c_index > current_c_index + 1e-12:
        return True
    if not np.isclose(candidate_c_index, current_c_index):
        return False
    return abs(float(candidate["alpha"])) < abs(float(current["alpha"]))


def _baseline_candidate(candidates: list[dict[str, float | bool]]) -> dict[str, float | bool] | None:
    for candidate in candidates:
        if np.isclose(float(candidate["alpha"]), 0.0):
            return candidate
    return None
