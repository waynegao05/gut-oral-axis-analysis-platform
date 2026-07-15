from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ctm_fusion_experiment.utils.metrics import concordance_index


@dataclass(frozen=True)
class CIndexEnsembleSelectionResult:
    candidate_name: str
    weights: list[float]
    c_index: float
    baseline_c_index: float
    candidates: list[dict[str, object]]


def apply_ensemble_weights(
    baseline_risk: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    if len(deltas) != len(weights):
        raise ValueError("deltas and weights must have the same length.")
    risk = baseline_risk.clone()
    for delta, weight in zip(deltas, weights):
        risk = risk + float(weight) * delta
    return risk


def choose_cindex_ensemble(
    *,
    baseline_risk: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    times: torch.Tensor,
    events: torch.Tensor,
    alpha_grid: Sequence[float],
    min_c_index_delta: float = 0.0,
    softmax_temperature: float = 0.01,
) -> CIndexEnsembleSelectionResult:
    if not deltas:
        raise ValueError("At least one delta tensor is required.")
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one candidate.")

    time_np = times.detach().cpu().numpy()
    event_np = events.detach().cpu().numpy()
    baseline_np = baseline_risk.detach().cpu().numpy()
    baseline_c_index = concordance_index(time_np, event_np, baseline_np)
    seed_c_indexes = [
        concordance_index(time_np, event_np, (baseline_risk + delta).detach().cpu().numpy())
        for delta in deltas
    ]
    candidates: list[dict[str, object]] = []
    zero_weights = [0.0 for _ in deltas]
    candidates.append(
        {
            "candidate_name": "baseline",
            "weights": zero_weights,
            "c_index": float(baseline_c_index),
            "c_index_delta": 0.0,
            "eligible": True,
        }
    )

    for seed_index in range(len(deltas)):
        for alpha in alpha_grid:
            weights = [0.0 for _ in deltas]
            weights[seed_index] = float(alpha)
            _append_candidate(
                candidates,
                name=f"seed_{seed_index}_alpha_{float(alpha):g}",
                weights=weights,
                baseline_risk=baseline_risk,
                deltas=deltas,
                time_np=time_np,
                event_np=event_np,
                baseline_c_index=baseline_c_index,
                min_c_index_delta=min_c_index_delta,
            )

    for alpha in alpha_grid:
        mean_weights = [float(alpha) / len(deltas) for _ in deltas]
        _append_candidate(
            candidates,
            name=f"mean_alpha_{float(alpha):g}",
            weights=mean_weights,
            baseline_risk=baseline_risk,
            deltas=deltas,
            time_np=time_np,
            event_np=event_np,
            baseline_c_index=baseline_c_index,
            min_c_index_delta=min_c_index_delta,
        )

    top_count = min(2, len(deltas))
    if top_count > 1:
        top_indices = list(np.argsort(seed_c_indexes)[-top_count:])
        for alpha in alpha_grid:
            top_weights = [0.0 for _ in deltas]
            for index in top_indices:
                top_weights[int(index)] = float(alpha) / top_count
            _append_candidate(
                candidates,
                name=f"top{top_count}_alpha_{float(alpha):g}",
                weights=top_weights,
                baseline_risk=baseline_risk,
                deltas=deltas,
                time_np=time_np,
                event_np=event_np,
                baseline_c_index=baseline_c_index,
                min_c_index_delta=min_c_index_delta,
            )

    softmax_base = np.asarray(seed_c_indexes, dtype=float)
    temperature = max(float(softmax_temperature), 1e-6)
    softmax_weights = np.exp((softmax_base - np.max(softmax_base)) / temperature)
    softmax_weights = softmax_weights / np.sum(softmax_weights)
    for alpha in alpha_grid:
        weights = [float(alpha) * float(weight) for weight in softmax_weights]
        _append_candidate(
            candidates,
            name=f"softmax_alpha_{float(alpha):g}",
            weights=weights,
            baseline_risk=baseline_risk,
            deltas=deltas,
            time_np=time_np,
            event_np=event_np,
            baseline_c_index=baseline_c_index,
            min_c_index_delta=min_c_index_delta,
        )

    best = _choose_best_candidate(candidates)
    if best is None:
        best = candidates[0]
    return CIndexEnsembleSelectionResult(
        candidate_name=str(best["candidate_name"]),
        weights=[float(weight) for weight in best["weights"]],
        c_index=float(best["c_index"]),
        baseline_c_index=float(baseline_c_index),
        candidates=candidates,
    )


def _append_candidate(
    candidates: list[dict[str, object]],
    *,
    name: str,
    weights: Sequence[float],
    baseline_risk: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    time_np: np.ndarray,
    event_np: np.ndarray,
    baseline_c_index: float,
    min_c_index_delta: float,
) -> None:
    risk = apply_ensemble_weights(baseline_risk, deltas, weights)
    c_index = concordance_index(time_np, event_np, risk.detach().cpu().numpy())
    delta = float(c_index - baseline_c_index)
    candidates.append(
        {
            "candidate_name": name,
            "weights": [float(weight) for weight in weights],
            "c_index": float(c_index),
            "c_index_delta": delta,
            "eligible": bool(delta >= float(min_c_index_delta)),
        }
    )


def _choose_best_candidate(candidates: Sequence[dict[str, object]]) -> dict[str, object] | None:
    eligible = [candidate for candidate in candidates if bool(candidate["eligible"])]
    if not eligible:
        return None

    def sort_key(candidate: dict[str, object]) -> tuple[float, float, float]:
        weights = [float(weight) for weight in candidate["weights"]]
        nonzero_count = sum(1 for weight in weights if not np.isclose(weight, 0.0))
        magnitude = sum(abs(weight) for weight in weights)
        return (
            float(candidate["c_index"]),
            -float(nonzero_count),
            -float(magnitude),
        )

    return max(eligible, key=sort_key)
