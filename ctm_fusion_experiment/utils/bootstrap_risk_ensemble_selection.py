from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ctm_fusion_experiment.utils.risk_ensemble_selection import (
    apply_risk_ensemble,
    choose_cindex_risk_ensemble,
)


@dataclass(frozen=True)
class BootstrapRiskSelectionResult:
    candidate_name: str
    weights: list[float]
    c_index: float
    reference_c_index: float
    reference_index: int
    risk_means: list[float]
    risk_stds: list[float]
    candidates: list[dict[str, object]]


def choose_bootstrap_risk_ensemble(
    *,
    risk_matrix: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    model_names: Sequence[str],
    reference_index: int = 0,
    min_validation_delta: float = 0.001,
    min_resample_quantile_delta: float = -0.0005,
    resamples: int = 80,
    subsample_fraction: float = 0.7,
    lower_quantile: float = 0.1,
    stability_penalty: float = 0.5,
    softmax_temperature: float = 0.01,
    candidate_policy: str = "ensemble_only_or_reference",
    max_top_k: int = 5,
    seed: int = 42,
) -> BootstrapRiskSelectionResult:
    base = choose_cindex_risk_ensemble(
        risk_matrix=risk_matrix,
        times=times,
        events=events,
        model_names=model_names,
        reference_index=reference_index,
        min_c_index_delta=-1.0,
        softmax_temperature=softmax_temperature,
        candidate_policy=candidate_policy,
        max_top_k=max_top_k,
    )
    risk_rows = [
        apply_risk_ensemble(
            risk_matrix,
            candidate["weights"],
            base.risk_means,
            base.risk_stds,
        )
        .detach()
        .cpu()
        .numpy()
        for candidate in base.candidates
    ]
    reference_candidate_index = _reference_candidate_index(base.candidates)
    reference_risk = risk_rows[reference_candidate_index]
    time_np = times.detach().cpu().numpy()
    event_np = events.detach().cpu().numpy()
    rng = np.random.default_rng(int(seed))
    sample_count = len(time_np)
    subsample_size = max(2, min(sample_count, int(round(sample_count * float(subsample_fraction)))))
    resample_indices = [
        rng.choice(sample_count, size=subsample_size, replace=False)
        for _ in range(max(1, int(resamples)))
    ]

    enriched = []
    for candidate, risk in zip(base.candidates, risk_rows):
        deltas = np.asarray(
            [
                _fast_cindex(time_np[index], event_np[index], risk[index])
                - _fast_cindex(time_np[index], event_np[index], reference_risk[index])
                for index in resample_indices
            ],
            dtype=float,
        )
        full_delta = float(candidate["c_index_delta"])
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))
        lower_delta = float(np.quantile(deltas, float(lower_quantile)))
        robust_score = mean_delta - float(stability_penalty) * std_delta
        eligible = (
            candidate["candidate_name"] == "reference"
            or (
                full_delta >= float(min_validation_delta)
                and lower_delta >= float(min_resample_quantile_delta)
            )
        )
        enriched.append(
            {
                **candidate,
                "resample_mean_delta": mean_delta,
                "resample_std_delta": std_delta,
                "resample_lower_quantile_delta": lower_delta,
                "robust_score": float(robust_score),
                "eligible": bool(eligible),
            }
        )

    best = _choose_best(enriched)
    return BootstrapRiskSelectionResult(
        candidate_name=str(best["candidate_name"]),
        weights=[float(weight) for weight in best["weights"]],
        c_index=float(best["c_index"]),
        reference_c_index=float(base.reference_c_index),
        reference_index=int(base.reference_index),
        risk_means=base.risk_means,
        risk_stds=base.risk_stds,
        candidates=enriched,
    )


def _reference_candidate_index(candidates: Sequence[dict[str, object]]) -> int:
    for index, candidate in enumerate(candidates):
        if candidate["candidate_name"] == "reference":
            return index
    raise ValueError("No reference candidate found.")


def _choose_best(candidates: Sequence[dict[str, object]]) -> dict[str, object]:
    eligible = [candidate for candidate in candidates if bool(candidate["eligible"])]
    if not eligible:
        eligible = [candidate for candidate in candidates if candidate["candidate_name"] == "reference"]
    if not eligible:
        raise ValueError("No eligible candidate and no reference candidate found.")

    def sort_key(candidate: dict[str, object]) -> tuple[float, float, float, float]:
        weights = [float(weight) for weight in candidate["weights"]]
        nonzero = sum(1 for weight in weights if abs(weight) > 1e-9)
        max_weight = max(abs(weight) for weight in weights)
        return (
            float(candidate["robust_score"]),
            float(candidate["c_index_delta"]),
            -float(nonzero),
            float(max_weight),
        )

    return max(eligible, key=sort_key)


def _fast_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    permissible = (time[:, None] < time[None, :]) & (event[:, None] > 0.0)
    count = int(np.sum(permissible))
    if count == 0:
        return 0.0
    risk_left = risk[:, None]
    risk_right = risk[None, :]
    credit = (risk_left > risk_right).astype(float) + 0.5 * (risk_left == risk_right)
    return float(np.sum(credit[permissible]) / count)
