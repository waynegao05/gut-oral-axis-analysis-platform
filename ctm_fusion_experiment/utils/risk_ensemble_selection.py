from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ctm_fusion_experiment.utils.metrics import concordance_index


@dataclass(frozen=True)
class RiskEnsembleSelectionResult:
    candidate_name: str
    weights: list[float]
    c_index: float
    reference_c_index: float
    reference_index: int
    risk_means: list[float]
    risk_stds: list[float]
    candidates: list[dict[str, object]]


def choose_cindex_risk_ensemble(
    *,
    risk_matrix: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    model_names: Sequence[str],
    reference_index: int = 0,
    min_c_index_delta: float = 0.0,
    softmax_temperature: float = 0.01,
    candidate_policy: str = "all",
    max_top_k: int = 3,
) -> RiskEnsembleSelectionResult:
    if risk_matrix.ndim != 2:
        raise ValueError("risk_matrix must have shape (num_models, num_samples).")
    if risk_matrix.size(0) != len(model_names):
        raise ValueError("model_names must match risk_matrix rows.")
    if not 0 <= reference_index < risk_matrix.size(0):
        raise ValueError("reference_index is out of range.")

    time_np = times.detach().cpu().numpy()
    event_np = events.detach().cpu().numpy()
    means = risk_matrix.mean(dim=1)
    stds = torch.clamp(risk_matrix.std(dim=1, unbiased=False), min=1e-6)
    standardized = _standardize_matrix(risk_matrix, means, stds)
    single_c_indexes = [
        concordance_index(time_np, event_np, standardized[index].detach().cpu().numpy())
        for index in range(risk_matrix.size(0))
    ]
    reference_c_index = float(single_c_indexes[reference_index])
    candidates: list[dict[str, object]] = []

    reference_weights = [0.0 for _ in range(risk_matrix.size(0))]
    reference_weights[reference_index] = 1.0
    _append_if_allowed(
        candidates,
        candidate_policy=candidate_policy,
        name="reference",
        weights=reference_weights,
        standardized_risks=standardized,
        time_np=time_np,
        event_np=event_np,
        reference_c_index=reference_c_index,
        min_c_index_delta=min_c_index_delta,
    )

    for index, model_name in enumerate(model_names):
        weights = [0.0 for _ in range(risk_matrix.size(0))]
        weights[index] = 1.0
        _append_if_allowed(
            candidates,
            candidate_policy=candidate_policy,
            name=f"single:{model_name}",
            weights=weights,
            standardized_risks=standardized,
            time_np=time_np,
            event_np=event_np,
            reference_c_index=reference_c_index,
            min_c_index_delta=min_c_index_delta,
        )

    top_k_limit = max(2, min(int(max_top_k), int(risk_matrix.size(0))))
    for count in range(2, top_k_limit + 1):
        top_indices = list(np.argsort(single_c_indexes)[-count:])
        weights = [0.0 for _ in range(risk_matrix.size(0))]
        for index in top_indices:
            weights[int(index)] = 1.0 / count
        _append_if_allowed(
            candidates,
            candidate_policy=candidate_policy,
            name=f"top{count}_mean",
            weights=weights,
            standardized_risks=standardized,
            time_np=time_np,
            event_np=event_np,
            reference_c_index=reference_c_index,
            min_c_index_delta=min_c_index_delta,
        )

    mean_weights = [1.0 / risk_matrix.size(0) for _ in range(risk_matrix.size(0))]
    _append_if_allowed(
        candidates,
        candidate_policy=candidate_policy,
        name="mean_all",
        weights=mean_weights,
        standardized_risks=standardized,
        time_np=time_np,
        event_np=event_np,
        reference_c_index=reference_c_index,
        min_c_index_delta=min_c_index_delta,
    )

    softmax_base = np.asarray(single_c_indexes, dtype=float)
    temperature = max(float(softmax_temperature), 1e-6)
    softmax_weights = np.exp((softmax_base - np.max(softmax_base)) / temperature)
    softmax_weights = softmax_weights / np.sum(softmax_weights)
    _append_if_allowed(
        candidates,
        candidate_policy=candidate_policy,
        name="softmax_all",
        weights=[float(weight) for weight in softmax_weights],
        standardized_risks=standardized,
        time_np=time_np,
        event_np=event_np,
        reference_c_index=reference_c_index,
        min_c_index_delta=min_c_index_delta,
    )

    best = _choose_best_candidate(candidates)
    if best is None:
        best = candidates[0]

    return RiskEnsembleSelectionResult(
        candidate_name=str(best["candidate_name"]),
        weights=[float(weight) for weight in best["weights"]],
        c_index=float(best["c_index"]),
        reference_c_index=reference_c_index,
        reference_index=reference_index,
        risk_means=[float(value) for value in means.detach().cpu().tolist()],
        risk_stds=[float(value) for value in stds.detach().cpu().tolist()],
        candidates=candidates,
    )


def apply_risk_ensemble(
    risk_matrix: torch.Tensor,
    weights: Sequence[float],
    risk_means: Sequence[float],
    risk_stds: Sequence[float],
) -> torch.Tensor:
    if risk_matrix.ndim != 2:
        raise ValueError("risk_matrix must have shape (num_models, num_samples).")
    if not (risk_matrix.size(0) == len(weights) == len(risk_means) == len(risk_stds)):
        raise ValueError("weights, risk_means, and risk_stds must match risk_matrix rows.")
    means = torch.tensor(risk_means, dtype=risk_matrix.dtype, device=risk_matrix.device)
    stds = torch.clamp(torch.tensor(risk_stds, dtype=risk_matrix.dtype, device=risk_matrix.device), min=1e-6)
    standardized = _standardize_matrix(risk_matrix, means, stds)
    weight_tensor = torch.tensor(weights, dtype=risk_matrix.dtype, device=risk_matrix.device)
    return torch.sum(standardized * weight_tensor.unsqueeze(1), dim=0)


def _standardize_matrix(
    risk_matrix: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    return (risk_matrix - means.unsqueeze(1)) / stds.unsqueeze(1)


def _append_if_allowed(
    candidates: list[dict[str, object]],
    *,
    candidate_policy: str,
    name: str,
    weights: Sequence[float],
    standardized_risks: torch.Tensor,
    time_np: np.ndarray,
    event_np: np.ndarray,
    reference_c_index: float,
    min_c_index_delta: float,
) -> None:
    if not _allowed_by_policy(name, candidate_policy):
        return
    _append_candidate(
        candidates,
        name=name,
        weights=weights,
        standardized_risks=standardized_risks,
        time_np=time_np,
        event_np=event_np,
        reference_c_index=reference_c_index,
        min_c_index_delta=min_c_index_delta,
    )


def _allowed_by_policy(name: str, candidate_policy: str) -> bool:
    if name == "reference":
        return True
    if candidate_policy == "all":
        return True
    if candidate_policy == "ensemble_only_or_reference":
        return not name.startswith("single:")
    if candidate_policy == "single_only_or_reference":
        return name.startswith("single:")
    raise ValueError(f"Unknown candidate_policy: {candidate_policy}")


def _append_candidate(
    candidates: list[dict[str, object]],
    *,
    name: str,
    weights: Sequence[float],
    standardized_risks: torch.Tensor,
    time_np: np.ndarray,
    event_np: np.ndarray,
    reference_c_index: float,
    min_c_index_delta: float,
) -> None:
    weight_tensor = torch.tensor(weights, dtype=standardized_risks.dtype, device=standardized_risks.device)
    risk = torch.sum(standardized_risks * weight_tensor.unsqueeze(1), dim=0)
    c_index = concordance_index(time_np, event_np, risk.detach().cpu().numpy())
    delta = float(c_index - reference_c_index)
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
        nonzero = sum(1 for weight in weights if not np.isclose(weight, 0.0))
        max_weight = max(abs(weight) for weight in weights)
        return (
            float(candidate["c_index"]),
            -float(nonzero),
            float(max_weight),
        )

    return max(eligible, key=sort_key)
