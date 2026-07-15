from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from research.ensemble_stack_v2 import PredictionMatrix, _apply_weights, _predict_split, _standardize_by_validation
from research.ensemble_v2 import load_checkpoints
from research.expert_stack_v2 import (
    FeatureSplit,
    _append_gnn_risk_features,
    _load_feature_splits,
    _standardize_baseline_for_training,
    _standardize_feature_splits,
    _zscore_torch,
)
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import resolve_device


@dataclass(frozen=True)
class AdapterPrediction:
    name: str
    train_risk: np.ndarray
    val_risk: np.ndarray
    test_risk: np.ndarray
    val_c_index: float
    test_c_index: float
    best_epoch: int
    metadata: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class RiskCandidate:
    name: str
    val_risk: np.ndarray
    test_risk: np.ndarray
    val_c_index: float
    test_c_index: float
    metadata: dict[str, Any]


class BoundedMlpResidualAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        dropout: float = 0.30,
        max_delta: float = 0.35,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, baseline_risk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.max_delta * torch.tanh(self.network(features).squeeze(-1))
        return baseline_risk + delta, delta


class DisagreementGatedResidualAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        gate_dim: int,
        hidden_dim: int = 24,
        gate_hidden_dim: int = 8,
        dropout: float = 0.30,
        max_delta: float = 0.35,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.residual_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(gate_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.constant_(self.gate_network[-1].bias, -1.0)

    def forward(
        self,
        features: torch.Tensor,
        disagreement: torch.Tensor,
        baseline_risk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = torch.tanh(self.residual_network(features).squeeze(-1))
        gate = torch.sigmoid(self.gate_network(disagreement).squeeze(-1))
        delta = self.max_delta * gate * residual
        return baseline_risk + delta, delta, gate


def evaluate_risk_adapters(
    config_path: str,
    checkpoint_glob: str,
    *,
    split_seed: int = 42,
    device_arg: str = "cuda",
    adapter_device_arg: str = "cpu",
    adapter_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    max_deltas: Sequence[float] = (0.20, 0.35, 0.50),
    distillation_weights: Sequence[float] = (0.10,),
    delta_l2_weights: Sequence[float] = (0.08,),
    baseline_modes: Sequence[str] = ("raw_top3", "standardized_top3"),
    selection_reference_mode: str = "standardized_top3",
    softmax_temperature: float = 0.003,
    epochs: int = 140,
    patience: int = 20,
    include_plain: bool = True,
    include_gated: bool = True,
    min_validation_delta: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    device = resolve_device(device_arg)
    adapter_device = resolve_device(adapter_device_arg)

    gnn_train = _predict_split(config, checkpoints, split_seed=split_seed, split="train", device=device)
    gnn_val = _predict_split(config, checkpoints, split_seed=split_seed, split="val", device=device)
    gnn_test = _predict_split(config, checkpoints, split_seed=split_seed, split="test", device=device)
    reference_weights = [1.0 / len(checkpoints) for _ in checkpoints]
    raw_train_risk = _apply_weights(gnn_train.risk_matrix, reference_weights)
    raw_val_risk = _apply_weights(gnn_val.risk_matrix, reference_weights)
    raw_test_risk = _apply_weights(gnn_test.risk_matrix, reference_weights)
    gnn_train_standardized, gnn_val_standardized, gnn_test_standardized, gnn_risk_scaler = (
        _standardize_gnn_members_by_validation(
            train_matrix=gnn_train.risk_matrix,
            val_matrix=gnn_val.risk_matrix,
            test_matrix=gnn_test.risk_matrix,
        )
    )

    member_val_c_indices = [
        concordance_index(gnn_val.time, gnn_val.event, gnn_val.risk_matrix[index])
        for index in range(gnn_val.risk_matrix.shape[0])
    ]
    softmax_weights = _validation_softmax_weights(member_val_c_indices, temperature=softmax_temperature)
    softmax_train_risk = _apply_weights(gnn_train_standardized, softmax_weights)
    softmax_val_risk = _apply_weights(gnn_val_standardized, softmax_weights)
    softmax_test_risk = _apply_weights(gnn_test_standardized, softmax_weights)
    top3_indices = np.argsort(np.asarray(member_val_c_indices, dtype=float))[-min(3, len(member_val_c_indices)) :][::-1]
    raw_top3_train_risk = np.mean(gnn_train.risk_matrix[top3_indices], axis=0)
    raw_top3_val_risk = np.mean(gnn_val.risk_matrix[top3_indices], axis=0)
    raw_top3_test_risk = np.mean(gnn_test.risk_matrix[top3_indices], axis=0)
    standardized_top3_train_risk = np.mean(gnn_train_standardized[top3_indices], axis=0)
    standardized_top3_val_risk = np.mean(gnn_val_standardized[top3_indices], axis=0)
    standardized_top3_test_risk = np.mean(gnn_test_standardized[top3_indices], axis=0)

    base_feature_names, train_raw, val_raw, test_raw = _load_feature_splits(config, split_seed)
    train_scaled, val_scaled, test_scaled, feature_scaler = _standardize_feature_splits(
        train_raw.features,
        val_raw.features,
        test_raw.features,
    )
    train_features = FeatureSplit(train_raw.sample_ids, train_raw.time, train_raw.event, train_scaled)
    val_features = FeatureSplit(val_raw.sample_ids, val_raw.time, val_raw.event, val_scaled)
    test_features = FeatureSplit(test_raw.sample_ids, test_raw.time, test_raw.event, test_scaled)
    risk_feature_names, train_features, val_features, test_features, risk_feature_scaler = _append_gnn_risk_features(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        gnn_train=gnn_train,
        gnn_val=gnn_val,
        gnn_test=gnn_test,
        reference_train_risk=raw_train_risk,
        reference_val_risk=raw_val_risk,
        reference_test_risk=raw_test_risk,
    )
    gate_feature_names, train_gate, val_gate, test_gate, gate_scaler = _build_standardized_disagreement_splits(
        gnn_train=gnn_train,
        gnn_val=gnn_val,
        gnn_test=gnn_test,
        raw_train_risk=raw_train_risk,
        raw_val_risk=raw_val_risk,
        raw_test_risk=raw_test_risk,
        top_indices=top3_indices,
    )

    adapters: list[AdapterPrediction] = []
    baseline_specs = _build_baseline_specs(
        baseline_modes=baseline_modes,
        raw_top3_train_risk=raw_top3_train_risk,
        raw_top3_val_risk=raw_top3_val_risk,
        raw_top3_test_risk=raw_top3_test_risk,
        standardized_top3_train_risk=standardized_top3_train_risk,
        standardized_top3_val_risk=standardized_top3_val_risk,
        standardized_top3_test_risk=standardized_top3_test_risk,
        softmax_train_risk=softmax_train_risk,
        softmax_val_risk=softmax_val_risk,
        softmax_test_risk=softmax_test_risk,
        softmax_temperature=softmax_temperature,
    )
    reference_spec = next(
        (baseline for baseline in baseline_specs if baseline["mode"] == str(selection_reference_mode)),
        None,
    )
    if reference_spec is None:
        raise ValueError("selection_reference_mode must be included in baseline_modes.")
    for baseline in baseline_specs:
        for seed in adapter_seeds:
            for max_delta in max_deltas:
                for distillation_weight in distillation_weights:
                    for delta_l2_weight in delta_l2_weights:
                        if include_plain:
                            adapters.append(
                                _fit_plain_adapter(
                                    name=(
                                        f"plain_mlp_{baseline['tag']}_seed{int(seed)}"
                                        f"_d{_tag_float(max_delta)}"
                                        f"_dist{_tag_float(distillation_weight)}"
                                        f"_l2{_tag_float(delta_l2_weight)}"
                                    ),
                                    baseline_name=str(baseline["name"]),
                                    input_dim=train_features.features.shape[1],
                                    train_features=train_features,
                                    val_features=val_features,
                                    test_features=test_features,
                                    train_baseline_risk=np.asarray(baseline["train_risk"], dtype=float),
                                    val_baseline_risk=np.asarray(baseline["val_risk"], dtype=float),
                                    test_baseline_risk=np.asarray(baseline["test_risk"], dtype=float),
                                    device=adapter_device,
                                    seed=int(seed) + 30000,
                                    epochs=epochs,
                                    patience=patience,
                                    max_delta=float(max_delta),
                                    distillation_weight=float(distillation_weight),
                                    delta_l2_weight=float(delta_l2_weight),
                                )
                            )
                        if include_gated:
                            adapters.append(
                                _fit_gated_adapter(
                                    name=(
                                        f"gated_mlp_{baseline['tag']}_seed{int(seed)}"
                                        f"_d{_tag_float(max_delta)}"
                                        f"_dist{_tag_float(distillation_weight)}"
                                        f"_l2{_tag_float(delta_l2_weight)}"
                                    ),
                                    baseline_name=str(baseline["name"]),
                                    input_dim=train_features.features.shape[1],
                                    gate_dim=train_gate.shape[1],
                                    train_features=train_features,
                                    val_features=val_features,
                                    test_features=test_features,
                                    train_gate=train_gate,
                                    val_gate=val_gate,
                                    test_gate=test_gate,
                                    train_baseline_risk=np.asarray(baseline["train_risk"], dtype=float),
                                    val_baseline_risk=np.asarray(baseline["val_risk"], dtype=float),
                                    test_baseline_risk=np.asarray(baseline["test_risk"], dtype=float),
                                    device=adapter_device,
                                    seed=int(seed) + 40000,
                                    epochs=epochs,
                                    patience=patience,
                                    max_delta=float(max_delta),
                                    distillation_weight=float(distillation_weight),
                                    delta_l2_weight=float(delta_l2_weight),
                                )
                            )

    candidates = _build_adapter_candidates(
        raw_val_risk=raw_val_risk,
        raw_test_risk=raw_test_risk,
        reference_name=str(reference_spec["name"]),
        reference_val_risk=np.asarray(reference_spec["val_risk"], dtype=float),
        reference_test_risk=np.asarray(reference_spec["test_risk"], dtype=float),
        adapters=adapters,
        val_time=gnn_val.time,
        val_event=gnn_val.event,
        test_time=gnn_test.time,
        test_event=gnn_test.event,
    )
    raw_top3_val_c_index = concordance_index(gnn_val.time, gnn_val.event, raw_top3_val_risk)
    raw_top3_test_c_index = concordance_index(gnn_test.time, gnn_test.event, raw_top3_test_risk)
    top3_val_c_index = concordance_index(gnn_val.time, gnn_val.event, standardized_top3_val_risk)
    reference_val_c_index = concordance_index(
        gnn_val.time,
        gnn_val.event,
        np.asarray(reference_spec["val_risk"], dtype=float),
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.name == f"{reference_spec['name']}_reference"
        or candidate.val_c_index - reference_val_c_index >= float(min_validation_delta)
    ]
    selected = max(eligible, key=lambda candidate: candidate.val_c_index)

    raw_test_c_index = concordance_index(gnn_test.time, gnn_test.event, raw_test_risk)
    top3_test_c_index = concordance_index(gnn_test.time, gnn_test.event, standardized_top3_test_risk)
    selected_adapter = next((adapter for adapter in adapters if adapter.name == selected.name), None)
    selected_adapter_artifact = None
    if output_path is not None and selected_adapter is not None:
        summary_path = Path(output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path = summary_path.with_name(f"{summary_path.stem}_selected_adapter.pt")
        reference_train_risk = np.asarray(reference_spec["train_risk"], dtype=float)
        torch.save(
            {
                "format_version": 1,
                "candidate_name": selected_adapter.name,
                "state_dict": selected_adapter.state_dict,
                "metadata": selected_adapter.metadata,
                "input_dim": int(train_features.features.shape[1]),
                "feature_names": base_feature_names + risk_feature_names,
                "feature_scaler": feature_scaler,
                "risk_feature_scaler": risk_feature_scaler,
                "gnn_member_risk_scaler": gnn_risk_scaler,
                "disagreement_feature_names": gate_feature_names,
                "disagreement_feature_scaler": gate_scaler,
                "selection_reference_mode": str(selection_reference_mode),
                "selection_reference_name": str(reference_spec["name"]),
                "selection_reference_train_mean": float(reference_train_risk.mean()),
                "selection_reference_train_std": max(float(reference_train_risk.std()), 1e-6),
                "softmax_temperature": float(softmax_temperature),
                "softmax_weights": [float(value) for value in softmax_weights],
                "checkpoints": [str(path) for path in checkpoints],
            },
            artifact_path,
        )
        selected_adapter_artifact = str(artifact_path.as_posix())
    result = {
        "config_path": config_path,
        "checkpoint_glob": checkpoint_glob,
        "split_seed": int(split_seed),
        "device": str(device),
        "adapter_device": str(adapter_device),
        "num_gnn_models": len(checkpoints),
        "adapter_count": len(adapters),
        "adapter_grid": {
            "seeds": [int(seed) for seed in adapter_seeds],
            "max_deltas": [float(value) for value in max_deltas],
            "distillation_weights": [float(value) for value in distillation_weights],
            "delta_l2_weights": [float(value) for value in delta_l2_weights],
            "baseline_modes": [str(value) for value in baseline_modes],
            "selection_reference_mode": str(selection_reference_mode),
            "softmax_temperature": float(softmax_temperature),
            "include_plain": bool(include_plain),
            "include_gated": bool(include_gated),
        },
        "references": {
            "raw_mean": {
                "validation_c_index": concordance_index(gnn_val.time, gnn_val.event, raw_val_risk),
                "test_c_index": raw_test_c_index,
            },
            "gnn_top3_raw": {
                "validation_c_index": raw_top3_val_c_index,
                "test_c_index": raw_top3_test_c_index,
                "indices": [int(index) for index in top3_indices.tolist()],
                "checkpoint_names": [Path(checkpoints[int(index)]).parent.name for index in top3_indices],
                "risk_scale": "raw_member_mean",
            },
            "gnn_top3": {
                "validation_c_index": top3_val_c_index,
                "test_c_index": top3_test_c_index,
                "indices": [int(index) for index in top3_indices.tolist()],
                "checkpoint_names": [Path(checkpoints[int(index)]).parent.name for index in top3_indices],
                "risk_scale": "validation_standardized_member_mean",
            },
            "softmax_ensemble": {
                "validation_c_index": concordance_index(gnn_val.time, gnn_val.event, softmax_val_risk),
                "test_c_index": concordance_index(gnn_test.time, gnn_test.event, softmax_test_risk),
                "weights": [float(value) for value in softmax_weights],
                "temperature": float(softmax_temperature),
                "risk_scale": "validation_standardized_softmax_member_weighting",
            },
            "selection_reference": {
                "mode": str(selection_reference_mode),
                "name": str(reference_spec["name"]),
                "validation_c_index": reference_val_c_index,
                "test_c_index": concordance_index(
                    gnn_test.time,
                    gnn_test.event,
                    np.asarray(reference_spec["test_risk"], dtype=float),
                ),
            },
        },
        "selected": {
            "candidate_name": selected.name,
            "validation_c_index": selected.val_c_index,
            "test_c_index": selected.test_c_index,
            "test_delta_vs_raw_mean": selected.test_c_index - raw_test_c_index,
            "test_delta_vs_gnn_top3": selected.test_c_index - top3_test_c_index,
            "test_delta_vs_selection_reference": selected.test_c_index
            - concordance_index(
                gnn_test.time,
                gnn_test.event,
                np.asarray(reference_spec["test_risk"], dtype=float),
            ),
            "metadata": selected.metadata,
            "adapter_artifact": selected_adapter_artifact,
        },
        "adapters": [
            {
                "name": adapter.name,
                "validation_c_index": adapter.val_c_index,
                "test_c_index": adapter.test_c_index,
                "best_epoch": adapter.best_epoch,
                "metadata": adapter.metadata,
            }
            for adapter in adapters
        ],
        "candidates": [
            {
                "name": candidate.name,
                "validation_c_index": candidate.val_c_index,
                "validation_delta_vs_gnn_top3": candidate.val_c_index - top3_val_c_index,
                "test_c_index": candidate.test_c_index,
                "test_delta_vs_raw_mean": candidate.test_c_index - raw_test_c_index,
                "test_delta_vs_gnn_top3": candidate.test_c_index - top3_test_c_index,
                "metadata": candidate.metadata,
            }
            for candidate in candidates
        ],
        "feature_names": base_feature_names + risk_feature_names,
        "disagreement_feature_names": gate_feature_names,
        "feature_scaler": feature_scaler,
        "risk_feature_scaler": risk_feature_scaler,
        "gnn_member_risk_scaler": gnn_risk_scaler,
        "disagreement_feature_scaler": gate_scaler,
        "validation_predictions": [
            {
                "sample_id": sample_id,
                "time": float(time),
                "event": float(event),
                "selection_reference_risk": float(reference_risk),
                "selected_risk": float(selected_risk),
            }
            for sample_id, time, event, reference_risk, selected_risk in zip(
                gnn_val.sample_ids,
                gnn_val.time,
                gnn_val.event,
                np.asarray(reference_spec["val_risk"], dtype=float),
                selected.val_risk,
            )
        ],
        "test_predictions": [
            {
                "sample_id": sample_id,
                "time": float(time),
                "event": float(event),
                "raw_mean_risk": float(raw_risk),
                "gnn_top3_raw_risk": float(raw_top3_risk),
                "gnn_top3_risk": float(top3_risk),
                "selection_reference_risk": float(reference_risk),
                "selected_risk": float(selected_risk),
            }
            for sample_id, time, event, raw_risk, raw_top3_risk, top3_risk, reference_risk, selected_risk in zip(
                gnn_test.sample_ids,
                gnn_test.time,
                gnn_test.event,
                raw_test_risk,
                raw_top3_test_risk,
                standardized_top3_test_risk,
                np.asarray(reference_spec["test_risk"], dtype=float),
                selected.test_risk,
            )
        ],
        "interpretation": (
            "Risk Adapter v2 trains bounded residual adapters on top of validation-selected GNN top3 risk. "
            "The gated variant scales the correction by GNN-member disagreement features, so corrections are "
            "encouraged where the ensemble is uncertain."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _fit_plain_adapter(
    *,
    name: str,
    baseline_name: str,
    input_dim: int,
    train_features: FeatureSplit,
    val_features: FeatureSplit,
    test_features: FeatureSplit,
    train_baseline_risk: np.ndarray,
    val_baseline_risk: np.ndarray,
    test_baseline_risk: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    max_delta: float,
    distillation_weight: float,
    delta_l2_weight: float,
) -> AdapterPrediction:
    model_factory = lambda: BoundedMlpResidualAdapter(input_dim=input_dim, max_delta=max_delta)
    return _fit_residual_adapter(
        name=name,
        model_factory=model_factory,
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_baseline_risk=train_baseline_risk,
        val_baseline_risk=val_baseline_risk,
        test_baseline_risk=test_baseline_risk,
        train_gate=None,
        val_gate=None,
        test_gate=None,
        device=device,
        seed=seed,
        epochs=epochs,
        patience=patience,
        lr=0.0015,
        weight_decay=0.05,
        distillation_weight=distillation_weight,
        delta_l2_weight=delta_l2_weight,
        metadata={
            "adapter_type": "plain_bounded_mlp",
            "hidden_dim": 24,
            "dropout": 0.30,
            "max_delta": float(max_delta),
            "distillation_weight": float(distillation_weight),
            "delta_l2_weight": float(delta_l2_weight),
            "baseline": baseline_name,
        },
    )


def _fit_gated_adapter(
    *,
    name: str,
    baseline_name: str,
    input_dim: int,
    gate_dim: int,
    train_features: FeatureSplit,
    val_features: FeatureSplit,
    test_features: FeatureSplit,
    train_gate: np.ndarray,
    val_gate: np.ndarray,
    test_gate: np.ndarray,
    train_baseline_risk: np.ndarray,
    val_baseline_risk: np.ndarray,
    test_baseline_risk: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    max_delta: float,
    distillation_weight: float,
    delta_l2_weight: float,
) -> AdapterPrediction:
    model_factory = lambda: DisagreementGatedResidualAdapter(
        input_dim=input_dim,
        gate_dim=gate_dim,
        max_delta=max_delta,
    )
    return _fit_residual_adapter(
        name=name,
        model_factory=model_factory,
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_baseline_risk=train_baseline_risk,
        val_baseline_risk=val_baseline_risk,
        test_baseline_risk=test_baseline_risk,
        train_gate=train_gate,
        val_gate=val_gate,
        test_gate=test_gate,
        device=device,
        seed=seed,
        epochs=epochs,
        patience=patience,
        lr=0.0015,
        weight_decay=0.05,
        distillation_weight=distillation_weight,
        delta_l2_weight=delta_l2_weight,
        metadata={
            "adapter_type": "disagreement_gated_mlp",
            "hidden_dim": 24,
            "gate_hidden_dim": 8,
            "dropout": 0.30,
            "max_delta": float(max_delta),
            "distillation_weight": float(distillation_weight),
            "delta_l2_weight": float(delta_l2_weight),
            "baseline": baseline_name,
        },
    )


def _build_baseline_specs(
    *,
    baseline_modes: Sequence[str],
    raw_top3_train_risk: np.ndarray,
    raw_top3_val_risk: np.ndarray,
    raw_top3_test_risk: np.ndarray,
    standardized_top3_train_risk: np.ndarray,
    standardized_top3_val_risk: np.ndarray,
    standardized_top3_test_risk: np.ndarray,
    softmax_train_risk: np.ndarray | None = None,
    softmax_val_risk: np.ndarray | None = None,
    softmax_test_risk: np.ndarray | None = None,
    softmax_temperature: float = 0.003,
) -> list[dict[str, Any]]:
    available = {
        "raw_top3": {
            "mode": "raw_top3",
            "name": "gnn_top3_raw",
            "tag": "raw_top3",
            "train_risk": raw_top3_train_risk,
            "val_risk": raw_top3_val_risk,
            "test_risk": raw_top3_test_risk,
        },
        "standardized_top3": {
            "mode": "standardized_top3",
            "name": "gnn_top3_standardized",
            "tag": "standardized_top3",
            "train_risk": standardized_top3_train_risk,
            "val_risk": standardized_top3_val_risk,
            "test_risk": standardized_top3_test_risk,
        },
    }
    if softmax_train_risk is not None and softmax_val_risk is not None and softmax_test_risk is not None:
        available["softmax_ensemble"] = {
            "mode": "softmax_ensemble",
            "name": f"gnn_softmax_ensemble_t{_tag_float(softmax_temperature)}",
            "tag": f"softmax_t{_tag_float(softmax_temperature)}",
            "train_risk": softmax_train_risk,
            "val_risk": softmax_val_risk,
            "test_risk": softmax_test_risk,
        }
    specs: list[dict[str, Any]] = []
    for mode in baseline_modes:
        key = str(mode).strip()
        if key not in available:
            raise ValueError(f"Unknown baseline mode: {key}")
        specs.append(available[key])
    if not specs:
        raise ValueError("At least one baseline mode is required.")
    return specs


def _fit_residual_adapter(
    *,
    name: str,
    model_factory: Callable[[], nn.Module],
    train_features: FeatureSplit,
    val_features: FeatureSplit,
    test_features: FeatureSplit,
    train_baseline_risk: np.ndarray,
    val_baseline_risk: np.ndarray,
    test_baseline_risk: np.ndarray,
    train_gate: np.ndarray | None,
    val_gate: np.ndarray | None,
    test_gate: np.ndarray | None,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    distillation_weight: float,
    delta_l2_weight: float,
    metadata: dict[str, Any],
) -> AdapterPrediction:
    _set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_x = torch.tensor(train_features.features, dtype=torch.float32, device=device)
    val_x = torch.tensor(val_features.features, dtype=torch.float32, device=device)
    train_gate_x = torch.tensor(train_gate, dtype=torch.float32, device=device) if train_gate is not None else None
    val_gate_x = torch.tensor(val_gate, dtype=torch.float32, device=device) if val_gate is not None else None
    train_time = torch.tensor(train_features.time, dtype=torch.float32, device=device)
    train_event = torch.tensor(train_features.event, dtype=torch.float32, device=device)
    train_base_np = _standardize_baseline_for_training(train_baseline_risk, train_baseline_risk)
    val_base_np = _standardize_baseline_for_training(val_baseline_risk, train_baseline_risk)
    test_base_np = _standardize_baseline_for_training(test_baseline_risk, train_baseline_risk)
    train_base = torch.tensor(train_base_np, dtype=torch.float32, device=device)
    val_base = torch.tensor(val_base_np, dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, delta = _forward_adapter(model, train_x, train_base, train_gate_x)
        cox_loss = cox_ph_loss(risk, train_time, train_event)
        distillation = torch.mean((_zscore_torch(risk) - _zscore_torch(train_base.detach())) ** 2)
        delta_l2 = torch.mean(delta.pow(2))
        loss = cox_loss + float(distillation_weight) * distillation + float(delta_l2_weight) * delta_l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_risk, _ = _forward_adapter(model, val_x, val_base, val_gate_x)
            val_risk_np = val_risk.detach().cpu().numpy()
        val_c_index = concordance_index(val_features.time, val_features.event, val_risk_np)
        if val_c_index > best_val + 0.00025:
            best_val = float(val_c_index)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    model.load_state_dict(best_state)
    train_risk = _predict_adapter(model, train_features.features, train_base_np, train_gate, device)
    val_risk = _predict_adapter(model, val_features.features, val_base_np, val_gate, device)
    test_risk = _predict_adapter(model, test_features.features, test_base_np, test_gate, device)
    return AdapterPrediction(
        name=name,
        train_risk=train_risk,
        val_risk=val_risk,
        test_risk=test_risk,
        val_c_index=concordance_index(val_features.time, val_features.event, val_risk),
        test_c_index=concordance_index(test_features.time, test_features.event, test_risk),
        best_epoch=best_epoch,
        metadata={**metadata, "seed": int(seed)},
        state_dict={
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    )


def _forward_adapter(
    model: nn.Module,
    features: torch.Tensor,
    baseline_risk: torch.Tensor,
    gate_features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_features is None:
        risk, delta = model(features, baseline_risk)
        return risk, delta
    risk, delta, _ = model(features, gate_features, baseline_risk)
    return risk, delta


def _predict_adapter(
    model: nn.Module,
    features: np.ndarray,
    baseline_risk: np.ndarray,
    gate_features: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device)
        base = torch.tensor(baseline_risk, dtype=torch.float32, device=device)
        gate = torch.tensor(gate_features, dtype=torch.float32, device=device) if gate_features is not None else None
        risk, _ = _forward_adapter(model, x, base, gate)
        return risk.detach().cpu().numpy().astype(float)


def _build_adapter_candidates(
    *,
    raw_val_risk: np.ndarray,
    raw_test_risk: np.ndarray,
    reference_name: str,
    reference_val_risk: np.ndarray,
    reference_test_risk: np.ndarray,
    adapters: Sequence[AdapterPrediction],
    val_time: np.ndarray,
    val_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
) -> list[RiskCandidate]:
    candidates: list[RiskCandidate] = [
        _candidate_from_risk("raw_mean_reference", raw_val_risk, raw_test_risk, val_time, val_event, test_time, test_event, {}),
        _candidate_from_risk(
            f"{reference_name}_reference",
            reference_val_risk,
            reference_test_risk,
            val_time,
            val_event,
            test_time,
            test_event,
            {"reference_name": reference_name},
        ),
    ]
    for adapter in adapters:
        candidates.append(
            _candidate_from_risk(
                adapter.name,
                adapter.val_risk,
                adapter.test_risk,
                val_time,
                val_event,
                test_time,
                test_event,
                adapter.metadata,
            )
        )

    if len(adapters) >= 2:
        val_matrix = np.vstack([adapter.val_risk for adapter in adapters])
        test_matrix = np.vstack([adapter.test_risk for adapter in adapters])
        val_standardized, test_standardized, _ = _standardize_by_validation(val_matrix, test_matrix)
        order = np.argsort(np.asarray([adapter.val_c_index for adapter in adapters], dtype=float))
        for count in range(2, min(5, len(adapters)) + 1):
            selected_indices = order[-count:]
            val_risk = val_standardized[selected_indices].mean(axis=0)
            test_risk = test_standardized[selected_indices].mean(axis=0)
            names = [adapters[int(index)].name for index in selected_indices]
            candidates.append(
                _candidate_from_risk(
                    f"adapter_top{count}_val_mean",
                    val_risk,
                    test_risk,
                    val_time,
                    val_event,
                    test_time,
                    test_event,
                    {"adapter_names": names},
                )
            )

        reference_val_std, adapter_val_std, scaler = _standardize_pair_by_validation(
            reference_val_risk,
            reference_test_risk,
            val_matrix,
            test_matrix,
        )
        for adapter_index, adapter in enumerate(adapters):
            for alpha in (0.25, 0.50, 0.75):
                val_risk = (1.0 - alpha) * reference_val_std + alpha * adapter_val_std[adapter_index]
                test_risk = (1.0 - alpha) * scaler["test_reference"] + alpha * scaler["test_adapters"][adapter_index]
                candidates.append(
                    _candidate_from_risk(
                        f"blend_reference_{adapter.name}_alpha{_tag_float(alpha)}",
                        val_risk,
                        test_risk,
                        val_time,
                        val_event,
                        test_time,
                        test_event,
                        {"adapter_name": adapter.name, "alpha": float(alpha)},
                    )
                )
    return candidates


def _candidate_from_risk(
    name: str,
    val_risk: np.ndarray,
    test_risk: np.ndarray,
    val_time: np.ndarray,
    val_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    metadata: dict[str, Any],
) -> RiskCandidate:
    return RiskCandidate(
        name=name,
        val_risk=np.asarray(val_risk, dtype=float),
        test_risk=np.asarray(test_risk, dtype=float),
        val_c_index=concordance_index(val_time, val_event, val_risk),
        test_c_index=concordance_index(test_time, test_event, test_risk),
        metadata=metadata,
    )


def _standardize_pair_by_validation(
    reference_val: np.ndarray,
    reference_test: np.ndarray,
    adapter_val_matrix: np.ndarray,
    adapter_test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    combined_val = np.vstack([np.asarray(reference_val, dtype=float)[None, :], adapter_val_matrix])
    combined_test = np.vstack([np.asarray(reference_test, dtype=float)[None, :], adapter_test_matrix])
    val_standardized, test_standardized, _ = _standardize_by_validation(combined_val, combined_test)
    return val_standardized[0], val_standardized[1:], {
        "test_reference": test_standardized[0],
        "test_adapters": test_standardized[1:],
    }


def _validation_softmax_weights(
    member_val_c_indices: Sequence[float],
    *,
    temperature: float,
) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scores = np.asarray(member_val_c_indices, dtype=float)
    if scores.size == 0:
        raise ValueError("At least one member validation c-index is required.")
    scaled = np.exp((scores - scores.max()) / float(temperature))
    return (scaled / scaled.sum()).astype(float).tolist()


def _standardize_gnn_members_by_validation(
    *,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    val = np.asarray(val_matrix, dtype=float)
    means = val.mean(axis=1, keepdims=True)
    stds = np.maximum(val.std(axis=1, keepdims=True), 1e-6)
    return (
        (np.asarray(train_matrix, dtype=float) - means) / stds,
        (val - means) / stds,
        (np.asarray(test_matrix, dtype=float) - means) / stds,
        {
            "risk_means": means.squeeze(axis=1).astype(float).tolist(),
            "risk_stds": stds.squeeze(axis=1).astype(float).tolist(),
        },
    )


def _build_standardized_disagreement_splits(
    *,
    gnn_train: PredictionMatrix,
    gnn_val: PredictionMatrix,
    gnn_test: PredictionMatrix,
    raw_train_risk: np.ndarray,
    raw_val_risk: np.ndarray,
    raw_test_risk: np.ndarray,
    top_indices: Sequence[int],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    names = [
        "risk_std_all",
        "risk_range_all",
        "risk_std_top3",
        "risk_range_top3",
        "abs_top3_minus_raw_mean",
        "max_abs_member_minus_raw_mean",
    ]
    train = _disagreement_features(gnn_train.risk_matrix, raw_train_risk, top_indices)
    val = _disagreement_features(gnn_val.risk_matrix, raw_val_risk, top_indices)
    test = _disagreement_features(gnn_test.risk_matrix, raw_test_risk, top_indices)
    means = train.mean(axis=0, keepdims=True)
    stds = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
    return (
        names,
        (train - means) / stds,
        (val - means) / stds,
        (test - means) / stds,
        {
            "feature_means": means.squeeze(axis=0).astype(float).tolist(),
            "feature_stds": stds.squeeze(axis=0).astype(float).tolist(),
        },
    )


def _disagreement_features(risk_matrix: np.ndarray, raw_mean_risk: np.ndarray, top_indices: Sequence[int]) -> np.ndarray:
    matrix = np.asarray(risk_matrix, dtype=float)
    raw = np.asarray(raw_mean_risk, dtype=float)
    top_matrix = matrix[np.asarray(top_indices, dtype=int)]
    top_mean = top_matrix.mean(axis=0)
    return np.vstack(
        [
            matrix.std(axis=0),
            matrix.max(axis=0) - matrix.min(axis=0),
            top_matrix.std(axis=0),
            top_matrix.max(axis=0) - top_matrix.min(axis=0),
            np.abs(top_mean - raw),
            np.max(np.abs(matrix - raw[None, :]), axis=0),
        ]
    ).T


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _tag_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument(
        "--checkpoint-glob",
        default="outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--adapter-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--adapter-seeds", default="7,21,42,123,2026")
    parser.add_argument("--max-deltas", default="0.2,0.35,0.5")
    parser.add_argument("--distillation-weights", default="0.1")
    parser.add_argument("--delta-l2-weights", default="0.08")
    parser.add_argument("--baseline-modes", default="raw_top3,standardized_top3")
    parser.add_argument("--selection-reference-mode", default="standardized_top3")
    parser.add_argument("--softmax-temperature", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--disable-plain", action="store_true")
    parser.add_argument("--disable-gated", action="store_true")
    parser.add_argument("--min-validation-delta", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/risk_adapter_v2/risk_adapter_v2_summary.json",
    )
    args = parser.parse_args()
    result = evaluate_risk_adapters(
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        split_seed=args.split_seed,
        device_arg=args.device,
        adapter_device_arg=args.adapter_device,
        adapter_seeds=_parse_int_list(args.adapter_seeds),
        max_deltas=_parse_float_list(args.max_deltas),
        distillation_weights=_parse_float_list(args.distillation_weights),
        delta_l2_weights=_parse_float_list(args.delta_l2_weights),
        baseline_modes=_parse_str_list(args.baseline_modes),
        selection_reference_mode=args.selection_reference_mode,
        softmax_temperature=args.softmax_temperature,
        epochs=args.epochs,
        patience=args.patience,
        include_plain=not args.disable_plain,
        include_gated=not args.disable_gated,
        min_validation_delta=args.min_validation_delta,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
