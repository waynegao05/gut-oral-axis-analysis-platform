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

from ctm_fusion_experiment.models.ctm import CTM
from research.ensemble_stack_v2 import PredictionMatrix, _apply_weights, _predict_split, _standardize_by_validation
from research.ensemble_v2 import build_loader, load_checkpoints
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import resolve_device


@dataclass(frozen=True)
class FeatureSplit:
    sample_ids: list[str]
    time: np.ndarray
    event: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class ExpertPrediction:
    name: str
    val_risk: np.ndarray
    test_risk: np.ndarray
    val_c_index: float
    test_c_index: float
    best_epoch: int


@dataclass(frozen=True)
class StackCandidate:
    name: str
    weights: list[float]
    val_risk: np.ndarray
    test_risk: np.ndarray
    val_c_index: float
    test_c_index: float


class LinearCoxExpert(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.risk_head = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.risk_head(features).squeeze(-1)


class MlpCoxExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.20) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MlpResidualRiskExpert(nn.Module):
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


class CTMFeatureCoxExpert(nn.Module):
    def __init__(
        self,
        num_features: int,
        token_dim: int = 16,
        d_model: int = 32,
        iterations: int = 6,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.feature_embedding = nn.Parameter(torch.empty(num_features, token_dim))
        self.value_projection = nn.Linear(1, token_dim, bias=False)
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        self.ctm = CTM(
            d_model=d_model,
            d_input=token_dim,
            iterations=iterations,
            memory_length=4,
            nlm_hidden_dim=32,
            n_heads=4,
            n_synch_action=16,
            n_synch_out=16,
            n_self_pairs=4,
            synapse_depth=2,
            dropout=dropout,
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_embedding.unsqueeze(0) + self.value_projection(features.unsqueeze(-1))
        representations = self.ctm(tokens).representations
        pooled = torch.cat([representations[:, -1, :], representations.mean(dim=1)], dim=1)
        return self.risk_head(pooled).squeeze(-1)


class CTMResidualRiskExpert(nn.Module):
    def __init__(
        self,
        num_features: int,
        token_dim: int = 16,
        d_model: int = 32,
        iterations: int = 6,
        dropout: float = 0.12,
        max_delta: float = 0.75,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.feature_embedding = nn.Parameter(torch.empty(num_features, token_dim))
        self.value_projection = nn.Linear(1, token_dim, bias=False)
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        self.ctm = CTM(
            d_model=d_model,
            d_input=token_dim,
            iterations=iterations,
            memory_length=4,
            nlm_hidden_dim=32,
            n_heads=4,
            n_synch_action=16,
            n_synch_out=16,
            n_self_pairs=4,
            synapse_depth=2,
            dropout=dropout,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor, baseline_risk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_embedding.unsqueeze(0) + self.value_projection(features.unsqueeze(-1))
        representations = self.ctm(tokens).representations
        pooled = torch.cat([representations[:, -1, :], representations.mean(dim=1)], dim=1)
        delta = self.max_delta * torch.tanh(self.delta_head(pooled).squeeze(-1))
        return baseline_risk + delta, delta


def evaluate_expert_stack(
    config_path: str,
    checkpoint_glob: str,
    *,
    split_seed: int | None = None,
    device_arg: str = "cuda",
    expert_device_arg: str = "cpu",
    expert_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    train_epochs: int = 160,
    patience: int = 24,
    min_validation_delta: float = 0.0,
    include_linear: bool = True,
    include_mlp: bool = True,
    include_standalone_mlp: bool = True,
    include_mlp_residual: bool = True,
    include_ctm: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    if split_seed is None:
        split_seed = config["train"].get("split_seed")
    if split_seed is None:
        split_seed = int(config["seed"])
    device = resolve_device(device_arg)
    expert_device = resolve_device(expert_device_arg)

    gnn_train = _predict_split(config, checkpoints, split_seed=int(split_seed), split="train", device=device)
    gnn_val = _predict_split(config, checkpoints, split_seed=int(split_seed), split="val", device=device)
    gnn_test = _predict_split(config, checkpoints, split_seed=int(split_seed), split="test", device=device)
    feature_names, train_features_raw, val_features_raw, test_features_raw = _load_feature_splits(config, int(split_seed))
    train_scaled, val_scaled, test_scaled, feature_scaler = _standardize_feature_splits(
        train_features_raw.features,
        val_features_raw.features,
        test_features_raw.features,
    )
    train_features = FeatureSplit(
        train_features_raw.sample_ids,
        train_features_raw.time,
        train_features_raw.event,
        train_scaled,
    )
    val_features = FeatureSplit(
        val_features_raw.sample_ids,
        val_features_raw.time,
        val_features_raw.event,
        val_scaled,
    )
    test_features = FeatureSplit(
        test_features_raw.sample_ids,
        test_features_raw.time,
        test_features_raw.event,
        test_scaled,
    )

    _assert_same_order(gnn_val, val_features.sample_ids, "validation")
    _assert_same_order(gnn_test, test_features.sample_ids, "test")
    reference_weights = [1.0 / len(checkpoints) for _ in checkpoints]
    reference_train_risk = _apply_weights(gnn_train.risk_matrix, reference_weights)
    reference_val_risk = _apply_weights(gnn_val.risk_matrix, reference_weights)
    reference_test_risk = _apply_weights(gnn_test.risk_matrix, reference_weights)
    member_val_c_indices = [
        concordance_index(gnn_val.time, gnn_val.event, gnn_val.risk_matrix[index])
        for index in range(gnn_val.risk_matrix.shape[0])
    ]
    top3_indices = np.argsort(np.asarray(member_val_c_indices, dtype=float))[-min(3, len(member_val_c_indices)) :]
    top3_train_risk = np.mean(gnn_train.risk_matrix[top3_indices], axis=0)
    top3_val_risk = np.mean(gnn_val.risk_matrix[top3_indices], axis=0)
    top3_test_risk = np.mean(gnn_test.risk_matrix[top3_indices], axis=0)
    risk_feature_names, train_features, val_features, test_features, risk_feature_scaler = _append_gnn_risk_features(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        gnn_train=gnn_train,
        gnn_val=gnn_val,
        gnn_test=gnn_test,
        reference_train_risk=reference_train_risk,
        reference_val_risk=reference_val_risk,
        reference_test_risk=reference_test_risk,
    )
    feature_names = feature_names + risk_feature_names

    experts = _train_experts(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        input_dim=len(feature_names),
        train_baseline_risk=reference_train_risk,
        val_baseline_risk=reference_val_risk,
        test_baseline_risk=reference_test_risk,
        train_top3_baseline_risk=top3_train_risk,
        val_top3_baseline_risk=top3_val_risk,
        test_top3_baseline_risk=top3_test_risk,
        seeds=expert_seeds,
        device=expert_device,
        epochs=train_epochs,
        patience=patience,
        include_linear=include_linear,
        include_mlp=include_mlp,
        include_standalone_mlp=include_standalone_mlp,
        include_mlp_residual=include_mlp_residual,
        include_ctm=include_ctm,
    )

    reference_val_c_index = concordance_index(gnn_val.time, gnn_val.event, reference_val_risk)
    reference_test_c_index = concordance_index(gnn_test.time, gnn_test.event, reference_test_risk)

    source_names = ["reference_raw_mean"]
    source_val_rows = [reference_val_risk]
    source_test_rows = [reference_test_risk]
    source_names.extend([f"gnn:{Path(path).parent.name}" for path in checkpoints])
    source_val_rows.extend([gnn_val.risk_matrix[index] for index in range(gnn_val.risk_matrix.shape[0])])
    source_test_rows.extend([gnn_test.risk_matrix[index] for index in range(gnn_test.risk_matrix.shape[0])])
    source_names.extend([f"expert:{expert.name}" for expert in experts])
    source_val_rows.extend([expert.val_risk for expert in experts])
    source_test_rows.extend([expert.test_risk for expert in experts])
    source_val = np.vstack(source_val_rows).astype(float)
    source_test = np.vstack(source_test_rows).astype(float)
    source_val_std, source_test_std, risk_scaler = _standardize_by_validation(source_val, source_test)

    candidates = _build_stack_candidates(
        source_names=source_names,
        val_standardized=source_val_std,
        test_standardized=source_test_std,
        val_time=gnn_val.time,
        val_event=gnn_val.event,
        test_time=gnn_test.time,
        test_event=gnn_test.event,
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.val_c_index - reference_val_c_index >= float(min_validation_delta)
    ]
    selected = max(eligible, key=lambda candidate: candidate.val_c_index) if eligible else None
    if selected is None:
        selected_name = "reference_raw_mean"
        selected_weights = [1.0] + [0.0 for _ in source_names[1:]]
        selected_val_c_index = reference_val_c_index
        selected_test_c_index = reference_test_c_index
        selected_test_risk = reference_test_risk
    else:
        selected_name = selected.name
        selected_weights = selected.weights
        selected_val_c_index = selected.val_c_index
        selected_test_c_index = selected.test_c_index
        selected_test_risk = selected.test_risk

    result = {
        "config_path": config_path,
        "checkpoint_glob": checkpoint_glob,
        "split_seed": int(split_seed),
        "device": str(device),
        "expert_device": str(expert_device),
        "num_gnn_models": len(checkpoints),
        "num_auxiliary_experts": len(experts),
        "source_names": source_names,
        "residual_baselines": {
            "raw_mean": "reference_raw_mean",
            "gnn_top3_indices": [int(index) for index in top3_indices.tolist()],
            "gnn_top3_source_names": [f"gnn:{Path(checkpoints[int(index)]).parent.name}" for index in top3_indices],
        },
        "reference": {
            "candidate_name": "reference_raw_mean",
            "validation_c_index": reference_val_c_index,
            "test_c_index": reference_test_c_index,
        },
        "selected": {
            "candidate_name": selected_name,
            "weights": selected_weights,
            "active_sources": _active_source_weights(source_names, selected_weights),
            "validation_c_index": selected_val_c_index,
            "validation_delta": selected_val_c_index - reference_val_c_index,
            "test_c_index": selected_test_c_index,
            "test_delta": selected_test_c_index - reference_test_c_index,
        },
        "gnn_members": [
            {
                "name": source_names[1 + index],
                "checkpoint": str(checkpoint),
                "validation_c_index": concordance_index(gnn_val.time, gnn_val.event, gnn_val.risk_matrix[index]),
                "test_c_index": concordance_index(gnn_test.time, gnn_test.event, gnn_test.risk_matrix[index]),
            }
            for index, checkpoint in enumerate(checkpoints)
        ],
        "auxiliary_experts": [
            {
                "name": expert.name,
                "validation_c_index": expert.val_c_index,
                "test_c_index": expert.test_c_index,
                "best_epoch": expert.best_epoch,
            }
            for expert in experts
        ],
        "feature_names": feature_names,
        "feature_scaler": feature_scaler,
        "risk_feature_scaler": risk_feature_scaler,
        "risk_standardization": risk_scaler,
        "candidates": [
            {
                "candidate_name": candidate.name,
                "active_sources": _active_source_weights(source_names, candidate.weights),
                "validation_c_index": candidate.val_c_index,
                "validation_delta": candidate.val_c_index - reference_val_c_index,
                "test_c_index": candidate.test_c_index,
                "test_delta": candidate.test_c_index - reference_test_c_index,
            }
            for candidate in candidates
        ],
        "test_predictions": [
            {
                "sample_id": sample_id,
                "time": float(time),
                "event": float(event),
                "reference_risk": float(reference_risk),
                "selected_risk": float(selected_risk),
            }
            for sample_id, time, event, reference_risk, selected_risk in zip(
                gnn_test.sample_ids,
                gnn_test.time,
                gnn_test.event,
                reference_test_risk,
                selected_test_risk,
            )
        ],
        "interpretation": (
            "Aggressive validation-gated expert stacking over the fixed Cox mainline. "
            "Auxiliary standalone and residual experts are trained only on the fixed train split; "
            "the selected stack is chosen only on validation and evaluated once on test."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _train_experts(
    *,
    train_features: FeatureSplit,
    val_features: FeatureSplit,
    test_features: FeatureSplit,
    input_dim: int,
    train_baseline_risk: np.ndarray,
    val_baseline_risk: np.ndarray,
    test_baseline_risk: np.ndarray,
    train_top3_baseline_risk: np.ndarray,
    val_top3_baseline_risk: np.ndarray,
    test_top3_baseline_risk: np.ndarray,
    seeds: Sequence[int],
    device: torch.device,
    epochs: int,
    patience: int,
    include_linear: bool,
    include_mlp: bool,
    include_standalone_mlp: bool,
    include_mlp_residual: bool,
    include_ctm: bool,
) -> list[ExpertPrediction]:
    experts: list[ExpertPrediction] = []
    if include_linear:
        experts.append(
            _fit_expert(
                name="linear_feature_cox",
                model_factory=lambda: LinearCoxExpert(input_dim),
                train_features=train_features,
                val_features=val_features,
                test_features=test_features,
                device=device,
                seed=0,
                epochs=max(80, epochs // 2),
                patience=max(12, patience // 2),
                lr=0.02,
                weight_decay=0.05,
            )
        )
    for seed in seeds:
        if include_mlp and include_standalone_mlp:
            experts.append(
                _fit_expert(
                    name=f"mlp_feature_cox_seed{int(seed)}",
                    model_factory=lambda input_dim=input_dim: MlpCoxExpert(input_dim, hidden_dim=32, dropout=0.20),
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    device=device,
                    seed=int(seed),
                    epochs=epochs,
                    patience=patience,
                    lr=0.003,
                    weight_decay=0.02,
                )
            )
        if include_mlp and include_mlp_residual:
            experts.append(
                _fit_residual_expert(
                    name=f"mlp_residual_raw_mean_seed{int(seed)}",
                    model_factory=lambda input_dim=input_dim: MlpResidualRiskExpert(input_dim),
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    train_baseline_risk=train_baseline_risk,
                    val_baseline_risk=val_baseline_risk,
                    test_baseline_risk=test_baseline_risk,
                    device=device,
                    seed=int(seed) + 20000,
                    epochs=epochs,
                    patience=patience,
                    lr=0.0015,
                    weight_decay=0.05,
                    distillation_weight=0.10,
                    delta_l2_weight=0.08,
                )
            )
            experts.append(
                _fit_residual_expert(
                    name=f"mlp_residual_gnn_top3_seed{int(seed)}",
                    model_factory=lambda input_dim=input_dim: MlpResidualRiskExpert(input_dim),
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    train_baseline_risk=train_top3_baseline_risk,
                    val_baseline_risk=val_top3_baseline_risk,
                    test_baseline_risk=test_top3_baseline_risk,
                    device=device,
                    seed=int(seed) + 30000,
                    epochs=epochs,
                    patience=patience,
                    lr=0.0015,
                    weight_decay=0.05,
                    distillation_weight=0.10,
                    delta_l2_weight=0.08,
                )
            )
        if include_ctm:
            experts.append(
                _fit_expert(
                    name=f"ctm_feature_cox_seed{int(seed)}",
                    model_factory=lambda input_dim=input_dim: CTMFeatureCoxExpert(input_dim),
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    device=device,
                    seed=int(seed),
                    epochs=epochs,
                    patience=patience,
                    lr=0.001,
                    weight_decay=0.01,
                )
            )
            experts.append(
                _fit_residual_expert(
                    name=f"ctm_residual_raw_mean_seed{int(seed)}",
                    model_factory=lambda input_dim=input_dim: CTMResidualRiskExpert(input_dim),
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    train_baseline_risk=train_baseline_risk,
                    val_baseline_risk=val_baseline_risk,
                    test_baseline_risk=test_baseline_risk,
                    device=device,
                    seed=int(seed),
                    epochs=epochs,
                    patience=patience,
                    lr=0.001,
                    weight_decay=0.01,
                    distillation_weight=0.02,
                    delta_l2_weight=0.01,
                )
            )
            experts.append(
                _fit_residual_expert(
                    name=f"ctm_residual_gnn_top3_seed{int(seed)}",
                    model_factory=lambda input_dim=input_dim: CTMResidualRiskExpert(input_dim),
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    train_baseline_risk=train_top3_baseline_risk,
                    val_baseline_risk=val_top3_baseline_risk,
                    test_baseline_risk=test_top3_baseline_risk,
                    device=device,
                    seed=int(seed) + 10000,
                    epochs=epochs,
                    patience=patience,
                    lr=0.001,
                    weight_decay=0.01,
                    distillation_weight=0.02,
                    delta_l2_weight=0.01,
                )
            )
    return experts


def _fit_expert(
    *,
    name: str,
    model_factory: Callable[[], nn.Module],
    train_features: FeatureSplit,
    val_features: FeatureSplit,
    test_features: FeatureSplit,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> ExpertPrediction:
    _set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_x = torch.tensor(train_features.features, dtype=torch.float32, device=device)
    train_time = torch.tensor(train_features.time, dtype=torch.float32, device=device)
    train_event = torch.tensor(train_features.event, dtype=torch.float32, device=device)
    val_x = torch.tensor(val_features.features, dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(train_x)
        loss = cox_ph_loss(risk, train_time, train_event)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_risk = model(val_x).detach().cpu().numpy()
        val_c_index = concordance_index(val_features.time, val_features.event, val_risk)
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
    val_risk = _predict_expert(model, val_features.features, device)
    test_risk = _predict_expert(model, test_features.features, device)
    return ExpertPrediction(
        name=name,
        val_risk=val_risk,
        test_risk=test_risk,
        val_c_index=concordance_index(val_features.time, val_features.event, val_risk),
        test_c_index=concordance_index(test_features.time, test_features.event, test_risk),
        best_epoch=best_epoch,
    )


def _fit_residual_expert(
    *,
    name: str,
    model_factory: Callable[[], CTMResidualRiskExpert],
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
    lr: float,
    weight_decay: float,
    distillation_weight: float,
    delta_l2_weight: float,
) -> ExpertPrediction:
    _set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_x = torch.tensor(train_features.features, dtype=torch.float32, device=device)
    train_time = torch.tensor(train_features.time, dtype=torch.float32, device=device)
    train_event = torch.tensor(train_features.event, dtype=torch.float32, device=device)
    train_base_np = _standardize_baseline_for_training(train_baseline_risk, train_baseline_risk)
    val_base_np = _standardize_baseline_for_training(val_baseline_risk, train_baseline_risk)
    test_base_np = _standardize_baseline_for_training(test_baseline_risk, train_baseline_risk)
    train_base = torch.tensor(train_base_np, dtype=torch.float32, device=device)
    val_x = torch.tensor(val_features.features, dtype=torch.float32, device=device)
    val_base = torch.tensor(val_base_np, dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, delta = model(train_x, train_base)
        cox_loss = cox_ph_loss(risk, train_time, train_event)
        distillation = torch.mean((_zscore_torch(risk) - _zscore_torch(train_base.detach())) ** 2)
        delta_l2 = torch.mean(delta.pow(2))
        loss = cox_loss + float(distillation_weight) * distillation + float(delta_l2_weight) * delta_l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_risk, _ = model(val_x, val_base)
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
    val_risk = _predict_residual_expert(model, val_features.features, val_base_np, device)
    test_risk = _predict_residual_expert(model, test_features.features, test_base_np, device)
    return ExpertPrediction(
        name=name,
        val_risk=val_risk,
        test_risk=test_risk,
        val_c_index=concordance_index(val_features.time, val_features.event, val_risk),
        test_c_index=concordance_index(test_features.time, test_features.event, test_risk),
        best_epoch=best_epoch,
    )


def _predict_expert(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device)
        return model(x).detach().cpu().numpy().astype(float)


def _predict_residual_expert(
    model: CTMResidualRiskExpert,
    features: np.ndarray,
    baseline_risk: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device)
        base = torch.tensor(baseline_risk, dtype=torch.float32, device=device)
        risk, _ = model(x, base)
        return risk.detach().cpu().numpy().astype(float)


def _load_feature_splits(config: dict[str, Any], split_seed: int) -> tuple[list[str], FeatureSplit, FeatureSplit, FeatureSplit]:
    _, dataset = build_loader(config, split_seed=split_seed, split="train")
    feature_names = _feature_names(config)
    return (
        feature_names,
        _collect_features(dataset.train_set),
        _collect_features(dataset.val_set),
        _collect_features(dataset.test_set),
    )


def _feature_names(config: dict[str, Any]) -> list[str]:
    names = [f"clinical:{name}" for name in config["model"]["clinical_columns"]]
    names.extend([f"metabolite:{name}" for name in config["model"]["metabolite_columns"]])
    names.extend(
        [
            "graph:node_abundance_mean",
            "graph:node_abundance_std",
            "graph:node_abundance_max",
            "graph:node_function_mean",
            "graph:node_function_std",
            "graph:node_function_max",
            "graph:edge_weight_mean",
            "graph:edge_weight_std",
            "graph:edge_weight_max",
            "graph:edge_weight_min",
            "graph:edge_weight_sum_per_node",
            "graph:density",
        ]
    )
    return names


def _append_gnn_risk_features(
    *,
    train_features: FeatureSplit,
    val_features: FeatureSplit,
    test_features: FeatureSplit,
    gnn_train: PredictionMatrix,
    gnn_val: PredictionMatrix,
    gnn_test: PredictionMatrix,
    reference_train_risk: np.ndarray,
    reference_val_risk: np.ndarray,
    reference_test_risk: np.ndarray,
) -> tuple[list[str], FeatureSplit, FeatureSplit, FeatureSplit, dict[str, list[float]]]:
    _assert_same_order(gnn_train, train_features.sample_ids, "train")
    _assert_same_order(gnn_val, val_features.sample_ids, "validation")
    _assert_same_order(gnn_test, test_features.sample_ids, "test")
    train_risk_features = np.vstack([reference_train_risk, gnn_train.risk_matrix]).T
    val_risk_features = np.vstack([reference_val_risk, gnn_val.risk_matrix]).T
    test_risk_features = np.vstack([reference_test_risk, gnn_test.risk_matrix]).T
    means = train_risk_features.mean(axis=0, keepdims=True)
    stds = np.maximum(train_risk_features.std(axis=0, keepdims=True), 1e-6)

    def append(split: FeatureSplit, risk_features: np.ndarray) -> FeatureSplit:
        scaled_risks = (risk_features - means) / stds
        return FeatureSplit(
            sample_ids=split.sample_ids,
            time=split.time,
            event=split.event,
            features=np.concatenate([split.features, scaled_risks], axis=1),
        )

    risk_feature_names = ["risk:reference_raw_mean"] + [
        f"risk:gnn_member_{index}" for index in range(gnn_train.risk_matrix.shape[0])
    ]
    return (
        risk_feature_names,
        append(train_features, train_risk_features),
        append(val_features, val_risk_features),
        append(test_features, test_risk_features),
        {
            "risk_feature_means": means.squeeze(axis=0).astype(float).tolist(),
            "risk_feature_stds": stds.squeeze(axis=0).astype(float).tolist(),
        },
    )


def _collect_features(items: Sequence[Any]) -> FeatureSplit:
    sample_ids: list[str] = []
    time: list[float] = []
    event: list[float] = []
    rows: list[list[float]] = []
    for item in items:
        sample_ids.append(str(item.sample_id))
        time.append(float(item.time.item()))
        event.append(float(item.event.item()))
        clinical = item.clinical.detach().cpu().numpy().astype(float).tolist()
        metabolites = item.metabolites.detach().cpu().numpy().astype(float).tolist()
        rows.append(clinical + metabolites + _graph_feature_row(item))
    return FeatureSplit(
        sample_ids=sample_ids,
        time=np.asarray(time, dtype=float),
        event=np.asarray(event, dtype=float),
        features=np.asarray(rows, dtype=float),
    )


def _graph_feature_row(item: Any) -> list[float]:
    x = item.x.detach().cpu().numpy().astype(float)
    edge = item.edge_attr.view(-1).detach().cpu().numpy().astype(float)
    num_nodes = max(float(x.shape[0]), 1.0)
    possible_directed_edges = max(num_nodes * max(num_nodes - 1.0, 1.0), 1.0)
    return [
        float(x[:, 0].mean()),
        float(x[:, 0].std()),
        float(x[:, 0].max()),
        float(x[:, 1].mean()),
        float(x[:, 1].std()),
        float(x[:, 1].max()),
        float(edge.mean()),
        float(edge.std()),
        float(edge.max()),
        float(edge.min()),
        float(edge.sum() / num_nodes),
        float(edge.size / possible_directed_edges),
    ]


def _standardize_baseline_for_training(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(train_values))
    std = max(float(np.std(train_values)), 1e-6)
    return ((np.asarray(values, dtype=float) - mean) / std).astype(float)


def _zscore_torch(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values - values.mean()) / torch.clamp(values.std(unbiased=False), min=float(eps))


def _standardize_feature_splits(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    means = train_features.mean(axis=0, keepdims=True)
    stds = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
    return (
        (train_features - means) / stds,
        (val_features - means) / stds,
        (test_features - means) / stds,
        {
            "feature_means": means.squeeze(axis=0).astype(float).tolist(),
            "feature_stds": stds.squeeze(axis=0).astype(float).tolist(),
        },
    )


def _build_stack_candidates(
    *,
    source_names: Sequence[str],
    val_standardized: np.ndarray,
    test_standardized: np.ndarray,
    val_time: np.ndarray,
    val_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
) -> list[StackCandidate]:
    source_val_c_indices = [
        concordance_index(val_time, val_event, val_standardized[index])
        for index in range(val_standardized.shape[0])
    ]
    gnn_indices = [index for index, name in enumerate(source_names) if name.startswith("gnn:")]
    expert_indices = [index for index, name in enumerate(source_names) if name.startswith("expert:")]
    all_candidate_indices = list(range(val_standardized.shape[0]))
    weight_specs: list[tuple[str, list[float]]] = []
    weight_specs.append(("reference_standardized", _one_hot(0, len(source_names))))

    for index, name in enumerate(source_names[1:], start=1):
        weight_specs.append((f"single:{name}", _one_hot(index, len(source_names))))

    weight_specs.extend(
        _topk_weight_specs("gnn", gnn_indices, source_val_c_indices, len(source_names), max_k=len(gnn_indices))
    )
    if expert_indices:
        weight_specs.extend(
            _topk_weight_specs(
                "expert",
                expert_indices,
                source_val_c_indices,
                len(source_names),
                max_k=min(6, len(expert_indices)),
            )
        )
    weight_specs.extend(
        _topk_weight_specs(
            "all",
            all_candidate_indices,
            source_val_c_indices,
            len(source_names),
            max_k=min(8, len(source_names)),
        )
    )
    for index, name in enumerate(source_names[1:], start=1):
        for alpha in (0.10, 0.20, 0.35, 0.50):
            weights = [0.0 for _ in source_names]
            weights[0] = 1.0 - alpha
            weights[index] = alpha
            weight_specs.append((f"reference_plus:{name}:alpha{alpha}", weights))

    for temperature in (0.003, 0.005, 0.01, 0.02, 0.05):
        scores = np.asarray(source_val_c_indices, dtype=float)
        scaled = np.exp((scores - scores.max()) / float(temperature))
        weights = (scaled / scaled.sum()).astype(float).tolist()
        weight_specs.append((f"softmax_all_val_t{temperature}", weights))

    return [
        _candidate_from_weights(
            name=name,
            weights=weights,
            val_standardized=val_standardized,
            test_standardized=test_standardized,
            val_time=val_time,
            val_event=val_event,
            test_time=test_time,
            test_event=test_event,
        )
        for name, weights in weight_specs
    ]


def _topk_weight_specs(
    prefix: str,
    indices: Sequence[int],
    source_val_c_indices: Sequence[float],
    num_sources: int,
    max_k: int,
) -> list[tuple[str, list[float]]]:
    if len(indices) < 2:
        return []
    ordered = sorted(indices, key=lambda index: source_val_c_indices[index])
    specs: list[tuple[str, list[float]]] = []
    for count in range(2, min(max_k, len(indices)) + 1):
        weights = [0.0 for _ in range(num_sources)]
        for index in ordered[-count:]:
            weights[index] = 1.0 / count
        specs.append((f"{prefix}_top{count}_val_mean", weights))
    return specs


def _candidate_from_weights(
    *,
    name: str,
    weights: Sequence[float],
    val_standardized: np.ndarray,
    test_standardized: np.ndarray,
    val_time: np.ndarray,
    val_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
) -> StackCandidate:
    val_risk = _apply_weights(val_standardized, weights)
    test_risk = _apply_weights(test_standardized, weights)
    return StackCandidate(
        name=name,
        weights=[float(weight) for weight in weights],
        val_risk=val_risk,
        test_risk=test_risk,
        val_c_index=concordance_index(val_time, val_event, val_risk),
        test_c_index=concordance_index(test_time, test_event, test_risk),
    )


def _active_source_weights(source_names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {
        name: float(weight)
        for name, weight in zip(source_names, weights)
        if abs(float(weight)) > 1e-9
    }


def _one_hot(index: int, size: int) -> list[float]:
    weights = [0.0 for _ in range(size)]
    weights[index] = 1.0
    return weights


def _assert_same_order(predictions: PredictionMatrix, sample_ids: Sequence[str], split_name: str) -> None:
    if list(predictions.sample_ids) != [str(sample_id) for sample_id in sample_ids]:
        raise RuntimeError(f"{split_name} feature rows are not aligned with GNN predictions.")


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
    parser.add_argument("--expert-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--expert-seeds", default="7,21,42,123,2026")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--min-validation-delta", type=float, default=0.0)
    parser.add_argument("--disable-linear", action="store_true")
    parser.add_argument("--disable-mlp", action="store_true")
    parser.add_argument("--disable-standalone-mlp", action="store_true")
    parser.add_argument("--disable-mlp-residual", action="store_true")
    parser.add_argument("--disable-ctm", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/cox_fixed_split_expert_stack/expert_stack_summary.json",
    )
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.expert_seeds.split(",") if value.strip()]
    result = evaluate_expert_stack(
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        split_seed=args.split_seed,
        device_arg=args.device,
        expert_device_arg=args.expert_device,
        expert_seeds=seeds,
        train_epochs=args.epochs,
        patience=args.patience,
        min_validation_delta=args.min_validation_delta,
        include_linear=not args.disable_linear,
        include_mlp=not args.disable_mlp,
        include_standalone_mlp=not args.disable_standalone_mlp,
        include_mlp_residual=not args.disable_mlp_residual,
        include_ctm=not args.disable_ctm,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
