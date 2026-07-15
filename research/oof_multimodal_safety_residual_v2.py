from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from research.ensemble_stack_v2 import _cohort_cox_loss, _fit_cox_risk_scale
from research.expert_stack_v2 import _load_feature_splits
from research.full_risk_head_refiner_v2 import ValidationCandidate, select_validation_candidate
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import resolve_device


@dataclass(frozen=True)
class ResidualSplit:
    sample_ids: list[str]
    time: np.ndarray
    event: np.ndarray
    features: np.ndarray
    baseline_risk: np.ndarray


@dataclass(frozen=True)
class FittedResidualModel:
    state_dict: dict[str, torch.Tensor]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    baseline_mean: float
    baseline_std: float
    best_epoch: int
    history: list[dict[str, float | int]]


class BoundedSafetyResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 24,
        dropout: float = 0.30,
        max_delta: float = 0.10,
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
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        baseline_risk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.max_delta * torch.tanh(self.network(features).squeeze(-1))
        return baseline_risk + delta, delta


def run_oof_multimodal_safety_residual(
    *,
    config_path: str,
    baseline_predictions_path: str | Path,
    split_seed: int,
    output_path: str | Path,
    device_arg: str = "cpu",
    num_folds: int = 5,
    final_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    max_epochs: int = 140,
    patience: int = 20,
    inner_validation_ratio: float = 0.15,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    hidden_dim: int = 24,
    dropout: float = 0.30,
    max_delta: float = 0.10,
    distillation_weight: float = 0.10,
    delta_l2_weight: float = 0.08,
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    min_oof_delta: float = 0.0003,
    min_validation_delta: float = 0.0003,
    max_cox_loss_increase: float = 0.0,
) -> dict[str, Any]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if not final_seeds:
        raise ValueError("At least one final seed is required.")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    prediction_path = Path(baseline_predictions_path)
    prediction_bundle = _load_prediction_bundle(prediction_path)
    feature_names, train_split, val_split, test_split = _build_residual_splits(
        config=config,
        split_seed=split_seed,
        prediction_bundle=prediction_bundle,
    )
    device = resolve_device(device_arg)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output.parent / f"{output.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    strata = _survival_strata(train_split.time, train_split.event, num_folds=num_folds)
    fold_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
    oof_risk = np.full(len(train_split.sample_ids), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    best_epochs: list[int] = []

    for fold_index, (outer_train_indices, holdout_indices) in enumerate(
        fold_splitter.split(np.zeros(len(strata)), strata),
        start=1,
    ):
        inner_train_indices, inner_val_indices = _inner_split_indices(
            outer_train_indices,
            strata,
            validation_ratio=inner_validation_ratio,
            seed=split_seed + fold_index,
        )
        fitted = fit_residual_model(
            train_split=_subset_split(train_split, inner_train_indices),
            validation_split=_subset_split(train_split, inner_val_indices),
            device=device,
            seed=int(final_seeds[(fold_index - 1) % len(final_seeds)]) + 1000,
            max_epochs=max_epochs,
            fixed_epochs=None,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_delta=max_delta,
            distillation_weight=distillation_weight,
            delta_l2_weight=delta_l2_weight,
        )
        holdout_split = _subset_split(train_split, holdout_indices)
        holdout_risk = predict_residual_model(
            fitted,
            holdout_split,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_delta=max_delta,
        )
        oof_risk[holdout_indices] = holdout_risk
        best_epochs.append(int(fitted.best_epoch))
        history_path = artifact_dir / f"oof_fold{fold_index}_history.json"
        history_path.write_text(json.dumps(fitted.history, indent=2), encoding="utf-8")
        fold_rows.append(
            {
                "fold": int(fold_index),
                "num_outer_train": int(len(outer_train_indices)),
                "num_inner_train": int(len(inner_train_indices)),
                "num_inner_validation": int(len(inner_val_indices)),
                "num_holdout": int(len(holdout_indices)),
                "best_epoch": int(fitted.best_epoch),
                "history_path": str(history_path.as_posix()),
                "holdout_baseline_c_index": concordance_index(
                    holdout_split.time,
                    holdout_split.event,
                    holdout_split.baseline_risk,
                ),
                "holdout_residual_c_index": concordance_index(
                    holdout_split.time,
                    holdout_split.event,
                    holdout_risk,
                ),
            }
        )
    if not np.isfinite(oof_risk).all():
        raise RuntimeError("OOF residual predictions were not populated for every training sample.")

    selected_alpha, oof_candidates = select_oof_alpha(
        baseline_risk=train_split.baseline_risk,
        residual_risk=oof_risk,
        time=train_split.time,
        event=train_split.event,
        alpha_grid=alpha_grid,
        min_oof_delta=min_oof_delta,
        max_cox_loss_increase=max_cox_loss_increase,
    )

    final_val_predictions: list[np.ndarray] = []
    final_test_predictions: list[np.ndarray] = []
    final_artifacts: list[dict[str, Any]] = []
    for model_index, seed in enumerate(final_seeds):
        fixed_epochs = int(best_epochs[model_index % len(best_epochs)])
        fitted = fit_residual_model(
            train_split=train_split,
            validation_split=None,
            device=device,
            seed=int(seed) + 2000,
            max_epochs=max_epochs,
            fixed_epochs=fixed_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_delta=max_delta,
            distillation_weight=distillation_weight,
            delta_l2_weight=delta_l2_weight,
        )
        val_risk = predict_residual_model(
            fitted,
            val_split,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_delta=max_delta,
        )
        test_risk = predict_residual_model(
            fitted,
            test_split,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_delta=max_delta,
        )
        final_val_predictions.append(val_risk)
        final_test_predictions.append(test_risk)
        artifact_path = artifact_dir / f"final_seed{int(seed)}.pt"
        torch.save(
            {
                "format_version": 1,
                "state_dict": fitted.state_dict,
                "feature_names": feature_names,
                "feature_means": fitted.feature_means,
                "feature_stds": fitted.feature_stds,
                "baseline_mean": fitted.baseline_mean,
                "baseline_std": fitted.baseline_std,
                "best_epoch": fitted.best_epoch,
                "seed": int(seed),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "max_delta": float(max_delta),
            },
            artifact_path,
        )
        final_artifacts.append(
            {
                "seed": int(seed),
                "fixed_epochs": int(fixed_epochs),
                "path": str(artifact_path.as_posix()),
            }
        )

    mean_val_residual = np.mean(np.vstack(final_val_predictions), axis=0)
    mean_test_residual = np.mean(np.vstack(final_test_predictions), axis=0)
    candidate_val_risk = _blend_risk(val_split.baseline_risk, mean_val_residual, selected_alpha)
    candidate_test_risk = _blend_risk(test_split.baseline_risk, mean_test_residual, selected_alpha)

    reference_val_calibration = _fit_cox_risk_scale(
        val_split.baseline_risk,
        val_split.time,
        val_split.event,
    )
    candidate_val_calibration = _fit_cox_risk_scale(
        candidate_val_risk,
        val_split.time,
        val_split.event,
    )
    validation_candidates = [
        ValidationCandidate(
            "reference",
            concordance_index(val_split.time, val_split.event, val_split.baseline_risk),
            float(reference_val_calibration["calibrated_validation_cox_loss"]),
        ),
        ValidationCandidate(
            "oof_safety_residual",
            concordance_index(val_split.time, val_split.event, candidate_val_risk),
            float(candidate_val_calibration["calibrated_validation_cox_loss"]),
        ),
    ]
    selected_name = select_validation_candidate(
        validation_candidates,
        reference_name="reference",
        min_validation_delta=min_validation_delta,
        max_validation_cox_loss_increase=max_cox_loss_increase,
    )
    selected_val_risk = val_split.baseline_risk if selected_name == "reference" else candidate_val_risk
    selected_test_risk = test_split.baseline_risk if selected_name == "reference" else candidate_test_risk
    selected_val_calibration = (
        reference_val_calibration if selected_name == "reference" else candidate_val_calibration
    )

    reference_test_c_index = concordance_index(
        test_split.time,
        test_split.event,
        test_split.baseline_risk,
    )
    selected_test_c_index = concordance_index(test_split.time, test_split.event, selected_test_risk)
    reference_calibrated_test_loss = _cohort_cox_loss(
        test_split.baseline_risk * float(reference_val_calibration["scale"]),
        test_split.time,
        test_split.event,
    )
    selected_calibrated_test_loss = _cohort_cox_loss(
        selected_test_risk * float(selected_val_calibration["scale"]),
        test_split.time,
        test_split.event,
    )

    predictions_path = output.with_name(f"{output.stem}_predictions.npz")
    np.savez_compressed(
        predictions_path,
        train_sample_ids=np.asarray(train_split.sample_ids),
        train_time=train_split.time,
        train_event=train_split.event,
        train_baseline_risk=train_split.baseline_risk,
        train_oof_residual_risk=oof_risk,
        val_sample_ids=np.asarray(val_split.sample_ids),
        val_time=val_split.time,
        val_event=val_split.event,
        val_baseline_risk=val_split.baseline_risk,
        val_candidate_risk=candidate_val_risk,
        val_selected_risk=selected_val_risk,
        test_sample_ids=np.asarray(test_split.sample_ids),
        test_time=test_split.time,
        test_event=test_split.event,
        test_baseline_risk=test_split.baseline_risk,
        test_candidate_risk=candidate_test_risk,
        test_selected_risk=selected_test_risk,
    )

    result = {
        "config_path": str(Path(config_path).as_posix()),
        "baseline_predictions_path": str(prediction_path.as_posix()),
        "split_seed": int(split_seed),
        "device": str(device),
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "num_folds": int(num_folds),
        "folds": fold_rows,
        "oof": {
            "alpha_candidates": oof_candidates,
            "selected_alpha": float(selected_alpha),
            "selection_uses_outer_validation": False,
            "baseline_c_index": concordance_index(
                train_split.time,
                train_split.event,
                train_split.baseline_risk,
            ),
            "selected_c_index": concordance_index(
                train_split.time,
                train_split.event,
                _blend_risk(train_split.baseline_risk, oof_risk, selected_alpha),
            ),
        },
        "final_models": final_artifacts,
        "validation": {
            "reference_c_index": validation_candidates[0].val_c_index,
            "candidate_c_index": validation_candidates[1].val_c_index,
            "candidate_delta": validation_candidates[1].val_c_index - validation_candidates[0].val_c_index,
            "reference_cox_scale_calibration": reference_val_calibration,
            "candidate_cox_scale_calibration": candidate_val_calibration,
            "selected_name": selected_name,
            "min_validation_delta": float(min_validation_delta),
            "max_cox_loss_increase": float(max_cox_loss_increase),
        },
        "test": {
            "reference_c_index": reference_test_c_index,
            "selected_c_index": selected_test_c_index,
            "selected_delta": selected_test_c_index - reference_test_c_index,
            "reference_calibrated_cox_loss": reference_calibrated_test_loss,
            "selected_calibrated_cox_loss": selected_calibrated_test_loss,
            "calibrated_cox_loss_delta": selected_calibrated_test_loss
            - reference_calibrated_test_loss,
            "used_for_selection": False,
        },
        "predictions_path": str(predictions_path.as_posix()),
        "limitation": (
            "The residual learner is cross-fitted, but the frozen GNN and full-risk-head training predictions are "
            "in-sample. A fully nested GNN OOF experiment would require retraining every base GNN inside each fold."
        ),
        "interpretation": (
            "The residual magnitude is selected from cross-fitted training predictions. Final residual models are "
            "then refit on all training samples for fold-derived epoch counts, averaged, and accepted only if the "
            "outer validation c-index and calibrated Cox-loss gates both pass."
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def fit_residual_model(
    *,
    train_split: ResidualSplit,
    validation_split: ResidualSplit | None,
    device: torch.device,
    seed: int,
    max_epochs: int,
    fixed_epochs: int | None,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    max_delta: float,
    distillation_weight: float,
    delta_l2_weight: float,
) -> FittedResidualModel:
    _set_seed(seed)
    feature_means = train_split.features.mean(axis=0)
    feature_stds = np.maximum(train_split.features.std(axis=0), 1e-6)
    baseline_mean = float(train_split.baseline_risk.mean())
    baseline_std = max(float(train_split.baseline_risk.std()), 1e-6)
    train_x = torch.as_tensor(
        (train_split.features - feature_means) / feature_stds,
        dtype=torch.float32,
        device=device,
    )
    train_base = torch.as_tensor(
        (train_split.baseline_risk - baseline_mean) / baseline_std,
        dtype=torch.float32,
        device=device,
    )
    train_time = torch.as_tensor(train_split.time, dtype=torch.float32, device=device)
    train_event = torch.as_tensor(train_split.event, dtype=torch.float32, device=device)
    model = BoundedSafetyResidual(
        train_x.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_delta=max_delta,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_state = _cpu_state_dict(model)
    best_epoch = 0
    best_val_c_index = (
        concordance_index(validation_split.time, validation_split.event, validation_split.baseline_risk)
        if validation_split is not None
        else float("-inf")
    )
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    epochs_to_run = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)

    for epoch in range(1, epochs_to_run + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, delta = model(train_x, train_base)
        cox_loss = cox_ph_loss(risk, train_time, train_event)
        distillation = torch.mean((_zscore(risk) - _zscore(train_base.detach())) ** 2)
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            cox_loss
            + float(distillation_weight) * distillation
            + float(delta_l2_weight) * delta_l2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "train_total_loss": float(loss.detach().item()),
            "train_cox_loss": float(cox_loss.detach().item()),
            "train_distillation_loss": float(distillation.detach().item()),
            "train_delta_l2": float(delta_l2.detach().item()),
        }
        if validation_split is not None:
            val_risk = _predict_with_scaler(
                model,
                validation_split,
                feature_means=feature_means,
                feature_stds=feature_stds,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                device=device,
            )
            val_c_index = concordance_index(
                validation_split.time,
                validation_split.event,
                val_risk,
            )
            row["validation_c_index"] = float(val_c_index)
            if val_c_index > best_val_c_index + 0.00025:
                best_val_c_index = float(val_c_index)
                best_epoch = int(epoch)
                best_state = _cpu_state_dict(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_epoch = int(epoch)
            best_state = _cpu_state_dict(model)
        history.append(row)
        if validation_split is not None and stale_epochs >= int(patience):
            break

    return FittedResidualModel(
        state_dict=best_state,
        feature_means=feature_means.astype(float),
        feature_stds=feature_stds.astype(float),
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        best_epoch=best_epoch,
        history=history,
    )


def predict_residual_model(
    fitted: FittedResidualModel,
    split: ResidualSplit,
    *,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    max_delta: float,
) -> np.ndarray:
    model = BoundedSafetyResidual(
        split.features.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_delta=max_delta,
    ).to(device)
    model.load_state_dict(fitted.state_dict)
    return _predict_with_scaler(
        model,
        split,
        feature_means=fitted.feature_means,
        feature_stds=fitted.feature_stds,
        baseline_mean=fitted.baseline_mean,
        baseline_std=fitted.baseline_std,
        device=device,
    )


def select_oof_alpha(
    *,
    baseline_risk: np.ndarray,
    residual_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    alpha_grid: Sequence[float],
    min_oof_delta: float,
    max_cox_loss_increase: float,
) -> tuple[float, list[dict[str, Any]]]:
    alphas = sorted(set(float(value) for value in alpha_grid))
    if 0.0 not in alphas:
        raise ValueError("alpha_grid must include 0.0 for safe fallback.")
    if any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("alpha values must be between 0 and 1.")
    rows: list[dict[str, Any]] = []
    candidates: list[ValidationCandidate] = []
    for alpha in alphas:
        risk = _blend_risk(baseline_risk, residual_risk, alpha)
        calibration = _fit_cox_risk_scale(risk, time, event)
        name = f"alpha_{alpha:g}"
        c_index = concordance_index(time, event, risk)
        rows.append(
            {
                "alpha": alpha,
                "c_index": c_index,
                "cox_scale_calibration": calibration,
            }
        )
        candidates.append(
            ValidationCandidate(
                name=name,
                val_c_index=c_index,
                calibrated_val_cox_loss=float(calibration["calibrated_validation_cox_loss"]),
            )
        )
    selected_name = select_validation_candidate(
        candidates,
        reference_name="alpha_0",
        min_validation_delta=min_oof_delta,
        max_validation_cox_loss_increase=max_cox_loss_increase,
    )
    return float(selected_name.removeprefix("alpha_")), rows


def _load_prediction_bundle(path: Path) -> dict[str, dict[str, Any]]:
    required = []
    for split_name in ("train", "val", "test"):
        required.extend(
            [
                f"{split_name}_sample_ids",
                f"{split_name}_time",
                f"{split_name}_event",
                f"{split_name}_reference_risk",
                f"{split_name}_selected_risk",
                f"{split_name}_selected_member_risk_matrix",
            ]
        )
    bundle: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as values:
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"Baseline prediction bundle is missing fields: {missing}")
        for split_name in ("train", "val", "test"):
            bundle[split_name] = {
                "sample_ids": [str(value) for value in values[f"{split_name}_sample_ids"].tolist()],
                "time": np.asarray(values[f"{split_name}_time"], dtype=float),
                "event": np.asarray(values[f"{split_name}_event"], dtype=float),
                "reference_risk": np.asarray(values[f"{split_name}_reference_risk"], dtype=float),
                "selected_risk": np.asarray(values[f"{split_name}_selected_risk"], dtype=float),
                "selected_member_risk_matrix": np.asarray(
                    values[f"{split_name}_selected_member_risk_matrix"],
                    dtype=float,
                ),
            }
    return bundle


def _build_residual_splits(
    *,
    config: dict[str, Any],
    split_seed: int,
    prediction_bundle: dict[str, dict[str, Any]],
) -> tuple[list[str], ResidualSplit, ResidualSplit, ResidualSplit]:
    base_names, train_features, val_features, test_features = _load_feature_splits(config, split_seed)
    feature_splits = {
        "train": train_features,
        "val": val_features,
        "test": test_features,
    }
    num_members = prediction_bundle["train"]["selected_member_risk_matrix"].shape[0]
    extra_names = [
        "risk:reference",
        "risk:selected_full_risk_head",
        "risk:full_risk_head_delta",
    ]
    extra_names.extend(f"risk:selected_member_{index}" for index in range(num_members))
    extra_names.extend(["risk:selected_member_std", "risk:selected_member_range"])
    result: dict[str, ResidualSplit] = {}
    for split_name, feature_split in feature_splits.items():
        prediction = prediction_bundle[split_name]
        if feature_split.sample_ids != prediction["sample_ids"]:
            raise RuntimeError(f"Feature and prediction sample IDs are not aligned for split {split_name}.")
        if not np.array_equal(feature_split.time, prediction["time"]) or not np.array_equal(
            feature_split.event,
            prediction["event"],
        ):
            raise RuntimeError(f"Feature and prediction outcomes are not aligned for split {split_name}.")
        member_matrix = prediction["selected_member_risk_matrix"]
        if member_matrix.shape[0] != num_members or member_matrix.shape[1] != len(feature_split.sample_ids):
            raise ValueError(f"Invalid selected member matrix shape for split {split_name}.")
        extra_features = np.column_stack(
            [
                prediction["reference_risk"],
                prediction["selected_risk"],
                prediction["selected_risk"] - prediction["reference_risk"],
                member_matrix.T,
                member_matrix.std(axis=0),
                member_matrix.max(axis=0) - member_matrix.min(axis=0),
            ]
        )
        result[split_name] = ResidualSplit(
            sample_ids=feature_split.sample_ids,
            time=feature_split.time,
            event=feature_split.event,
            features=np.concatenate([feature_split.features, extra_features], axis=1),
            baseline_risk=prediction["selected_risk"],
        )
    return base_names + extra_names, result["train"], result["val"], result["test"]


def _survival_strata(time: np.ndarray, event: np.ndarray, *, num_folds: int) -> np.ndarray:
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=int)
    quantiles = np.quantile(time_values, [0.25, 0.50, 0.75])
    time_bins = np.digitize(time_values, np.unique(quantiles), right=True)
    combined = event_values * 10 + time_bins
    _, counts = np.unique(combined, return_counts=True)
    if counts.min(initial=num_folds) < num_folds:
        return event_values.astype(str)
    return combined.astype(str)


def _inner_split_indices(
    outer_train_indices: np.ndarray,
    strata: np.ndarray,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer_strata = strata[outer_train_indices]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(validation_ratio),
        random_state=int(seed),
    )
    inner_train_relative, inner_val_relative = next(
        splitter.split(np.zeros(len(outer_train_indices)), outer_strata)
    )
    return outer_train_indices[inner_train_relative], outer_train_indices[inner_val_relative]


def _subset_split(split: ResidualSplit, indices: np.ndarray) -> ResidualSplit:
    return ResidualSplit(
        sample_ids=[split.sample_ids[int(index)] for index in indices],
        time=split.time[indices],
        event=split.event[indices],
        features=split.features[indices],
        baseline_risk=split.baseline_risk[indices],
    )


def _predict_with_scaler(
    model: BoundedSafetyResidual,
    split: ResidualSplit,
    *,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features = torch.as_tensor(
            (split.features - feature_means) / feature_stds,
            dtype=torch.float32,
            device=device,
        )
        baseline = torch.as_tensor(
            (split.baseline_risk - baseline_mean) / baseline_std,
            dtype=torch.float32,
            device=device,
        )
        risk, _ = model(features, baseline)
        return (risk.detach().cpu().numpy().astype(float) * baseline_std) + baseline_mean


def _blend_risk(baseline_risk: np.ndarray, residual_risk: np.ndarray, alpha: float) -> np.ndarray:
    baseline = np.asarray(baseline_risk, dtype=float)
    residual = np.asarray(residual_risk, dtype=float)
    if baseline.shape != residual.shape:
        raise ValueError("baseline_risk and residual_risk must have identical shapes.")
    return baseline + float(alpha) * (residual - baseline)


def _zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values - values.mean()) / torch.clamp(values.std(unbiased=False), min=float(eps))


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--final-seeds", default="7,21,42,123,2026")
    parser.add_argument("--max-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--inner-validation-ratio", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--max-delta", type=float, default=0.10)
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2-weight", type=float, default=0.08)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--min-oof-delta", type=float, default=0.0003)
    parser.add_argument("--min-validation-delta", type=float, default=0.0003)
    parser.add_argument("--max-cox-loss-increase", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/oof_multimodal_safety_residual_v2/summary.json",
    )
    args = parser.parse_args()
    result = run_oof_multimodal_safety_residual(
        config_path=args.config,
        baseline_predictions_path=args.baseline_predictions,
        split_seed=args.split_seed,
        output_path=args.output,
        device_arg=args.device,
        num_folds=args.num_folds,
        final_seeds=_parse_int_list(args.final_seeds),
        max_epochs=args.max_epochs,
        patience=args.patience,
        inner_validation_ratio=args.inner_validation_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_delta=args.max_delta,
        distillation_weight=args.distillation_weight,
        delta_l2_weight=args.delta_l2_weight,
        alpha_grid=_parse_float_list(args.alpha_grid),
        min_oof_delta=args.min_oof_delta,
        min_validation_delta=args.min_validation_delta,
        max_cox_loss_increase=args.max_cox_loss_increase,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
