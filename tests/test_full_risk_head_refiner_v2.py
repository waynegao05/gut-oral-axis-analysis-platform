from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.full_risk_head_refiner_v2 import (
    CachedCoxHeadRefiner,
    CachedHeadSplit,
    ValidationCandidate,
    _standardize_members_by_validation,
    _validation_softmax_weights,
    fit_cached_refiner,
    select_validation_candidate,
)


def _cached_split(features: np.ndarray, time: np.ndarray, event: np.ndarray) -> CachedHeadSplit:
    latent = np.column_stack([features, features**2]).astype(np.float32)
    base_risk = (0.15 * latent[:, 0]).astype(float)
    return CachedHeadSplit(
        sample_ids=[f"S{index}" for index in range(len(features))],
        time=time.astype(float),
        event=event.astype(float),
        graph_embedding=features[:, None].astype(np.float32),
        clinical=np.zeros((len(features), 1), dtype=np.float32),
        metabolites=np.zeros((len(features), 1), dtype=np.float32),
        latent=latent,
        base_risk=base_risk,
    )


def test_cached_refiner_head_only_preserves_shapes_and_can_train() -> None:
    features = np.linspace(-1.0, 1.0, num=24, dtype=np.float32)
    time = np.linspace(24.0, 1.0, num=24, dtype=np.float32)
    event = np.ones(24, dtype=np.float32)
    split = _cached_split(features, time, event)
    fusion = nn.Sequential(nn.Linear(3, 2), nn.GELU())
    risk_head = nn.Linear(2, 1)
    with torch.no_grad():
        risk_head.weight[:] = torch.tensor([[0.15, 0.0]])
        risk_head.bias.zero_()

    prediction = fit_cached_refiner(
        mode="head_only",
        fusion_template=fusion,
        risk_head_template=risk_head,
        train_split=split,
        val_split=split,
        test_split=split,
        device=torch.device("cpu"),
        seed=7,
        epochs=8,
        patience=4,
        learning_rate=0.01,
        weight_decay=0.0,
        distillation_weight=0.01,
        parameter_anchor_weight=0.0,
        ranking_weight=0.0,
        ranking_margin=0.0,
        grad_clip_norm=1.0,
    )

    assert prediction.train_risk.shape == (24,)
    assert prediction.val_risk.shape == (24,)
    assert prediction.test_risk.shape == (24,)
    assert prediction.best_epoch >= 0
    assert prediction.history


def test_refiner_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        CachedCoxHeadRefiner(
            mode="bad",
            fusion=nn.Identity(),
            risk_head=nn.Linear(2, 1),
        )


def test_validation_selection_uses_c_index_and_cox_gate() -> None:
    candidates = [
        ValidationCandidate("reference", 0.72, 5.70),
        ValidationCandidate("cindex_only", 0.721, 5.71),
        ValidationCandidate("safe", 0.7206, 5.69),
    ]

    selected = select_validation_candidate(
        candidates,
        reference_name="reference",
        min_validation_delta=0.0003,
        max_validation_cox_loss_increase=0.0,
    )

    assert selected == "safe"


def test_validation_selection_falls_back_to_reference() -> None:
    candidates = [
        ValidationCandidate("reference", 0.72, 5.70),
        ValidationCandidate("weak", 0.7202, 5.69),
    ]

    selected = select_validation_candidate(
        candidates,
        reference_name="reference",
        min_validation_delta=0.0003,
        max_validation_cox_loss_increase=0.0,
    )

    assert selected == "reference"


def test_member_standardization_reuses_validation_statistics() -> None:
    train = np.asarray([[2.0, 4.0], [10.0, 14.0]])
    val = np.asarray([[1.0, 3.0], [6.0, 10.0]])
    test = np.asarray([[5.0], [18.0]])

    train_scaled, val_scaled, test_scaled, scaler = _standardize_members_by_validation(
        train,
        val,
        test,
    )

    np.testing.assert_allclose(val_scaled, [[-1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(train_scaled, [[0.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(test_scaled, [[3.0], [5.0]])
    assert scaler["risk_means"] == pytest.approx([2.0, 8.0])


def test_softmax_weights_favor_best_validation_member() -> None:
    weights = _validation_softmax_weights([0.70, 0.72, 0.71], temperature=0.003)

    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] > weights[2] > weights[0]
