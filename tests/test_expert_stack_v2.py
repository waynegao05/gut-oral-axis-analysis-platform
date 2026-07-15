from __future__ import annotations

import numpy as np
import pytest
import torch

from research.expert_stack_v2 import (
    MlpResidualRiskExpert,
    _active_source_weights,
    _build_stack_candidates,
    _standardize_feature_splits,
    _topk_weight_specs,
)


def test_standardize_feature_splits_uses_train_statistics() -> None:
    train = np.asarray([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    val = np.asarray([[7.0, 22.0]])
    test = np.asarray([[9.0, 26.0]])

    train_scaled, val_scaled, test_scaled, scaler = _standardize_feature_splits(train, val, test)

    assert train_scaled.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert scaler["feature_means"] == pytest.approx([3.0, 14.0])
    assert scaler["feature_stds"] == pytest.approx([np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)])
    assert val_scaled[0, 0] == pytest.approx((7.0 - 3.0) / np.sqrt(8.0 / 3.0))
    assert test_scaled[0, 1] == pytest.approx((26.0 - 14.0) / np.sqrt(32.0 / 3.0))


def test_topk_weight_specs_uses_best_validation_sources() -> None:
    specs = _topk_weight_specs(
        "all",
        indices=[0, 1, 2, 3],
        source_val_c_indices=[0.7, 0.9, 0.8, 0.6],
        num_sources=4,
        max_k=3,
    )

    by_name = {name: weights for name, weights in specs}
    assert by_name["all_top2_val_mean"] == pytest.approx([0.0, 0.5, 0.5, 0.0])
    assert by_name["all_top3_val_mean"] == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])


def test_build_stack_candidates_includes_reference_plus_source() -> None:
    val = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.2, 3.2, 2.2, 1.2],
        ]
    )
    test = val.copy()
    time = np.asarray([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4)

    candidates = _build_stack_candidates(
        source_names=["reference_raw_mean", "gnn:a", "expert:b"],
        val_standardized=val,
        test_standardized=test,
        val_time=time,
        val_event=event,
        test_time=time,
        test_event=event,
    )

    names = {candidate.name for candidate in candidates}
    assert "reference_plus:expert:b:alpha0.35" in names
    assert "all_top2_val_mean" in names
    selected = next(candidate for candidate in candidates if candidate.name == "all_top2_val_mean")
    assert selected.weights == pytest.approx([0.0, 0.5, 0.5])


def test_active_source_weights_drops_zeroes() -> None:
    assert _active_source_weights(["a", "b", "c"], [0.0, 0.25, 0.75]) == {"b": 0.25, "c": 0.75}


def test_mlp_residual_expert_bounds_delta() -> None:
    model = MlpResidualRiskExpert(input_dim=3, max_delta=0.2)
    features = np.asarray([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    baseline = np.asarray([0.1, -0.2], dtype=np.float32)

    risk, delta = model(
        torch.tensor(features),
        torch.tensor(baseline),
    )

    assert risk.shape == delta.shape
    assert float(delta.detach().abs().max()) <= 0.2
