from __future__ import annotations

import numpy as np
import pytest

from research.ensemble_stack_v2 import (
    _apply_weights,
    _build_candidates,
    _candidate_allowed,
    _cohort_cox_loss,
    _fit_cox_risk_scale,
    _standardize_by_validation,
)


def test_standardize_by_validation_reuses_validation_scaler() -> None:
    val = np.asarray([[1.0, 2.0, 3.0], [10.0, 12.0, 14.0]])
    test = np.asarray([[4.0, 5.0], [16.0, 18.0]])

    val_scaled, test_scaled, scaler = _standardize_by_validation(val, test)

    assert scaler["risk_means"] == pytest.approx([2.0, 12.0])
    assert scaler["risk_stds"] == pytest.approx([np.sqrt(2.0 / 3.0), np.sqrt(8.0 / 3.0)])
    assert val_scaled.mean(axis=1) == pytest.approx([0.0, 0.0])
    assert test_scaled[0, 0] == pytest.approx((4.0 - 2.0) / np.sqrt(2.0 / 3.0))


def test_build_candidates_includes_validation_top_k_and_softmax() -> None:
    val = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.1, 3.1, 2.1, 1.1],
        ]
    )
    test = val.copy()
    time = np.asarray([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4)

    candidates = _build_candidates(
        val_standardized=val,
        test_standardized=test,
        val_time=time,
        val_event=event,
        test_time=time,
        test_event=event,
        member_val_c_indices=[0.0, 1.0, 1.0],
    )

    names = {candidate.name for candidate in candidates}
    assert "top2_val_mean" in names
    assert "softmax_val_t0.003" in names
    top2 = next(candidate for candidate in candidates if candidate.name == "top2_val_mean")
    assert top2.weights == pytest.approx([0.0, 0.5, 0.5])
    assert top2.val_c_index == pytest.approx(1.0)


def test_apply_weights_validates_shape() -> None:
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    assert _apply_weights(matrix, [0.25, 0.75]).tolist() == pytest.approx([2.5, 3.5])
    with pytest.raises(ValueError):
        _apply_weights(matrix, [1.0])


def test_candidate_policy_filters_expected_candidate_groups() -> None:
    assert _candidate_allowed("top3_val_mean", "topk_mean_or_reference")
    assert not _candidate_allowed("softmax_val_t0.003", "topk_mean_or_reference")
    assert _candidate_allowed("softmax_val_t0.003", "softmax_or_reference")
    assert _candidate_allowed("single:0", "single_or_reference")
    assert _candidate_allowed("single:0", "all")
    with pytest.raises(ValueError):
        _candidate_allowed("top3_val_mean", "unknown")


def test_fit_cox_risk_scale_reduces_validation_loss_without_changing_order() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones(5)
    risk = np.asarray([0.4, 0.3, 0.2, 0.1, 0.0])

    calibration = _fit_cox_risk_scale(risk, time, event)
    calibrated_risk = risk * float(calibration["scale"])

    assert calibration["scale"] > 0.0
    assert calibration["preserves_ranking"] is True
    assert _cohort_cox_loss(calibrated_risk, time, event) <= _cohort_cox_loss(risk, time, event)
    assert np.argsort(calibrated_risk).tolist() == np.argsort(risk).tolist()


def test_fit_cox_risk_scale_validates_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        _fit_cox_risk_scale(np.ones(2), np.ones(3), np.ones(2))
