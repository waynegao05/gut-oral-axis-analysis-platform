from __future__ import annotations

import numpy as np
import pytest
import torch

from research.oof_multimodal_safety_residual_v2 import (
    BoundedSafetyResidual,
    _blend_risk,
    _survival_strata,
    select_oof_alpha,
)


def test_bounded_safety_residual_starts_as_exact_fallback() -> None:
    model = BoundedSafetyResidual(input_dim=5, max_delta=0.1)
    features = torch.randn(8, 5)
    baseline = torch.linspace(-1.0, 1.0, steps=8)

    risk, delta = model(features, baseline)

    torch.testing.assert_close(risk, baseline)
    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_blend_risk_respects_alpha_endpoints() -> None:
    baseline = np.asarray([1.0, 2.0, 3.0])
    residual = np.asarray([3.0, 2.0, 1.0])

    np.testing.assert_allclose(_blend_risk(baseline, residual, 0.0), baseline)
    np.testing.assert_allclose(_blend_risk(baseline, residual, 1.0), residual)
    np.testing.assert_allclose(_blend_risk(baseline, residual, 0.5), [2.0, 2.0, 2.0])


def test_oof_alpha_selection_can_choose_improving_residual() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones(5)
    baseline = np.asarray([5.0, 3.0, 4.0, 2.0, 1.0])
    residual = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])

    alpha, rows = select_oof_alpha(
        baseline_risk=baseline,
        residual_risk=residual,
        time=time,
        event=event,
        alpha_grid=[0.0, 0.5, 1.0],
        min_oof_delta=0.0,
        max_cox_loss_increase=10.0,
    )

    assert alpha in {0.5, 1.0}
    assert len(rows) == 3


def test_oof_alpha_selection_falls_back_when_delta_gate_fails() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4)
    baseline = np.asarray([4.0, 3.0, 2.0, 1.0])

    alpha, _ = select_oof_alpha(
        baseline_risk=baseline,
        residual_risk=baseline,
        time=time,
        event=event,
        alpha_grid=[0.0, 0.5, 1.0],
        min_oof_delta=0.0003,
        max_cox_loss_increase=0.0,
    )

    assert alpha == 0.0


def test_survival_strata_has_one_label_per_sample() -> None:
    time = np.arange(1.0, 21.0)
    event = np.asarray([0, 1] * 10)

    strata = _survival_strata(time, event, num_folds=2)

    assert strata.shape == time.shape
    assert len(np.unique(strata)) >= 2


def test_oof_alpha_requires_zero_fallback() -> None:
    with pytest.raises(ValueError, match="include 0.0"):
        select_oof_alpha(
            baseline_risk=np.asarray([2.0, 1.0]),
            residual_risk=np.asarray([2.0, 1.0]),
            time=np.asarray([1.0, 2.0]),
            event=np.ones(2),
            alpha_grid=[0.5, 1.0],
            min_oof_delta=0.0,
            max_cox_loss_increase=0.0,
        )
