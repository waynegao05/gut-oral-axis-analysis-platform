from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.topology_v7_ac_icam_transfer_v10.mbr_distillation import (
    fit_distilled_risk,
)
from experiments.topology_v7_ac_icam_transfer_v10.development_transfer import (
    _add_source_prior,
    _select_residual_alpha,
)
from experiments.topology_v7_ac_icam_transfer_v10.source_audit import (
    RidgeCoxModel,
    _cox_loss_gradient,
    fit_ridge_cox,
)


def test_cox_gradient_matches_finite_difference() -> None:
    values = np.asarray(
        [
            [-1.0, 0.5],
            [-0.2, -0.3],
            [0.4, 1.2],
            [1.1, -0.4],
            [1.6, 0.8],
        ],
        dtype=float,
    )
    time = np.asarray([10.0, 8.0, 8.0, 5.0, 3.0])
    event = np.asarray([1.0, 1.0, 0.0, 1.0, 1.0])
    beta = np.asarray([0.2, -0.1])
    loss, gradient = _cox_loss_gradient(
        beta, values, time, event, l2=0.1
    )
    assert np.isfinite(loss)
    numerical = np.zeros_like(beta)
    epsilon = 1e-6
    for index in range(len(beta)):
        step = np.zeros_like(beta)
        step[index] = epsilon
        plus = _cox_loss_gradient(
            beta + step, values, time, event, l2=0.1
        )[0]
        minus = _cox_loss_gradient(
            beta - step, values, time, event, l2=0.1
        )[0]
        numerical[index] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_ridge_cox_learns_risk_direction() -> None:
    rng = np.random.default_rng(42)
    feature = rng.normal(size=80)
    event_time = np.exp(4.0 - 0.8 * feature + rng.normal(0.0, 0.1, 80))
    censor_time = rng.uniform(20.0, 120.0, size=80)
    frame = pd.DataFrame(
        {
            "feature": feature,
            "time": np.minimum(event_time, censor_time),
            "event": (event_time <= censor_time).astype(float),
        }
    )
    model = fit_ridge_cox(frame, ["feature"], l2=0.01)
    assert model.optimization_success
    assert model.coefficients[0] > 0.0


def test_distilled_risk_handles_missing_validation_values() -> None:
    train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [1.0, 0.0, 1.0, 0.0],
            "mbr_score": [0.0, 1.0, 2.0, 3.0],
        }
    )
    model = fit_distilled_risk(train, ["a", "b"], alpha=0.1)
    test = pd.DataFrame({"a": [np.nan, 2.5], "b": [1.0, np.nan]})
    predictions = model.predict(test)
    assert np.isfinite(predictions).all()


def test_source_prior_maps_normal_colon_features_to_stool() -> None:
    taxa = [
        "fusobacterium",
        "porphyromonas",
        "prevotella",
        "streptococcus",
        "lactobacillus",
    ]
    frame = pd.DataFrame(
        {
            **{f"stool_clr__{taxon}": [0.1, -0.2] for taxon in taxa},
            **{f"stool_raw__{taxon}": [0.01, 0.02] for taxon in taxa},
        }
    )
    model = RidgeCoxModel(
        feature_columns=tuple(
            [f"normal_clr__{taxon}" for taxon in taxa]
            + ["normal_log_panel_load"]
        ),
        mean=(0.0,) * 6,
        scale=(1.0,) * 6,
        coefficients=(0.1,) * 6,
        l2=0.1,
        optimization_success=True,
    )
    result = _add_source_prior(frame, model)
    assert np.isfinite(result["ac_icam_pfs_prior"]).all()


def test_residual_alpha_can_fall_back_to_baseline() -> None:
    validation = pd.DataFrame(
        {
            "time": [4.0, 3.0, 2.0, 1.0],
            "event": [1.0, 1.0, 1.0, 1.0],
            "ac_icam_pfs_prior": [-1.0, -0.5, 0.5, 1.0],
        }
    )
    test = validation.copy()
    result = _select_residual_alpha(
        validation=validation,
        test=test,
        baseline_validation=np.asarray([0.0, 1.0, 2.0, 3.0]),
        baseline_test=np.asarray([0.0, 1.0, 2.0, 3.0]),
    )
    assert result["selected_alpha"] == 0.0
