from __future__ import annotations

import pytest
import torch

from ctm_fusion_experiment.utils.bootstrap_risk_ensemble_selection import choose_bootstrap_risk_ensemble


def test_choose_bootstrap_risk_ensemble_selects_stable_ensemble() -> None:
    risk_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.1, 4.1, 3.1, 2.1, 1.1],
        ]
    )
    time = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    event = torch.ones(5)

    result = choose_bootstrap_risk_ensemble(
        risk_matrix=risk_matrix,
        times=time,
        events=event,
        model_names=["reference", "better_a", "better_b"],
        reference_index=0,
        min_validation_delta=0.01,
        min_resample_quantile_delta=0.0,
        resamples=12,
        subsample_fraction=0.8,
        seed=7,
    )

    assert result.candidate_name == "top2_mean"
    assert result.c_index == pytest.approx(1.0)
    selected = next(candidate for candidate in result.candidates if candidate["candidate_name"] == result.candidate_name)
    assert selected["resample_lower_quantile_delta"] >= 0.0
