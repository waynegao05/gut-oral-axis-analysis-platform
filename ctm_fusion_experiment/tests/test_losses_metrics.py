from __future__ import annotations

import math

import pytest
import torch

from ctm_fusion_experiment.utils.losses import (
    cox_partial_likelihood_loss,
    ctm_cox_loss,
    select_stable_ticks,
)
from ctm_fusion_experiment.utils.metrics import concordance_index, summarize_paired_folds


def test_cox_loss_normalizes_by_observed_events() -> None:
    risk = torch.tensor([0.0, 0.0])
    time = torch.tensor([3.0, 2.0])
    event = torch.tensor([1.0, 1.0])

    loss = cox_partial_likelihood_loss(risk, time, event)

    assert loss.item() == pytest.approx(math.log(2.0) / 2.0)


def test_select_stable_ticks_uses_each_samples_smallest_adjacent_change() -> None:
    risk_per_tick = torch.tensor(
        [
            [0.0, 1.0, 1.1],
            [0.0, 0.2, 0.21],
        ]
    )

    stable_ticks = select_stable_ticks(risk_per_tick)

    assert stable_ticks.tolist() == [2, 2]


def test_ctm_cox_loss_combines_best_loss_tick_and_stable_tick() -> None:
    risk_per_tick = torch.tensor(
        [
            [0.1, 0.2, 0.21],
            [0.2, 0.3, 0.31],
            [0.3, 0.1, 0.11],
        ]
    )
    time = torch.tensor([3.0, 2.0, 1.0])
    event = torch.tensor([1.0, 1.0, 1.0])

    result = ctm_cox_loss(risk_per_tick, time, event)

    assert result.loss.ndim == 0
    assert 0 <= result.best_loss_tick < 3
    assert result.stable_tick == 2
    assert result.loss.item() == pytest.approx(
        0.5
        * (
            result.losses_per_tick[result.best_loss_tick].item()
            + result.losses_per_tick[result.stable_tick].item()
        )
    )


def test_concordance_index_rewards_higher_risk_for_earlier_events() -> None:
    time = [1.0, 2.0, 3.0]
    event = [1.0, 1.0, 0.0]
    risk = [3.0, 2.0, 1.0]

    assert concordance_index(time, event, risk) == pytest.approx(1.0)


def test_summarize_paired_folds_reports_positive_delta() -> None:
    summary = summarize_paired_folds(
        baseline=[0.60, 0.62, 0.61],
        ctm=[0.63, 0.64, 0.65],
    )

    assert summary["mean_delta"] == pytest.approx(0.03)
    assert summary["paired_t_test"]["num_pairs"] == 3
    assert 0.0 <= summary["paired_t_test"]["p_value"] <= 1.0


def test_summarize_paired_folds_handles_identical_scores_without_nan() -> None:
    summary = summarize_paired_folds(
        baseline=[0.60, 0.62, 0.61],
        ctm=[0.60, 0.62, 0.61],
    )

    assert summary["mean_delta"] == pytest.approx(0.0)
    assert summary["paired_t_test"]["statistic"] == pytest.approx(0.0)
    assert summary["paired_t_test"]["p_value"] == pytest.approx(1.0)
