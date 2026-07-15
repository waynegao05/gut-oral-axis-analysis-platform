from __future__ import annotations

import numpy as np
import pytest

from research.survival_auc_v2 import (
    _kaplan_meier_censoring_left_limit,
    cumulative_dynamic_auc,
)


def _auc(risk: np.ndarray) -> float:
    result = cumulative_dynamic_auc(
        train_time=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        train_event=np.ones(6),
        test_time=np.asarray([1.0, 2.0, 5.0, 6.0]),
        test_event=np.ones(4),
        risk=risk,
        horizon=3.0,
    )
    return float(result["auc"])


def test_cumulative_dynamic_auc_is_one_for_perfect_separation() -> None:
    assert _auc(np.asarray([4.0, 3.0, 2.0, 1.0])) == pytest.approx(1.0)


def test_cumulative_dynamic_auc_is_zero_for_reversed_separation() -> None:
    assert _auc(np.asarray([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.0)


def test_cumulative_dynamic_auc_counts_ties_as_half() -> None:
    assert _auc(np.ones(4)) == pytest.approx(0.5)


def test_censored_before_horizon_is_excluded() -> None:
    result = cumulative_dynamic_auc(
        train_time=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        train_event=np.ones(5),
        test_time=np.asarray([1.0, 2.0, 4.0, 5.0]),
        test_event=np.asarray([1, 0, 1, 1]),
        risk=np.asarray([4.0, 3.0, 2.0, 1.0]),
        horizon=3.0,
    )

    assert result["num_cases"] == 1
    assert result["num_controls"] == 2
    assert result["num_excluded_censored"] == 1


def test_censoring_km_uses_left_limit() -> None:
    survival = _kaplan_meier_censoring_left_limit(
        train_time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        train_event=np.asarray([1, 0, 1, 1]),
        query_time=np.asarray([2.0, 3.0]),
    )

    assert survival[0] == pytest.approx(1.0)
    assert survival[1] == pytest.approx(2.0 / 3.0)


def test_horizon_must_be_inside_training_follow_up() -> None:
    with pytest.raises(ValueError, match="below the maximum"):
        cumulative_dynamic_auc(
            train_time=np.asarray([1.0, 2.0]),
            train_event=np.ones(2),
            test_time=np.asarray([1.0, 2.0]),
            test_event=np.ones(2),
            risk=np.asarray([2.0, 1.0]),
            horizon=2.0,
        )
