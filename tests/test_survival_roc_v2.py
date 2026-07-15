from __future__ import annotations

import numpy as np
import pytest

from research.survival_auc_v2 import cumulative_dynamic_auc
from research.survival_roc_v2 import cumulative_dynamic_roc


def _roc(risk: np.ndarray) -> dict:
    return cumulative_dynamic_roc(
        train_time=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        train_event=np.ones(6),
        test_time=np.asarray([1.0, 2.0, 5.0, 6.0]),
        test_event=np.ones(4),
        risk=risk,
        horizon=3.0,
    )


def test_roc_is_perfect_for_perfect_separation() -> None:
    result = _roc(np.asarray([4.0, 3.0, 2.0, 1.0]))

    assert result["auc"] == pytest.approx(1.0)
    assert result["false_positive_rate"][0] == pytest.approx(0.0)
    assert result["true_positive_rate"][0] == pytest.approx(0.0)
    assert result["false_positive_rate"][-1] == pytest.approx(1.0)
    assert result["true_positive_rate"][-1] == pytest.approx(1.0)


def test_roc_is_reversed_for_reversed_separation() -> None:
    result = _roc(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert result["auc"] == pytest.approx(0.0)


def test_weighted_roc_matches_pairwise_dynamic_auc_with_censoring() -> None:
    arguments = {
        "train_time": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "train_event": np.asarray([1, 0, 1, 1, 0, 1]),
        "test_time": np.asarray([1.0, 2.0, 4.0, 5.0, 6.0]),
        "test_event": np.asarray([1, 0, 1, 0, 1]),
        "risk": np.asarray([0.8, 0.7, 0.6, 0.4, 0.2]),
        "horizon": 3.0,
    }

    roc_result = cumulative_dynamic_roc(**arguments)
    auc_result = cumulative_dynamic_auc(**arguments)

    assert roc_result["auc"] == pytest.approx(auc_result["auc"])
    assert roc_result["num_excluded_censored"] == 1
