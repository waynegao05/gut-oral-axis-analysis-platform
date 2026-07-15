from __future__ import annotations

import numpy as np
import pytest

from research.structured_ctm_outer_oof_v2 import (
    _candidate_specs,
    _select_top_indices,
    _standardize_with_selection_stats,
)


def test_candidate_specs_match_saved_baseline_v9_order() -> None:
    specs = _candidate_specs(graph_seeds=[0, 101], baseline_seeds=[11, 23])

    assert [spec.name for spec in specs] == [
        "reference",
        "g0_b11",
        "g0_b23",
        "g101_b11",
        "g101_b23",
    ]
    assert specs[0].is_reference
    assert not specs[-1].is_reference


def test_candidate_specs_require_graph_seed() -> None:
    with pytest.raises(ValueError, match="At least one graph seed"):
        _candidate_specs(graph_seeds=[], baseline_seeds=[11])


def test_standardize_with_selection_stats_is_memberwise() -> None:
    risk_matrix = np.asarray([[2.0, 4.0], [10.0, 14.0]])

    standardized = _standardize_with_selection_stats(
        risk_matrix,
        means=np.asarray([2.0, 12.0]),
        stds=np.asarray([2.0, 2.0]),
    )

    np.testing.assert_allclose(standardized, np.asarray([[0.0, 1.0], [-1.0, 1.0]]))


def test_standardize_with_selection_stats_clamps_zero_std() -> None:
    standardized = _standardize_with_selection_stats(
        np.asarray([[1.0, 1.000001]]),
        means=np.asarray([1.0]),
        stds=np.asarray([0.0]),
    )

    assert np.isfinite(standardized).all()


def test_select_top_indices_returns_descending_top_k() -> None:
    assert _select_top_indices([0.2, 0.5, 0.1], top_k=2).tolist() == [1, 0]
