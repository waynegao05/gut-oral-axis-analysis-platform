from __future__ import annotations

import numpy as np
import pytest

from research.structured_feature_export_v2 import (
    _apply_member_reference_scaler,
    _member_reference_scaler,
    _parse_splits,
    _risk_disagreement_matrix,
    _select_topk_by_validation,
    _summarize_member_array,
)


def test_select_topk_by_validation_uses_descending_cindex() -> None:
    assert _select_topk_by_validation([0.7, 0.9, 0.8], top_k=2).tolist() == [1, 2]


def test_member_reference_scaler_reuses_validation_statistics() -> None:
    val = np.asarray([[1.0, 3.0], [10.0, 14.0]])
    test = np.asarray([[5.0], [18.0]])

    means, stds = _member_reference_scaler(val)
    scaled = _apply_member_reference_scaler(test, means, stds)

    assert means.tolist() == pytest.approx([2.0, 12.0])
    assert stds.tolist() == pytest.approx([1.0, 2.0])
    np.testing.assert_allclose(scaled, [[3.0], [3.0]])


def test_summarize_member_array_returns_mean_and_std_per_sample() -> None:
    values = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 6.0], [7.0, 8.0]],
        ]
    )

    result = _summarize_member_array(values)

    np.testing.assert_allclose(result["mean"], [[2.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(result["std"], [[1.0, 2.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        _summarize_member_array(np.asarray([[1.0, 2.0]]))


def test_risk_disagreement_matrix_matches_expected_columns() -> None:
    risk = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 7.0],
        ]
    )

    result = _risk_disagreement_matrix(risk, top_indices=[1, 2])

    assert result.shape == (3, 6)
    assert result[:, 0] == pytest.approx(risk.std(axis=0))
    assert result[:, 1] == pytest.approx(risk.max(axis=0) - risk.min(axis=0))
    assert result[:, 2] == pytest.approx(risk[[1, 2]].std(axis=0))
    assert result[:, 3] == pytest.approx(risk[[1, 2]].max(axis=0) - risk[[1, 2]].min(axis=0))


def test_parse_splits_validates_known_splits() -> None:
    assert _parse_splits("train, val,test") == ["train", "val", "test"]
    with pytest.raises(ValueError):
        _parse_splits("train,holdout")
