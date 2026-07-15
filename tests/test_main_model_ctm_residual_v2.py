from __future__ import annotations

import numpy as np
import pytest

from research.main_model_ctm_residual_v2 import _assert_split_order, _parse_floats, _parse_ints


def test_parse_ints_and_floats_ignore_empty_parts() -> None:
    assert _parse_ints("7, 21,,42") == [7, 21, 42]
    assert _parse_floats("0.05, 0.1,,0.2") == [0.05, 0.1, 0.2]


def test_assert_split_order_accepts_matching_ids() -> None:
    arrays = {"train_sample_ids": np.asarray(["S1", "S2"])}

    _assert_split_order("train", arrays, ["S1", "S2"])


def test_assert_split_order_rejects_mismatch() -> None:
    arrays = {"val_sample_ids": np.asarray(["S1", "S2"])}

    with pytest.raises(RuntimeError, match="sample order"):
        _assert_split_order("val", arrays, ["S2", "S1"])
