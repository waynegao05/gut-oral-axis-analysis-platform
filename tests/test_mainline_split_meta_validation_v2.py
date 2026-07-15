from __future__ import annotations

from pathlib import Path

import pytest

from research.mainline_split_meta_validation_v2 import (
    _checkpoint_glob,
    _parse_floats,
    _parse_ints,
    _parse_strings,
    _summarize_rows,
)


def test_checkpoint_glob_points_to_split_seed_model_dirs() -> None:
    pattern = _checkpoint_glob(Path("outputs/run/gnn"), split_seed=43)

    assert pattern == "outputs/run/gnn/split_seed_43/model_seed_*/best_model.pt"


def test_summarize_rows_counts_positive_deltas() -> None:
    result = _summarize_rows(
        [
            {"selected_test_delta_vs_main": 0.1},
            {"selected_test_delta_vs_main": -0.2},
            {"selected_test_delta_vs_main": 0.0},
        ],
        {"tag": "toy"},
    )

    assert result["tag"] == "toy"
    assert result["num_splits"] == 3
    assert result["mean_selected_delta_vs_main"] == pytest.approx(-0.03333333333333333)
    assert result["num_positive_selected_deltas"] == 1


def test_parse_helpers() -> None:
    assert _parse_ints("42, 43,,44") == [42, 43, 44]
    assert _parse_floats("0.03,0.05,,") == pytest.approx([0.03, 0.05])
    assert _parse_strings("linear, mlp,,") == ["linear", "mlp"]
