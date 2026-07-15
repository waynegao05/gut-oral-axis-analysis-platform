from __future__ import annotations

from pathlib import Path

import pytest

from research.mainline_repeated_split_oof_v2 import _build_config, _parse_ints, _summarize_rows


def _base_config() -> dict:
    return {
        "seed": 42,
        "train": {"epochs": 180, "early_stop_patience": 18},
        "paths": {"output_dir": "outputs/original"},
    }


def test_build_config_keeps_base_config_untouched() -> None:
    base = _base_config()

    config = _build_config(
        base,
        split_seed=43,
        model_seed=7,
        output_dir=Path("outputs/new"),
        epochs_override=2,
        patience_override=1,
    )

    assert config["seed"] == 7
    assert config["train"]["split_seed"] == 43
    assert config["train"]["epochs"] == 2
    assert config["train"]["early_stop_patience"] == 1
    assert config["paths"]["output_dir"] == "outputs/new"
    assert base["seed"] == 42
    assert "split_seed" not in base["train"]


def test_summarize_rows_groups_by_split_seed() -> None:
    result = _summarize_rows(
        [
            {"split_seed": 42, "model_seed": 7, "test_c_index": 0.7, "test_loss": 1.0},
            {"split_seed": 42, "model_seed": 21, "test_c_index": 0.8, "test_loss": 0.9},
            {"split_seed": 43, "model_seed": 7, "test_c_index": 0.6, "test_loss": 1.1},
        ],
        {"tag": "toy", "skip_completed": True},
    )

    assert result["tag"] == "toy"
    assert result["skip_completed"] is True
    assert result["num_runs"] == 3
    assert result["mean_test_c_index"] == pytest.approx(0.7)
    assert result["split_summaries"]["42"]["num_model_seeds"] == 2
    assert result["split_summaries"]["42"]["mean_test_c_index"] == pytest.approx(0.75)


def test_parse_ints() -> None:
    assert _parse_ints("42, 43,,44") == [42, 43, 44]
