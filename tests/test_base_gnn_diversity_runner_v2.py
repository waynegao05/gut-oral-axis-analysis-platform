from __future__ import annotations

from pathlib import Path

import pytest

from research.base_gnn_diversity_runner_v2 import _build_config, _parse_ints, _parse_strings, _summarize_rows


def _base_config() -> dict:
    return {
        "seed": 42,
        "train": {
            "epochs": 180,
            "early_stop_patience": 18,
            "ranking_weight": 0.0,
            "ranking_margin": 0.0,
            "dropout": 0.25,
        },
        "paths": {"output_dir": "outputs/original"},
        "graph_preprocess": {"keep_top_k_edges": None, "min_edge_weight": None},
    }


def test_build_config_applies_variant_without_mutating_base() -> None:
    base = _base_config()

    config = _build_config(
        base,
        variant="ranking_w0p02",
        seed=7,
        split_seed=42,
        output_dir=Path("outputs/diverse"),
        epochs_override=2,
        patience_override=1,
    )

    assert config["seed"] == 7
    assert config["train"]["split_seed"] == 42
    assert config["train"]["ranking_weight"] == pytest.approx(0.02)
    assert config["train"]["epochs"] == 2
    assert config["train"]["early_stop_patience"] == 1
    assert config["paths"]["output_dir"] == "outputs/diverse"
    assert config["experiment"]["base_gnn_diversity_variant"] == "ranking_w0p02"
    assert base["train"]["ranking_weight"] == 0.0
    assert "experiment" not in base


def test_build_config_applies_graph_preprocess_variant() -> None:
    config = _build_config(
        _base_config(),
        variant="topk8",
        seed=21,
        split_seed=42,
        output_dir=Path("outputs/topk8"),
        epochs_override=None,
        patience_override=None,
    )

    assert config["graph_preprocess"]["keep_top_k_edges"] == 8
    assert config["graph_preprocess"]["min_edge_weight"] is None


def test_build_config_applies_large_cox_risk_set_variant() -> None:
    base = _base_config()
    base["train"]["batch_size"] = 8

    config = _build_config(
        base,
        variant="cox_batch32",
        seed=42,
        split_seed=42,
        output_dir=Path("outputs/batch32"),
        epochs_override=None,
        patience_override=None,
    )

    assert config["train"]["batch_size"] == 32
    assert config["train"]["ranking_weight"] == 0.0
    assert base["train"]["batch_size"] == 8


def test_build_config_applies_train_only_tabular_standardization() -> None:
    base = _base_config()

    config = _build_config(
        base,
        variant="tabular_standardized",
        seed=7,
        split_seed=42,
        output_dir=Path("outputs/tabular_standardized"),
        epochs_override=None,
        patience_override=None,
    )

    assert config["tabular_preprocess"]["standardize"] is True
    assert "tabular_preprocess" not in base


def test_build_config_combines_tabular_standardization_and_ranking() -> None:
    config = _build_config(
        _base_config(),
        variant="tabular_standardized_ranking_w0p02",
        seed=7,
        split_seed=42,
        output_dir=Path("outputs/tabular_standardized_ranking"),
        epochs_override=None,
        patience_override=None,
    )

    assert config["tabular_preprocess"]["standardize"] is True
    assert config["train"]["ranking_weight"] == pytest.approx(0.02)
    assert config["train"]["ranking_warmup_epochs"] == 8


def test_summarize_rows_ranks_variants() -> None:
    result = _summarize_rows(
        [
            {"variant": "baseline", "seed": 7, "test_c_index": 0.70, "test_loss": 1.0, "best_val_c_index": 0.65},
            {"variant": "baseline", "seed": 21, "test_c_index": 0.72, "test_loss": 0.9, "best_val_c_index": 0.66},
            {"variant": "ranking_w0p02", "seed": 7, "test_c_index": 0.74, "test_loss": 0.8, "best_val_c_index": 0.67},
        ],
        {"tag": "toy"},
    )

    assert result["tag"] == "toy"
    assert result["num_runs"] == 3
    assert result["variant_summaries"]["baseline"]["mean_test_c_index"] == pytest.approx(0.71)
    assert result["ranking_table"][0]["variant"] == "ranking_w0p02"


def test_parse_helpers() -> None:
    assert _parse_ints("7,21,,42") == [7, 21, 42]
    assert _parse_strings("baseline, topk8,,") == ["baseline", "topk8"]
