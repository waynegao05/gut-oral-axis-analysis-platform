from __future__ import annotations

import json

import pytest

from research.meta_selector_sensitivity_v2 import (
    _parse_floats,
    _parse_paths,
    _select_for_threshold,
    build_selector_sensitivity_report,
)


def _rows() -> list[dict]:
    return [
        {
            "candidate_name": "main_model",
            "alpha": 0.0,
            "oof_c_index": 0.70,
            "oof_delta": 0.0,
            "val_c_index": 0.70,
            "val_delta": 0.0,
            "high_disagreement_val_delta": 0.0,
            "test_c_index": 0.70,
        },
        {
            "candidate_name": "weak_oof_high_val",
            "alpha": 1.0,
            "oof_c_index": 0.70001,
            "oof_delta": 0.00001,
            "val_c_index": 0.72,
            "val_delta": 0.02,
            "high_disagreement_val_delta": 0.01,
            "test_c_index": 0.69,
        },
        {
            "candidate_name": "strict_candidate",
            "alpha": 1.0,
            "oof_c_index": 0.70004,
            "oof_delta": 0.00004,
            "val_c_index": 0.71,
            "val_delta": 0.01,
            "high_disagreement_val_delta": 0.005,
            "test_c_index": 0.73,
        },
    ]


def test_select_for_threshold_keeps_main_model_as_fallback() -> None:
    selected = _select_for_threshold(
        _rows(),
        min_oof_delta=0.00005,
        min_val_delta=0.0,
        min_high_disagreement_val_delta=0.0,
        selection_policy="validation_then_oof",
    )

    assert selected["eligible_count"] == 1
    assert selected["selected"]["candidate_name"] == "main_model"


def test_select_for_threshold_filters_weak_oof_candidate() -> None:
    selected = _select_for_threshold(
        _rows(),
        min_oof_delta=0.00003,
        min_val_delta=0.0,
        min_high_disagreement_val_delta=0.0,
        selection_policy="validation_then_oof",
    )

    assert selected["selected"]["candidate_name"] == "strict_candidate"


def test_select_for_threshold_supports_hybrid_validation_oof_policy() -> None:
    selected = _select_for_threshold(
        [
            {
                "candidate_name": "high_val",
                "alpha": 1.0,
                "oof_c_index": 0.7001,
                "oof_delta": 0.0001,
                "val_c_index": 0.7200,
                "val_delta": 0.0200,
                "high_disagreement_val_delta": 0.001,
                "test_c_index": 0.70,
            },
            {
                "candidate_name": "balanced",
                "alpha": 1.0,
                "oof_c_index": 0.8000,
                "oof_delta": 0.1000,
                "val_c_index": 0.7110,
                "val_delta": 0.0110,
                "high_disagreement_val_delta": 0.001,
                "test_c_index": 0.72,
            },
        ],
        min_oof_delta=0.0,
        min_val_delta=0.0,
        min_high_disagreement_val_delta=0.0,
        selection_policy="hybrid_validation_oof",
    )

    assert selected["selected"]["candidate_name"] == "balanced"


def test_build_selector_sensitivity_report_reads_summary(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "references": {"test_main_c_index": 0.70},
                "selected": {
                    "candidate_name": "weak_oof_high_val",
                    "alpha": 1.0,
                    "oof_delta": 0.00001,
                    "val_delta": 0.02,
                    "high_disagreement_val_delta": 0.01,
                    "test_c_index": 0.69,
                    "test_delta_vs_main": -0.01,
                    "candidate_table": _rows(),
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_selector_sensitivity_report(
        summary_paths=[summary_path],
        min_oof_deltas=[0.0, 0.00003],
    )

    assert report["reports"][0]["threshold_results"][0]["selected"]["candidate_name"] == "weak_oof_high_val"
    assert report["reports"][0]["threshold_results"][1]["selected"]["candidate_name"] == "strict_candidate"


def test_parse_helpers() -> None:
    assert _parse_paths("a.json, b.json,,") == ["a.json", "b.json"]
    assert _parse_floats("0, 0.00003,,") == pytest.approx([0.0, 0.00003])
