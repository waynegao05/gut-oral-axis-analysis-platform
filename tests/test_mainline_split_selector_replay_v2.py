from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.mainline_split_selector_replay_v2 import (
    _infer_split_seed,
    build_split_selector_replay_report,
)


def test_infer_split_seed_from_summary_path() -> None:
    path = Path("outputs/run/split_seed_44/meta/main_model_meta_oof_v2_summary.json")

    assert _infer_split_seed(path) == 44


def test_replay_report_selects_oof_first_candidate(tmp_path: Path) -> None:
    summary_path = tmp_path / "split_seed_44" / "meta" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "references": {"test_main_c_index": 0.70},
                "selection_policy": "validation_then_oof",
                "selected": {
                    "candidate_name": "old",
                    "alpha": 1.0,
                    "test_c_index": 0.69,
                    "test_delta_vs_main": -0.01,
                    "candidate_table": [
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
                            "candidate_name": "a",
                            "alpha": 1.0,
                            "oof_c_index": 0.72,
                            "oof_delta": 0.02,
                            "val_c_index": 0.705,
                            "val_delta": 0.005,
                            "high_disagreement_val_delta": 0.001,
                            "test_c_index": 0.71,
                        },
                        {
                            "candidate_name": "b",
                            "alpha": 1.0,
                            "oof_c_index": 0.715,
                            "oof_delta": 0.015,
                            "val_c_index": 0.71,
                            "val_delta": 0.01,
                            "high_disagreement_val_delta": 0.001,
                            "test_c_index": 0.705,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_split_selector_replay_report(summary_paths=[summary_path], selection_policy="oof_then_validation")

    assert report["num_splits"] == 1
    assert report["mean_selected_delta_vs_main"] == pytest.approx(0.01)
    assert report["rows"][0]["selected_candidate"] == "a"


def test_replay_report_defaults_to_hybrid_policy(tmp_path: Path) -> None:
    summary_path = tmp_path / "split_seed_44" / "meta" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "references": {"test_main_c_index": 0.70},
                "selection_policy": "validation_then_oof",
                "selected": {
                    "candidate_name": "old",
                    "alpha": 1.0,
                    "test_c_index": 0.69,
                    "test_delta_vs_main": -0.01,
                    "candidate_table": [
                        {
                            "candidate_name": "a",
                            "alpha": 1.0,
                            "oof_c_index": 0.72,
                            "oof_delta": 0.02,
                            "val_c_index": 0.705,
                            "val_delta": 0.005,
                            "high_disagreement_val_delta": 0.001,
                            "test_c_index": 0.71,
                        },
                        {
                            "candidate_name": "b",
                            "alpha": 1.0,
                            "oof_c_index": 0.715,
                            "oof_delta": 0.015,
                            "val_c_index": 0.71,
                            "val_delta": 0.01,
                            "high_disagreement_val_delta": 0.001,
                            "test_c_index": 0.705,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_split_selector_replay_report(summary_paths=[summary_path])

    assert report["selection_policy"] == "hybrid_validation_oof"
    assert report["rows"][0]["selected_candidate"] == "b"


def test_replay_report_default_high_disagreement_gate_falls_back_to_main(tmp_path: Path) -> None:
    summary_path = tmp_path / "split_seed_45" / "meta" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "references": {"test_main_c_index": 0.70},
                "selection_policy": "hybrid_validation_oof",
                "selected": {
                    "candidate_name": "unsafe",
                    "alpha": 1.0,
                    "test_c_index": 0.69,
                    "test_delta_vs_main": -0.01,
                    "candidate_table": [
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
                            "candidate_name": "unsafe",
                            "alpha": 1.0,
                            "oof_c_index": 0.73,
                            "oof_delta": 0.03,
                            "val_c_index": 0.71,
                            "val_delta": 0.01,
                            "high_disagreement_val_delta": 0.0001,
                            "test_c_index": 0.69,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_split_selector_replay_report(summary_paths=[summary_path])

    assert report["min_high_disagreement_val_delta"] == pytest.approx(0.0005)
    assert report["rows"][0]["selected_candidate"] == "main_model"
