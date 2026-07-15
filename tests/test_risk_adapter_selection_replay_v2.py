from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.risk_adapter_selection_replay_v2 import build_risk_adapter_selection_replay


def _write_summary(path: Path, *, validation_delta: float, test_delta: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "split_seed": 44,
                "references": {
                    "selection_reference": {
                        "name": "ensemble",
                        "validation_c_index": 0.70,
                        "test_c_index": 2.0 / 3.0,
                    }
                },
                "selected": {
                    "candidate_name": "adapter",
                    "validation_c_index": 0.70 + validation_delta,
                    "test_c_index": 2.0 / 3.0 + test_delta,
                },
                "validation_predictions": [
                    {
                        "sample_id": "v1",
                        "time": 1.0,
                        "event": 1.0,
                        "selection_reference_risk": 0.2,
                        "selected_risk": 0.4,
                    },
                    {
                        "sample_id": "v2",
                        "time": 2.0,
                        "event": 1.0,
                        "selection_reference_risk": 0.3,
                        "selected_risk": 0.2,
                    },
                    {
                        "sample_id": "v3",
                        "time": 3.0,
                        "event": 0.0,
                        "selection_reference_risk": 0.1,
                        "selected_risk": 0.1,
                    },
                ],
                "test_predictions": [
                    {
                        "sample_id": "a",
                        "time": 1.0,
                        "event": 1.0,
                        "selection_reference_risk": 0.2,
                        "selected_risk": 0.4,
                    },
                    {
                        "sample_id": "b",
                        "time": 2.0,
                        "event": 1.0,
                        "selection_reference_risk": 0.3,
                        "selected_risk": 0.2,
                    },
                    {
                        "sample_id": "c",
                        "time": 3.0,
                        "event": 0.0,
                        "selection_reference_risk": 0.1,
                        "selected_risk": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_replay_selects_candidate_above_validation_gate(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, validation_delta=0.001, test_delta=1.0 / 3.0)

    result = build_risk_adapter_selection_replay(summary_paths=[path], min_validation_delta=0.0003)

    assert result["num_adapter_selections"] == 1
    assert result["mean_selected_test_delta"] == pytest.approx(1.0 / 3.0)
    assert result["rows"][0]["pair_change"]["net_corrected_pairs"] == 1


def test_replay_falls_back_below_validation_gate(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, validation_delta=0.0001, test_delta=-0.1)

    result = build_risk_adapter_selection_replay(summary_paths=[path], min_validation_delta=0.0003)

    row = result["rows"][0]
    assert result["num_adapter_selections"] == 0
    assert row["selected_test_delta"] == pytest.approx(0.0)
    assert row["selected_candidate"] == "ensemble_reference"
    assert row["pair_change"]["corrected_pairs"] == 0
    assert row["selected_cohort_cox_loss_delta"] == pytest.approx(0.0)
    assert row["calibrated_selected_cohort_cox_loss_delta"] == pytest.approx(0.0)
