from __future__ import annotations

import json

import pandas as pd
import pytest

from ctm_fusion_experiment.evaluate_baseline_v5_reselect import build_baseline_v5_reselected_summary


def test_baseline_v5_reselect_uses_ensemble_policy(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    fold_dir = source / "fold_01"
    fold_dir.mkdir(parents=True)
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "reference_baseline": {"test": {"c_index": 0.5}},
                "candidate_models": [
                    {"name": "reference"},
                    {"name": "g0_b1"},
                    {"name": "g0_b2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (fold_dir / "baseline_v5_selection.json").write_text(
        json.dumps(
            {
                "validation_reference_c_index": 0.5,
                "risk_means": [0.0, 0.0, 0.0],
                "risk_stds": [1.0, 1.0, 1.0],
                "candidates": [
                    {"candidate_name": "reference", "weights": [1.0, 0.0, 0.0], "c_index": 0.5},
                    {"candidate_name": "single:g0_b1", "weights": [0.0, 1.0, 0.0], "c_index": 0.9},
                    {"candidate_name": "top2_mean", "weights": [0.0, 0.5, 0.5], "c_index": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1.0, 1.0, 1.0, 1.0],
            "reference_risk": [1.0, 2.0, 3.0, 4.0],
            "g0_b1_risk": [4.0, 3.0, 2.0, 1.0],
            "g0_b2_risk": [4.0, 3.0, 2.0, 1.0],
        }
    ).to_csv(fold_dir / "test_predictions.csv", index=False)

    summary = build_baseline_v5_reselected_summary(
        source,
        output,
        min_validation_delta=0.001,
        policy="ensemble_only_or_reference",
    )

    assert summary["folds"][0]["selected_candidate"] == "top2_mean"
    assert summary["selected_paired_comparison"]["mean_delta"] == pytest.approx(0.5)
    assert (output / "baseline_v5_reselected_summary.json").exists()
    assert (output / "baseline_v5_reselected_fold_comparison.csv").exists()
