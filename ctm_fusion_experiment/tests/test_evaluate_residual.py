from __future__ import annotations

import json

import pytest

from ctm_fusion_experiment.evaluate_residual import build_residual_comparison_summary


def test_build_residual_comparison_summary_uses_residual_ctm_key(tmp_path) -> None:
    for fold, baseline, residual in [(1, 0.60, 0.61), (2, 0.62, 0.63)]:
        fold_dir = tmp_path / f"fold_{fold:02d}"
        fold_dir.mkdir()
        (fold_dir / "fold_summary.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "baseline": {"parameters": 10, "test": {"c_index": baseline}},
                    "residual_ctm": {"parameters": 20, "test": {"c_index": residual}},
                }
            ),
            encoding="utf-8",
        )

    summary = build_residual_comparison_summary(tmp_path)

    assert summary["num_folds"] == 2
    assert summary["paired_comparison"]["mean_delta"] == pytest.approx(0.01)
    assert (tmp_path / "residual_comparison_summary.json").exists()
    assert (tmp_path / "residual_fold_comparison.csv").exists()
