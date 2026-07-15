from __future__ import annotations

import json

import pytest

from ctm_fusion_experiment.evaluate import build_comparison_summary


def test_build_comparison_summary_reads_fold_artifacts_and_writes_outputs(tmp_path) -> None:
    for fold, baseline, ctm in [(1, 0.60, 0.63), (2, 0.62, 0.64)]:
        fold_dir = tmp_path / f"fold_{fold:02d}"
        fold_dir.mkdir()
        (fold_dir / "fold_summary.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "baseline": {"test": {"c_index": baseline}},
                    "ctm": {"test": {"c_index": ctm}},
                }
            ),
            encoding="utf-8",
        )

    summary = build_comparison_summary(tmp_path)

    assert summary["num_folds"] == 2
    assert summary["paired_comparison"]["mean_delta"] == pytest.approx(0.025)
    assert (tmp_path / "comparison_summary.json").exists()
    assert (tmp_path / "fold_comparison.csv").exists()


def test_build_comparison_summary_preserves_run_metadata(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "baseline": {"test": {"c_index": 0.60}},
                "ctm": {"test": {"c_index": 0.63}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"device": "cpu", "sample_count": 32}),
        encoding="utf-8",
    )
    (tmp_path / "comparison_summary.json").write_text(
        json.dumps({"plots": ["existing_plot.png"]}),
        encoding="utf-8",
    )

    summary = build_comparison_summary(tmp_path)

    assert summary["device"] == "cpu"
    assert summary["sample_count"] == 32
    assert summary["plots"] == ["existing_plot.png"]
