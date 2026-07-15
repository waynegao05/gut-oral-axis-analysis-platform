from __future__ import annotations

import json

from ctm_fusion_experiment.plot_results import build_plots


def test_build_plots_writes_prompt_required_visualizations(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "baseline": {"parameters": 10, "test": {"c_index": 0.60}},
                "ctm": {"parameters": 20, "test": {"c_index": 0.63}},
            }
        ),
        encoding="utf-8",
    )
    history = [{"epoch": 1, "train_loss": 1.0, "c_index": 0.5}, {"epoch": 2, "train_loss": 0.8, "c_index": 0.6}]
    (fold_dir / "baseline_history.json").write_text(json.dumps(history), encoding="utf-8")
    (fold_dir / "ctm_history.json").write_text(json.dumps(history), encoding="utf-8")
    (fold_dir / "ctm_analysis.json").write_text(
        json.dumps(
            {
                "stable_tick_histogram": {"1": 3, "2": 5},
                "mean_attention_by_tick_and_modality": [
                    [0.2, 0.3, 0.5],
                    [0.4, 0.3, 0.3],
                ],
            }
        ),
        encoding="utf-8",
    )

    written = build_plots(tmp_path)

    assert len(written) == 5
    for filename in written:
        assert (tmp_path / "plots" / filename).exists()
