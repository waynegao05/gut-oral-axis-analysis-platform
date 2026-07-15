from __future__ import annotations

import json

import pytest
import torch
import yaml

from ctm_fusion_experiment.evaluate_residual_v2 import build_residual_v2_comparison_summary
from ctm_fusion_experiment.utils.calibration import apply_residual_alpha, choose_residual_alpha


def test_choose_residual_alpha_can_fall_back_to_baseline() -> None:
    baseline = torch.tensor([0.4, 0.3, 0.2, 0.1])
    harmful_delta = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_residual_alpha(
        baseline_risk=baseline,
        delta=harmful_delta,
        times=time,
        events=event,
        alpha_grid=[0.0, 0.5, 1.0],
    )

    assert result.alpha == 0.0
    assert result.c_index == pytest.approx(1.0)
    assert torch.allclose(apply_residual_alpha(baseline, harmful_delta, result.alpha), baseline)


def test_build_residual_v2_summary_reports_raw_and_calibrated_scores(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "baseline": {"parameters": 10, "test": {"c_index": 0.60}},
                "residual_ctm_raw": {"parameters": 20, "test": {"c_index": 0.58}},
                "residual_ctm_calibrated": {
                    "parameters": 20,
                    "selected_alpha": 0.0,
                    "test": {"c_index": 0.60},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_residual_v2_comparison_summary(tmp_path)

    assert summary["raw_paired_comparison"]["mean_delta"] == pytest.approx(-0.02)
    assert summary["calibrated_paired_comparison"]["mean_delta"] == pytest.approx(0.0)
    assert (tmp_path / "residual_v2_comparison_summary.json").exists()
    assert (tmp_path / "residual_v2_fold_comparison.csv").exists()


def test_residual_v2_configs_are_independent() -> None:
    v1 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm.yaml", encoding="utf-8"))
    v2 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm_v2.yaml", encoding="utf-8"))
    smoke = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_v2_smoke.yaml", encoding="utf-8"))

    assert v1["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_formal"
    assert v2["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v2_formal"
    assert smoke["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v2_smoke"
    assert 0.0 in v2["calibration"]["alpha_grid"]
