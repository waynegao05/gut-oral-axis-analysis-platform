from __future__ import annotations

import json

import pytest
import torch
import yaml

from ctm_fusion_experiment.evaluate_residual_v3 import build_residual_v3_comparison_summary
from ctm_fusion_experiment.utils.aggressive_calibration import choose_aggressive_residual_alpha
from ctm_fusion_experiment.utils.utility_metrics import risk_utility_metrics


def test_risk_utility_metrics_reports_top_risk_event_lift() -> None:
    metrics = risk_utility_metrics(
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        event=[1.0, 1.0, 0.0, 0.0, 0.0],
        risk=[5.0, 4.0, 3.0, 2.0, 1.0],
        top_fraction=0.4,
    )

    assert metrics["c_index"] == pytest.approx(1.0)
    assert metrics["top_count"] == pytest.approx(2.0)
    assert metrics["top_event_rate"] == pytest.approx(1.0)
    assert metrics["top_event_lift"] == pytest.approx(2.5)
    assert metrics["high_low_event_gap"] == pytest.approx(1.0)
    assert metrics["time_separation"] > 0.0


def test_choose_aggressive_residual_alpha_can_use_inverted_delta() -> None:
    baseline = torch.tensor([1.0, 2.0, 3.0, 4.0])
    delta = torch.tensor([-2.0, -4.0, -6.0, -8.0])
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_aggressive_residual_alpha(
        baseline_risk=baseline,
        delta=delta,
        times=time,
        events=event,
        alpha_grid=[0.0, 1.0],
        top_fraction=0.25,
    )

    assert result.alpha == pytest.approx(1.0)
    assert result.c_index == pytest.approx(1.0)
    assert result.objective > 0.0
    assert any(candidate["eligible"] for candidate in result.candidates)


def test_choose_aggressive_residual_alpha_ignores_tiny_objective_gain() -> None:
    baseline = torch.tensor([4.0, 3.0, 2.0, 1.0])
    tiny_delta = torch.tensor([0.000001, 0.0, 0.0, 0.0])
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_aggressive_residual_alpha(
        baseline_risk=baseline,
        delta=tiny_delta,
        times=time,
        events=event,
        alpha_grid=[0.0, 1.0],
        top_fraction=0.25,
        min_objective_delta=0.00005,
    )

    assert result.alpha == pytest.approx(0.0)
    assert result.objective == pytest.approx(0.0)


def test_build_residual_v3_summary_reports_utility_comparisons(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    utility_base = {
        "top_event_lift": 1.0,
        "high_low_event_gap": 0.1,
        "risk_std": 0.5,
        "time_separation": 1.0,
    }
    utility_raw = {
        "top_event_lift": 1.1,
        "high_low_event_gap": 0.12,
        "risk_std": 0.6,
        "time_separation": 1.2,
    }
    utility_aggressive = {
        "top_event_lift": 1.3,
        "high_low_event_gap": 0.2,
        "risk_std": 0.7,
        "time_separation": 1.4,
    }
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "baseline": {"parameters": 10, "test": {"c_index": 0.60}, "test_utility": utility_base},
                "residual_ctm_raw": {
                    "parameters": 20,
                    "test": {"c_index": 0.61},
                    "test_utility": utility_raw,
                },
                "residual_ctm_aggressive": {
                    "parameters": 20,
                    "selected_alpha": 1.5,
                    "validation_objective": 0.02,
                    "validation_c_index": 0.62,
                    "test": {"c_index": 0.63},
                    "test_utility": utility_aggressive,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_residual_v3_comparison_summary(tmp_path)

    assert summary["aggressive_paired_comparison"]["mean_delta"] == pytest.approx(0.03)
    assert summary["folds"][0]["aggressive_top_event_lift_delta"] == pytest.approx(0.3)
    assert "top_event_lift" in summary["utility_paired_comparisons"]
    assert (tmp_path / "residual_v3_comparison_summary.json").exists()
    assert (tmp_path / "residual_v3_fold_comparison.csv").exists()


def test_residual_v3_configs_are_independent() -> None:
    v2 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm_v2.yaml", encoding="utf-8"))
    v3 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm_v3.yaml", encoding="utf-8"))
    smoke = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_v3_smoke.yaml", encoding="utf-8"))

    assert v2["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v2_formal"
    assert v3["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v3_formal"
    assert smoke["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v3_smoke"
    assert min(v3["calibration"]["alpha_grid"]) < 0.0
    assert max(v3["calibration"]["alpha_grid"]) > 1.0
    assert "aggressive_selection" in v3
