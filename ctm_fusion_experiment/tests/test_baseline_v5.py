from __future__ import annotations

import json

import pytest
import torch
import yaml

from ctm_fusion_experiment.evaluate_baseline_v5 import build_baseline_v5_comparison_summary
from ctm_fusion_experiment.utils.risk_ensemble_selection import apply_risk_ensemble, choose_cindex_risk_ensemble


def test_choose_cindex_risk_ensemble_selects_better_candidate() -> None:
    risk_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_cindex_risk_ensemble(
        risk_matrix=risk_matrix,
        times=time,
        events=event,
        model_names=["reference", "better"],
        reference_index=0,
        min_c_index_delta=0.01,
    )

    assert result.c_index == pytest.approx(1.0)
    assert result.candidate_name == "single:better"
    assert result.weights == pytest.approx([0.0, 1.0])


def test_apply_risk_ensemble_uses_validation_scaling() -> None:
    risk_matrix = torch.tensor(
        [
            [10.0, 20.0],
            [2.0, 4.0],
        ]
    )
    risk = apply_risk_ensemble(
        risk_matrix,
        weights=[0.5, 0.5],
        risk_means=[15.0, 3.0],
        risk_stds=[5.0, 1.0],
    )

    assert risk.tolist() == pytest.approx([-1.0, 1.0])


def test_choose_cindex_risk_ensemble_can_restrict_to_ensembles() -> None:
    risk_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_cindex_risk_ensemble(
        risk_matrix=risk_matrix,
        times=time,
        events=event,
        model_names=["reference", "better_a", "better_b"],
        reference_index=0,
        min_c_index_delta=0.01,
        candidate_policy="ensemble_only_or_reference",
    )

    assert result.candidate_name == "top2_mean"
    assert result.weights == pytest.approx([0.0, 0.5, 0.5])


def test_choose_cindex_risk_ensemble_can_generate_larger_top_k() -> None:
    risk_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.1, 3.1, 2.1, 1.1],
            [4.2, 3.2, 2.2, 1.2],
            [4.3, 3.3, 2.3, 1.3],
        ]
    )
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_cindex_risk_ensemble(
        risk_matrix=risk_matrix,
        times=time,
        events=event,
        model_names=["reference", "a", "b", "c", "d"],
        reference_index=0,
        min_c_index_delta=0.01,
        candidate_policy="ensemble_only_or_reference",
        max_top_k=4,
    )

    assert any(candidate["candidate_name"] == "top4_mean" for candidate in result.candidates)


def test_build_baseline_v5_summary_reports_selected_and_oracle(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "graph_encoders": [{"training_seconds": 1.0}, {"training_seconds": 2.0}],
                "reference_baseline": {"test": {"c_index": 0.60}},
                "candidate_models": [
                    {"training_seconds": 1.0, "test": {"c_index": 0.60}},
                    {"training_seconds": 2.0, "test": {"c_index": 0.62}},
                ],
                "baseline_v5_selected": {
                    "candidate_name": "single:g0_b1",
                    "weights": [0.0, 1.0],
                    "validation_reference_c_index": 0.58,
                    "validation_c_index": 0.61,
                    "test": {"c_index": 0.62},
                    "pair_diagnostics": {
                        "improved_pairs": 8.0,
                        "regressed_pairs": 3.0,
                        "net_improved_pairs": 5.0,
                        "pair_credit_delta": 5.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_baseline_v5_comparison_summary(tmp_path)

    assert summary["selected_paired_comparison"]["mean_delta"] == pytest.approx(0.02)
    assert summary["oracle_best_single_paired_comparison"]["mean_delta"] == pytest.approx(0.02)
    assert summary["folds"][0]["net_improved_pairs"] == pytest.approx(5.0)
    assert (tmp_path / "baseline_v5_comparison_summary.json").exists()
    assert (tmp_path / "baseline_v5_fold_comparison.csv").exists()


def test_baseline_v5_configs_are_independent() -> None:
    v4 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm_v4_hardpair.yaml", encoding="utf-8"))
    v5 = yaml.safe_load(open("ctm_fusion_experiment/configs/baseline_v5.yaml", encoding="utf-8"))
    smoke = yaml.safe_load(open("ctm_fusion_experiment/configs/baseline_v5_smoke.yaml", encoding="utf-8"))

    assert v4["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v4_hardpair_formal"
    assert v5["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/baseline_v5_formal"
    assert smoke["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/baseline_v5_smoke"
    assert "baseline_v5" in v5
    assert len(v5["baseline_v5"]["baseline_seeds"]) == 3


def test_baseline_v7_config_uses_diverse_graph_ensemble() -> None:
    v7 = yaml.safe_load(open("ctm_fusion_experiment/configs/baseline_v7_diverse.yaml", encoding="utf-8"))

    assert v7["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/baseline_v7_diverse_formal"
    assert len(v7["baseline_v5"]["graph_seeds"]) == 3
    assert v7["baseline_v5"]["selection_policy"] == "ensemble_only_or_reference"
    assert v7["baseline_v5"]["max_top_k"] == 5


def test_baseline_v8_config_uses_bootstrap_selector() -> None:
    v8 = yaml.safe_load(open("ctm_fusion_experiment/configs/baseline_v8_bootstrap.yaml", encoding="utf-8"))

    assert v8["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/baseline_v8_bootstrap_formal"
    assert v8["baseline_v5"]["selection_strategy"] == "bootstrap"
    assert v8["baseline_v5"]["bootstrap"]["resamples"] > 0
