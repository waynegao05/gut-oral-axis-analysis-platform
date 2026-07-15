from __future__ import annotations

import json

import pytest
import torch
import yaml

from ctm_fusion_experiment.evaluate_residual_v4 import build_residual_v4_comparison_summary
from ctm_fusion_experiment.utils.cindex_ensemble_selection import choose_cindex_ensemble
from ctm_fusion_experiment.utils.losses import baseline_discordant_pairwise_loss, pairwise_ranking_loss
from ctm_fusion_experiment.utils.pair_diagnostics import pairwise_cindex_diagnostics


def test_pairwise_ranking_loss_rewards_correct_order() -> None:
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.ones(3)
    correct = torch.tensor([3.0, 2.0, 1.0])
    reversed_risk = torch.tensor([1.0, 2.0, 3.0])

    assert pairwise_ranking_loss(correct, time, event) < pairwise_ranking_loss(reversed_risk, time, event)


def test_baseline_discordant_pairwise_loss_focuses_wrong_baseline_pairs() -> None:
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.ones(3)
    wrong_baseline = torch.tensor([1.0, 2.0, 3.0])
    corrected = torch.tensor([3.0, 2.0, 1.0])
    still_wrong = torch.tensor([1.0, 2.0, 3.0])

    assert baseline_discordant_pairwise_loss(corrected, wrong_baseline, time, event) < baseline_discordant_pairwise_loss(
        still_wrong,
        wrong_baseline,
        time,
        event,
    )


def test_choose_cindex_ensemble_selects_validation_improver() -> None:
    baseline = torch.zeros(4)
    deltas = [
        torch.tensor([4.0, 3.0, 2.0, 1.0]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    ]
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)

    result = choose_cindex_ensemble(
        baseline_risk=baseline,
        deltas=deltas,
        times=time,
        events=event,
        alpha_grid=[0.0, 1.0],
        min_c_index_delta=0.01,
    )

    assert result.c_index == pytest.approx(1.0)
    assert result.candidate_name != "baseline"
    assert result.weights[0] > 0.0


def test_pairwise_cindex_diagnostics_counts_fixed_pairs() -> None:
    diagnostics = pairwise_cindex_diagnostics(
        time=[1.0, 2.0, 3.0],
        event=[1.0, 1.0, 1.0],
        baseline_risk=[1.0, 2.0, 3.0],
        candidate_risk=[3.0, 2.0, 1.0],
    )

    assert diagnostics["permissible_pairs"] == pytest.approx(3.0)
    assert diagnostics["improved_pairs"] == pytest.approx(3.0)
    assert diagnostics["regressed_pairs"] == pytest.approx(0.0)
    assert diagnostics["pair_credit_delta"] == pytest.approx(3.0)


def test_build_residual_v4_summary_reports_selected_and_oracle(tmp_path) -> None:
    fold_dir = tmp_path / "fold_01"
    fold_dir.mkdir()
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "graph_encoder": {"training_seconds": 1.0},
                "baseline": {"parameters": 10, "training_seconds": 2.0, "test": {"c_index": 0.60}},
                "residual_ctm_seeds": [
                    {"parameters": 20, "training_seconds": 3.0, "test": {"c_index": 0.61}},
                    {"parameters": 20, "training_seconds": 4.0, "test": {"c_index": 0.62}},
                ],
                "residual_ctm_v4_selected": {
                    "candidate_name": "mean_alpha_1",
                    "weights": [0.5, 0.5],
                    "validation_baseline_c_index": 0.58,
                    "validation_c_index": 0.59,
                    "test": {"c_index": 0.63},
                    "pair_diagnostics": {
                        "improved_pairs": 10.0,
                        "regressed_pairs": 4.0,
                        "net_improved_pairs": 6.0,
                        "pair_credit_delta": 5.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_residual_v4_comparison_summary(tmp_path)

    assert summary["selected_paired_comparison"]["mean_delta"] == pytest.approx(0.03)
    assert summary["oracle_best_seed_paired_comparison"]["mean_delta"] == pytest.approx(0.02)
    assert summary["folds"][0]["net_improved_pairs"] == pytest.approx(6.0)
    assert (tmp_path / "residual_v4_comparison_summary.json").exists()
    assert (tmp_path / "residual_v4_fold_comparison.csv").exists()


def test_residual_v4_configs_are_independent() -> None:
    v3 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm_v3.yaml", encoding="utf-8"))
    v4 = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm_v4.yaml", encoding="utf-8"))
    smoke = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_v4_smoke.yaml", encoding="utf-8"))

    assert v3["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v3_formal"
    assert v4["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v4_formal"
    assert smoke["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_v4_smoke"
    assert len(v4["ensemble"]["residual_seeds"]) == 3
    assert v4["loss"]["pairwise_ranking_weight"] > 0.0
