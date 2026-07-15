from __future__ import annotations

import torch
import yaml

from ctm_fusion_experiment.models.residual_ctm_fusion import ResidualCTMFusionModel
from ctm_fusion_experiment.utils.losses import residual_ctm_cox_loss


def test_residual_ctm_initially_preserves_baseline_risk() -> None:
    model = ResidualCTMFusionModel(
        graph_dim=9,
        clinical_dim=4,
        metabolite_dim=3,
        d_input=8,
        d_model=12,
        iterations=3,
        memory_length=3,
        nlm_hidden_dim=4,
        n_heads=2,
        n_synch_action=6,
        n_synch_out=7,
        n_self_pairs=2,
        synapse_depth=2,
        dropout=0.0,
        max_residual_gate=1.0,
        initial_gate_logit=-2.0,
    )
    baseline_risk = torch.tensor([0.2, -0.1, 0.5, 0.0, -0.4])

    output = model(
        graph_features=torch.randn(5, 9),
        clinical_features=torch.randn(5, 4),
        metabolite_features=torch.randn(5, 3),
        baseline_risk=baseline_risk,
        track_attention=True,
    )

    assert torch.allclose(
        output["risk_per_tick"],
        baseline_risk.unsqueeze(1).expand(-1, 3),
    )
    assert torch.allclose(output["delta_per_tick"], torch.zeros(5, 3))
    assert output["attention_weights"].shape == (5, 3, 2, 3)


def test_residual_ctm_loss_combines_deep_supervision_and_distillation() -> None:
    baseline_risk = torch.tensor([0.1, 0.4, -0.3, 0.2])
    risk_per_tick = torch.stack(
        [
            baseline_risk,
            baseline_risk + torch.tensor([0.02, -0.01, 0.00, 0.01]),
            baseline_risk + torch.tensor([0.03, -0.02, 0.01, 0.00]),
        ],
        dim=1,
    )
    time = torch.tensor([4.0, 3.0, 2.0, 1.0])
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])

    result = residual_ctm_cox_loss(
        risk_scores_per_tick=risk_per_tick,
        baseline_risk=baseline_risk,
        times=time,
        events=event,
        mean_weight=0.5,
        final_weight=0.3,
        best_weight=0.2,
        distillation_weight=0.1,
        separation_weight=0.05,
        min_risk_std=0.05,
    )

    assert result.loss.ndim == 0
    assert result.components["mean_tick_cox"].item() > 0
    assert result.components["distillation"].item() >= 0
    assert 0 <= result.best_loss_tick < 3


def test_residual_configs_use_separate_output_dirs() -> None:
    original = yaml.safe_load(open("ctm_fusion_experiment/configs/ctm_fusion.yaml", encoding="utf-8"))
    residual = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_ctm.yaml", encoding="utf-8"))
    smoke = yaml.safe_load(open("ctm_fusion_experiment/configs/residual_smoke.yaml", encoding="utf-8"))

    assert original["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/formal"
    assert residual["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_formal"
    assert smoke["paths"]["output_dir"] == "outputs/ctm_fusion_experiment/residual_smoke"
