from __future__ import annotations

import pytest
import torch

from research.losses import cox_ph_loss
from research.train_v2 import compute_cohort_evaluation_losses


def test_cohort_cox_loss_uses_full_risk_set_and_adds_auxiliary_terms() -> None:
    risk = torch.tensor([1.2, 0.4, -0.2], dtype=torch.float32)
    time = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    event = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

    metrics = compute_cohort_evaluation_losses(
        risk=risk,
        time=time,
        event=event,
        survival_head_type="cox",
        time_bin_edges=None,
        time_logits=None,
        ranking_weight=0.0,
        ranking_margin=0.0,
        graph_aux_loss=0.1,
        node_aux_loss=0.2,
        graph_aux_weight=0.08,
        node_aux_weight=0.05,
    )

    expected_cox = float(cox_ph_loss(risk, time, event).item())
    assert metrics["cohort_cox_loss"] == pytest.approx(expected_cox)
    assert metrics["cohort_loss"] == pytest.approx(expected_cox + 0.08 * 0.1 + 0.05 * 0.2)


def test_discrete_time_cohort_loss_requires_logits_and_bin_edges() -> None:
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="time_bin_edges and time_logits"):
        compute_cohort_evaluation_losses(
            risk=values,
            time=values,
            event=torch.ones_like(values),
            survival_head_type="discrete_time",
            time_bin_edges=None,
            time_logits=None,
            ranking_weight=0.0,
            ranking_margin=0.0,
            graph_aux_loss=0.0,
            node_aux_loss=0.0,
            graph_aux_weight=0.0,
            node_aux_weight=0.0,
        )
