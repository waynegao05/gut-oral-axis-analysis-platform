from __future__ import annotations

import torch

from experiments.topology_v7_rank_objective_v6.losses import (
    comparable_pair_logistic_loss,
    horizon_pair_logistic_loss,
)


def test_comparable_pair_loss_rewards_correct_order() -> None:
    time = torch.tensor([10.0, 20.0, 30.0, 40.0])
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])
    correct = torch.tensor([3.0, 2.0, 1.0, 0.0])
    reversed_risk = -correct

    correct_loss = comparable_pair_logistic_loss(
        correct,
        time,
        event,
    )
    reversed_loss = comparable_pair_logistic_loss(
        reversed_risk,
        time,
        event,
    )

    assert torch.isfinite(correct_loss)
    assert correct_loss < reversed_loss


def test_horizon_loss_excludes_ambiguous_early_censoring() -> None:
    risk = torch.tensor([2.0, -2.0, 1.0, 0.0])
    time = torch.tensor([20.0, 20.0, 80.0, 100.0])
    event = torch.tensor([1.0, 0.0, 1.0, 0.0])

    first = horizon_pair_logistic_loss(
        risk,
        time,
        event,
        horizons=[60.0],
    )
    changed_ambiguous_risk = risk.clone()
    changed_ambiguous_risk[1] = 100.0
    second = horizon_pair_logistic_loss(
        changed_ambiguous_risk,
        time,
        event,
        horizons=[60.0],
    )

    assert torch.equal(first, second)


def test_losses_are_differentiable_and_handle_no_pairs() -> None:
    risk = torch.tensor([0.2, 0.1], requires_grad=True)
    time = torch.tensor([10.0, 10.0])
    event = torch.tensor([0.0, 0.0])
    loss = comparable_pair_logistic_loss(
        risk,
        time,
        event,
    ) + horizon_pair_logistic_loss(
        risk,
        time,
        event,
        horizons=[5.0],
    )

    loss.backward()

    assert torch.equal(loss.detach(), torch.tensor(0.0))
    assert risk.grad is not None
    assert torch.equal(risk.grad, torch.zeros_like(risk))
