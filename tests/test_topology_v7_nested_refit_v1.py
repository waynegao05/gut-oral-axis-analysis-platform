from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from experiments.topology_v7_nested_refit_v1.model import canonical_edge_vector
from experiments.topology_v7_nested_refit_v1.runner import (
    _cluster_pair_matrices,
    _select_development_candidate,
    _weighted_cluster_c_index,
    build_pair_cache,
    equal_group_cox_loss,
    soft_pairwise_ranking_loss,
)
from research.losses import cox_ph_loss
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "experiments"
    / "topology_v7_nested_refit_v1"
    / "experiment_plan.json"
)


def test_experiment_plan_separates_development_and_audit() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["status"] == "locked_before_development_generation"
    assert plan["development_generation_seed"] != plan["audit_generation_seed"]
    assert plan["audit_policy"]["train_on_all_four_non_test_groups"] is True
    assert plan["audit_policy"]["audit_test_labels_used_for_selection"] is False
    assert plan["audit_policy"]["audit_generation_seed_reruns_prohibited"] is True
    assert plan["audit_uncertainty"]["cluster_column"] == "primary_anchor_patient_id"
    assert plan["candidates"][0]["name"] == "baseline_pooled_cox"


def test_pair_cache_contains_only_valid_comparable_pairs() -> None:
    times = torch.tensor([1.0, 2.0, 3.0, 4.0])
    events = torch.tensor([1.0, 0.0, 1.0, 0.0])

    cache = build_pair_cache(times, events, use_ipcw=True)
    pairs = set(zip(cache.earlier.tolist(), cache.later.tolist()))

    assert pairs == {(0, 1), (0, 2), (0, 3), (2, 3)}
    assert cache.size == 4
    assert torch.isfinite(cache.weights).all()
    assert torch.all(cache.weights > 0)
    assert torch.isclose(cache.weights.mean(), torch.tensor(1.0))


def test_canonical_edge_vector_keeps_one_value_per_undirected_edge() -> None:
    graph = Data(
        x=torch.ones((3, 1)),
        node_type=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_index=torch.tensor(
            [
                [0, 1, 0, 2, 1, 2],
                [1, 0, 2, 0, 2, 1],
            ],
            dtype=torch.long,
        ),
        edge_attr=torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]]),
    )
    batch = Batch.from_data_list([graph])

    values = canonical_edge_vector(batch, num_node_types=3)

    assert values.shape == (1, 3)
    assert torch.equal(values, torch.tensor([[1.0, 2.0, 3.0]]))


def test_canonical_edge_vector_rejects_asymmetric_weights() -> None:
    graph = Data(
        x=torch.ones((2, 1)),
        node_type=torch.tensor([0, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0], [2.0]]),
    )
    batch = Batch.from_data_list([graph])

    try:
        canonical_edge_vector(batch, num_node_types=2)
    except ValueError as error:
        assert "equal weight" in str(error)
    else:
        raise AssertionError("Asymmetric edge weights must be rejected.")


def test_soft_ranking_loss_rewards_correct_order() -> None:
    earlier = torch.tensor([0, 0, 1], dtype=torch.long)
    later = torch.tensor([1, 2, 2], dtype=torch.long)
    weights = torch.ones(3)
    correct = torch.tensor([2.0, 1.0, 0.0])
    reversed_risk = torch.tensor([0.0, 1.0, 2.0])

    correct_loss = soft_pairwise_ranking_loss(
        correct,
        earlier=earlier,
        later=later,
        weights=weights,
        temperature=1.0,
    )
    reversed_loss = soft_pairwise_ranking_loss(
        reversed_risk,
        earlier=earlier,
        later=later,
        weights=weights,
        temperature=1.0,
    )

    assert correct_loss < reversed_loss


def test_equal_group_cox_is_unweighted_group_mean() -> None:
    risk = torch.tensor([1.2, 0.4, -0.3, 0.8, 0.1, -0.5])
    times = torch.tensor([1.0, 3.0, 4.0, 2.0, 5.0, 6.0])
    events = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])

    actual = equal_group_cox_loss(risk, times, events, groups)
    expected = torch.stack(
        [
            cox_ph_loss(
                risk[groups == group],
                times[groups == group],
                events[groups == group],
                ties_method="breslow",
            )
            for group in (0, 1)
        ]
    ).mean()

    assert torch.allclose(actual, expected)


def _aggregate(
    name: str,
    scores: list[float],
    *,
    mean_loss: float,
) -> dict:
    return {
        "candidate": {"name": name},
        "macro_mean_c_index": sum(scores) / len(scores),
        "minimum_group_c_index": min(scores),
        "mean_cox_loss": mean_loss,
        "median_best_epoch": 40,
        "folds": [
            {
                "holdout_group": group,
                "c_index": score,
                "cox_loss": mean_loss,
                "best_epoch": 40,
            }
            for group, score in enumerate(scores)
        ],
    }


def test_development_selection_requires_all_safety_checks() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    baseline = _aggregate(
        "baseline_pooled_cox",
        [0.740, 0.741, 0.742, 0.743, 0.744],
        mean_loss=5.0,
    )
    unsafe_high_mean = _aggregate(
        "unsafe",
        [0.750, 0.750, 0.750, 0.750, 0.730],
        mean_loss=5.0,
    )
    safe = _aggregate(
        "safe",
        [0.742, 0.743, 0.744, 0.745, 0.746],
        mean_loss=5.001,
    )

    selected, decisions = _select_development_candidate(
        plan,
        [baseline, unsafe_high_mean, safe],
    )
    decision_by_name = {row["candidate_name"]: row for row in decisions}

    assert selected["candidate"]["name"] == "safe"
    assert decision_by_name["unsafe"]["checks"]["worst_group_preserved"] is False
    assert decision_by_name["safe"]["eligible"] is True


def test_development_selection_falls_back_when_gain_is_too_small() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    baseline = _aggregate(
        "baseline_pooled_cox",
        [0.740, 0.741, 0.742, 0.743, 0.744],
        mean_loss=5.0,
    )
    tiny_gain = _aggregate(
        "tiny",
        [0.7402, 0.7412, 0.7422, 0.7432, 0.7442],
        mean_loss=5.0,
    )

    selected, _ = _select_development_candidate(plan, [baseline, tiny_gain])

    assert selected["candidate"]["name"] == "baseline_pooled_cox"


def test_cluster_pair_matrix_matches_exact_concordance() -> None:
    times = torch.tensor([1.0, 4.0, 2.0, 5.0, 3.0]).numpy()
    events = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]).numpy()
    risk = torch.tensor([2.0, -1.0, 1.0, 0.5, 0.0]).numpy()
    clusters = torch.tensor([0, 0, 1, 2, 2]).numpy()
    groups = torch.tensor([0, 0, 0, 1, 1]).numpy()

    denominator, numerators, _, _ = _cluster_pair_matrices(
        time_values=times,
        event_values=events,
        risks={"model": risk},
        cluster_index=clusters,
        group_values=groups,
        num_clusters=3,
    )
    matrix_c_index = _weighted_cluster_c_index(
        torch.ones((1, 3)).numpy(),
        numerator=numerators["model"],
        denominator=denominator,
    )[0]

    assert matrix_c_index == concordance_index(times, events, risk)
