from __future__ import annotations

import json
from pathlib import Path

from experiments.topology_v7_rank_objective_v6.runner import (
    _inner_groups,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments/topology_v7_rank_objective_v6/"
    "experiment_plan.json"
)


def test_v6_uses_full_inner_logo_and_fresh_seeds() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["development_generation_seed"] == 20261016
    assert plan["audit_generation_seed"] == 20261017
    assert plan["development_refit_model_seeds"] == [42]
    assert plan["audit_refit_model_seeds"] == [
        7,
        21,
        42,
        123,
        2026,
    ]
    for outer_group in range(5):
        inner = _inner_groups(outer_group)
        assert len(inner) == 4
        assert outer_group not in inner
        assert sorted(inner + [outer_group]) == list(range(5))


def test_v6_locks_only_one_prescreened_rank_candidate() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    candidates = {
        row["name"]: row for row in plan["candidates"]
    }

    assert len(candidates) == 2
    candidate = candidates[
        "exact_internal_edge_cox_horizon_rank_020"
    ]
    assert candidate["comparable_rank_weight"] == 0.0
    assert candidate["horizon_rank_weight"] == 0.2
    assert (
        plan["inference_contract"][
            "rank_objectives_used_only_during_training"
        ]
        is True
    )
