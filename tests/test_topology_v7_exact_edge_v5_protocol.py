from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments/topology_v7_exact_edge_v5/"
    "experiment_plan.json"
)


def test_v5_plan_uses_fresh_locked_cohorts() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["development_generation_seed"] == 20261014
    assert plan["audit_generation_seed"] == 20261015
    assert plan["development_model_seeds"] == [42]
    assert plan["audit_model_seeds"] == [7, 21, 42, 123, 2026]
    assert len(plan["candidates"]) == 2
    assert (
        plan["edge_fidelity_gate"]["minimum_held_out_r2"]
        == 0.999
    )
    assert (
        plan["inference_contract"][
            "edge_weights_computed_inside_forward"
        ]
        is True
    )


def test_v5_noninferiority_is_tight_and_cannot_unlock_audit() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    replacement = plan["replacement_noninferiority_gate"]

    assert replacement["minimum_macro_c_index_gain"] == -0.00025
    assert replacement["minimum_integrated_auc_gain"] == -0.00025
    assert replacement["maximum_worst_group_regression"] == 0.00075
    assert (
        plan["audit_policy"][
            "noninferiority_alone_does_not_trigger_audit"
        ]
        is True
    )
