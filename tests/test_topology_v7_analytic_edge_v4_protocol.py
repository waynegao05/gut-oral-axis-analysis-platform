from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments/topology_v7_analytic_edge_v4/"
    "experiment_plan.json"
)


def test_v4_plan_uses_fresh_cohorts_and_five_seed_audit() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["development_generation_seed"] == 20261012
    assert plan["audit_generation_seed"] == 20261013
    assert plan["development_model_seeds"] == [42]
    assert plan["audit_model_seeds"] == [7, 21, 42, 123, 2026]
    assert len(plan["candidates"]) == 3
    assert (
        plan["inference_contract"]["edge_weights_computed_inside_forward"]
        is True
    )
    assert (
        plan["audit_policy"]["noninferiority_alone_does_not_trigger_audit"]
        is True
    )


def test_v4_performance_gate_remains_stricter_than_noninferiority() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    performance = plan["development_gate"]
    replacement = plan["replacement_noninferiority_gate"]

    assert performance["minimum_macro_c_index_gain"] == 0.0015
    assert performance["minimum_integrated_auc_gain"] == 0.003
    assert replacement["minimum_macro_c_index_gain"] == -0.001
    assert replacement["minimum_integrated_auc_gain"] == -0.001
