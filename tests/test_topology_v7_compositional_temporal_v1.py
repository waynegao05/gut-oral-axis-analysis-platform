from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.topology_v7_compositional_temporal_v1.metrics import (
    fit_breslow_survival,
    ipcw_brier_score,
    normalized_trapezoid,
    uno_c_index,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments"
    / "topology_v7_compositional_temporal_v1"
    / "experiment_plan.json"
)


def test_experiment_plan_keeps_audit_and_edges_isolated() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["isolation"]["modify_deployed_backend"] is False
    assert plan["isolation"]["reuse_seen_audit_for_model_selection"] is False
    assert (
        plan["future_cohorts"]["development_generation_seed"]
        != plan["future_cohorts"]["audit_generation_seed"]
    )
    assert (
        plan["internal_edge_policy"][
            "precomputed_edge_weight_used_as_model_input"
        ]
        is False
    )
    assert "event" in plan["internal_edge_policy"]["prohibited_inputs"]
    assert "generation_group_id" in plan["internal_edge_policy"]["prohibited_inputs"]


def test_uno_c_index_recognizes_perfect_and_reversed_order() -> None:
    train_time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    train_event = np.ones(5, dtype=int)
    test_time = np.asarray([1.0, 2.0, 3.0, 4.0])
    test_event = np.ones(4, dtype=int)

    perfect = uno_c_index(
        train_time=train_time,
        train_event=train_event,
        test_time=test_time,
        test_event=test_event,
        risk=np.asarray([4.0, 3.0, 2.0, 1.0]),
        tau=4.0,
    )
    reversed_score = uno_c_index(
        train_time=train_time,
        train_event=train_event,
        test_time=test_time,
        test_event=test_event,
        risk=np.asarray([1.0, 2.0, 3.0, 4.0]),
        tau=4.0,
    )

    assert perfect == 1.0
    assert reversed_score == 0.0


def test_breslow_survival_is_monotone_and_risk_ordered() -> None:
    survival = fit_breslow_survival(
        train_time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        train_event=np.asarray([1, 1, 0, 1]),
        train_risk=np.asarray([1.0, 0.5, 0.0, -0.5]),
        evaluation_risk=np.asarray([1.0, -1.0]),
        horizons=[1.0, 2.0, 3.0],
    )

    assert survival.shape == (2, 3)
    assert np.all(np.diff(survival, axis=1) <= 0)
    assert np.all(survival[0] <= survival[1])


def test_ipcw_brier_is_zero_for_perfect_uncensored_predictions() -> None:
    score = ipcw_brier_score(
        train_time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        train_event=np.ones(4, dtype=int),
        test_time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        test_event=np.ones(4, dtype=int),
        survival_probability=np.asarray([0.0, 0.0, 1.0, 1.0]),
        horizon=2.5,
    )

    assert score == 0.0


def test_normalized_trapezoid_constant_curve() -> None:
    assert normalized_trapezoid([0.75, 0.75, 0.75], [1.0, 2.0, 4.0]) == 0.75
