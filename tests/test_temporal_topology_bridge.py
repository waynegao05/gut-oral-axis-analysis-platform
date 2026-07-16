from __future__ import annotations

import pytest

from src.temporal_topology_bridge import get_temporal_topology_model_bridge


@pytest.fixture(scope="module")
def bridge():
    return get_temporal_topology_model_bridge()


def test_bridge_replays_saved_cross_split_consensus(bridge) -> None:
    replay = bridge.score_reference_sample("S1")

    assert replay["maximum_absolute_error"] < 1e-5
    assert set(replay["splits"]) == {"42", "43"}


def test_web_score_uses_inferred_topology_and_latest_release(bridge) -> None:
    prediction = bridge.score(
        {
            "Fusobacterium": 0.18,
            "Porphyromonas": 0.15,
            "Prevotella": 0.10,
        },
        {"age": 52.0, "bmi": 24.5, "smoking": 1.0},
        {"bile_acids": 0.8, "scfa": 0.3},
    )

    risk = prediction.risk_result
    features = prediction.model_features
    assert risk["backend"] == "temporal_topology_aft_cross_split_consensus"
    assert risk["model_release"] == "temporal_topology_aft_cross_split_consensus_v1"
    assert 0.0 <= float(risk["risk_percentile"]) <= 100.0
    assert set(risk["split_consensus_risks"]) == {"42", "43"}
    assert features["topology_source"] == "inferred_from_web_inputs"
    assert len(features["inferred_function_scores"]) == 5
    assert len(features["inferred_edge_weights"]) == 10
    assert all(0.0 <= float(value) <= 1.0 for value in features["inferred_function_scores"].values())
    assert all(0.0 <= float(value) <= 1.0 for value in features["inferred_edge_weights"].values())
    assert features["gnn_inference_context"] == "fixed_median_batch_normalization_anchor"
    assert all(
        runtime.topology_model.training_size == 2160
        for runtime in bridge.runtimes.values()
    )
