from __future__ import annotations

import numpy as np
import pytest
import yaml

from experiments.temporal_independent_v3.topology_aft_fusion import (
    build_topology_fingerprint_dataframe,
    select_feature_set,
    select_blend_alpha,
)
from experiments.temporal_independent_v3.cross_split_consensus import select_consensus_alpha


def test_topology_fingerprint_keeps_edge_identity_without_label_features() -> None:
    with open("research_config_v2.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    frame, feature_columns, metadata = build_topology_fingerprint_dataframe(config)

    assert frame["sample_id"].is_unique
    assert len(frame) == 3600
    assert metadata["num_edge_identity_features"] >= 10
    assert any(column.startswith("edge_edge_weight__") for column in feature_columns)
    assert "time" not in feature_columns
    assert "event" not in feature_columns
    assert np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all()

    legacy = select_feature_set(feature_columns, config, "legacy_summary")
    edge_identity = select_feature_set(feature_columns, config, "edge_identity")
    assert not any(column.startswith("edge_edge_weight__") for column in legacy)
    assert sum(column.startswith("edge_edge_weight__") for column in edge_identity) == 10
    assert len(edge_identity) > len(legacy)


def test_safe_blend_falls_back_when_expert_reverses_reference() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones_like(time)
    reference = -time
    expert = time

    result = select_blend_alpha(reference, expert, time, event, minimum_c_index_delta=0.0001)

    assert result["selected_expert"] is False
    assert result["selected"]["alpha"] == pytest.approx(0.0)


def test_safe_blend_selects_complementary_expert() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.ones_like(time)
    reference = np.asarray([4.0, 3.0, 5.0, 2.0, 1.0, 0.0])
    expert = -time

    result = select_blend_alpha(reference, expert, time, event, minimum_c_index_delta=0.01)

    assert result["selected_expert"] is True
    assert result["selected"]["alpha"] > 0.0
    assert result["selected"]["validation_c_index_delta"] >= 0.01


def test_consensus_uses_smallest_alpha_that_retains_each_split_gain() -> None:
    curves = [
        [
            {"alpha": 0.0, "validation_c_index": 0.70, "calibrated_validation_cox_loss": 1.0},
            {"alpha": 0.5, "validation_c_index": 0.79, "calibrated_validation_cox_loss": 0.9},
            {"alpha": 1.0, "validation_c_index": 0.80, "calibrated_validation_cox_loss": 0.95},
        ],
        [
            {"alpha": 0.0, "validation_c_index": 0.71, "calibrated_validation_cox_loss": 1.1},
            {"alpha": 0.5, "validation_c_index": 0.755, "calibrated_validation_cox_loss": 1.0},
            {"alpha": 1.0, "validation_c_index": 0.76, "calibrated_validation_cox_loss": 1.05},
        ],
    ]

    result = select_consensus_alpha(curves, gain_retention=0.90)

    assert result["selected_expert"] is True
    assert result["selected"]["alpha"] == pytest.approx(0.5)
