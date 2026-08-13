from pathlib import Path

import yaml

from experiments.topology_v7_diagnosis.diagnose import (
    GROUP_COLUMN,
    _as_builtin,
    _diagnostic_interpretation,
    _feature_frame,
    _group_split,
    _random_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_frame():
    config = yaml.safe_load((PROJECT_ROOT / "research_config_v2.yaml").read_text(encoding="utf-8"))
    frame, feature_sets, _ = _feature_frame(config)
    return config, frame, feature_sets


def test_generation_group_is_split_metadata_not_model_feature() -> None:
    _, _, feature_sets = _load_frame()
    assert all(GROUP_COLUMN not in columns for columns in feature_sets.values())


def test_numpy_scalars_are_serializable() -> None:
    import json

    import numpy as np

    json.dumps(_as_builtin({"value": np.int64(42), "score": np.float64(0.5)}))


def test_interpretation_allows_missing_new_gnn_checkpoints() -> None:
    rows = [
        {
            "split_strategy": strategy,
            "model_name": "xgb_aft",
            "feature_set": "edge_identity",
            "mean_test_c_index": score,
        }
        for strategy, score in (("group_disjoint", 0.72), ("random", 0.74))
    ]
    result = _diagnostic_interpretation(
        rows,
        {"42": {"five_seed": {"available": False}}},
        {"42": {"domain_classifier_auc": 0.80}},
        {"deterministic_event_time_risk_c_index": 0.76},
    )

    assert result["evidence"]["mean_existing_gnn_test_c_index"] is None
    assert result["evidence"]["group_best_minus_gnn_c_index"] is None
    assert (
        result["primary_diagnosis"]
        == "controlled_signal_recovered_pending_full_gnn_retraining"
    )


def test_controlled_group_and_random_splits_have_expected_behavior() -> None:
    config, frame, feature_sets = _load_frame()
    columns = ["sample_id", "time", "event", GROUP_COLUMN, *feature_sets["full_topology"]]
    controlled = frame[columns].copy()
    group = _group_split(
        controlled,
        seed=42,
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
    )
    random = _random_split(
        controlled,
        seed=42,
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
    )

    group_sets = [set(part[GROUP_COLUMN].astype(int)) for part in (group.train, group.val, group.test)]
    assert not group_sets[0].intersection(group_sets[1])
    assert not group_sets[0].intersection(group_sets[2])
    assert not group_sets[1].intersection(group_sets[2])
    assert all(len(set(part[GROUP_COLUMN].astype(int))) == 5 for part in (random.train, random.val, random.test))
    assert [len(part) for part in (group.train, group.val, group.test)] == [2160, 720, 720]
    assert [len(part) for part in (random.train, random.val, random.test)] == [2160, 720, 720]
