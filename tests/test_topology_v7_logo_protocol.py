from pathlib import Path

import pandas as pd
import yaml

from experiments.temporal_independent_v3.topology_aft_fusion import (
    build_topology_fingerprint_dataframe,
)
from experiments.topology_v7_diagnosis.diagnose import GROUP_COLUMN, _feature_frame
from experiments.topology_v7_generator_v3.logo_benchmark import (
    _config_for_data_dir,
    _explicit_logo_split,
    _lineage_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def test_logo_protocol_covers_each_group_once_without_outer_test_leakage() -> None:
    template = yaml.safe_load(
        (ROOT / "config/research/research_config_v7_gnn_optimized.yaml").read_text(encoding="utf-8")
    )
    config = _config_for_data_dir(template, ROOT / "data" / "research")
    frame, feature_sets, _ = _feature_frame(config)

    observed_test_groups = []
    for test_group in range(5):
        validation_group = (test_group + 1) % 5
        split = _explicit_logo_split(
            frame,
            feature_sets["full_topology"],
            test_group=test_group,
            validation_group=validation_group,
        )
        train_groups = set(split.train[GROUP_COLUMN].astype(int))
        val_groups = set(split.val[GROUP_COLUMN].astype(int))
        test_groups = set(split.test[GROUP_COLUMN].astype(int))
        assert test_groups == {test_group}
        assert val_groups == {validation_group}
        assert train_groups == set(range(5)).difference(test_groups | val_groups)
        observed_test_groups.extend(test_groups)

    assert sorted(observed_test_groups) == list(range(5))


def test_logo_domain_audit_keeps_each_public_anchor_in_one_generation_group() -> None:
    provenance = pd.read_csv(ROOT / "data/research/topology_v7_sample_provenance.csv")
    audit = _lineage_audit(provenance)

    assert audit["all_descendants_of_each_anchor_stay_in_one_group"] is True
    assert audit["public_anchor_overlap_between_generation_groups"] == {}


def test_direct_aft_fingerprint_never_exposes_generation_group_as_a_feature() -> None:
    template = yaml.safe_load(
        (ROOT / "config/research/research_config_v7_v3_gnn_locked.yaml").read_text(encoding="utf-8")
    )
    frame, feature_columns, _ = build_topology_fingerprint_dataframe(template)

    assert GROUP_COLUMN in frame.columns
    assert GROUP_COLUMN not in feature_columns
