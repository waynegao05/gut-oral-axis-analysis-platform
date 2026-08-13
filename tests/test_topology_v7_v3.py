from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "data" / "research"
V3_ROOT = V2_ROOT / "topology_v7_generator_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> dict:
    return json.loads((V3_ROOT / "topology_v7_manifest.json").read_text(encoding="utf-8"))


def test_v3_is_independent_and_keeps_v2_as_its_parent() -> None:
    v2_manifest_path = V2_ROOT / "topology_v7_manifest.json"
    v2_manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    v3_manifest = _manifest()

    assert v2_manifest["generator_version"] == "topology_v7_hybrid_generator_v2"
    assert v3_manifest["generator_version"] == "topology_v7_hybrid_generator_v3"
    assert v3_manifest["sample_count"] == 3600
    assert v3_manifest["parent_v2_manifest_sha256"] == _sha256(v2_manifest_path)
    assert all("topology_v7_generator_v3" in path for path in v3_manifest["outputs"])


def test_v3_has_consistent_recoverable_signal_in_every_generation_group() -> None:
    survival = _manifest()["generation"]["survival"]
    group_values = list(survival["generation_group_latent_risk_c_index"].values())

    assert 0.72 <= survival["deterministic_latent_risk_c_index"] <= 0.80
    assert min(group_values) >= 0.68
    assert max(group_values) - min(group_values) <= 0.10
    assert 1.10 <= survival["event_noise_to_signal_sd_ratio"] <= 1.40
    assert survival["event_noise_sampling"] == "independent_pseudorandom_normal"
    assert survival["censor_noise_sampling"] == "independent_pseudorandom_normal"
    assert survival["censor_location_mode"] == "analytic_prior_calibration"
    assert survival["generation_group_used_for_outcome_generation"] is False
    assert abs(
        survival["generated_event_rate"] - survival["target_event_rate_from_v6"]
    ) <= 0.05


def test_v3_harmonization_is_outcome_blind_and_uses_frozen_calibration() -> None:
    microbiome = _manifest()["generation"]["microbiome"]

    assert microbiome["anchor_balance"]["method"] == (
        "outcome_blind_randomized_balanced_partition_search"
    )
    assert microbiome["target_balanced_within_generation_group"] is True
    assert microbiome["local_anchor_weight_range"] == [0.55, 0.55]
    assert 0.95 <= microbiome["approximate_class_covariance_fraction"] <= 1.05
    assert microbiome["quantile_calibration"] == "frozen_public_anchor_to_v6_reference"


def test_v3_audit_only_survival_columns_do_not_leak_into_model_tables() -> None:
    manifest = _manifest()
    prohibited = set(manifest["prohibited_model_features"])
    provenance = pd.read_csv(V3_ROOT / "topology_v7_sample_provenance.csv", nrows=1)
    graph = pd.read_csv(V3_ROOT / "topology_v7_sample_graph_table.csv", nrows=1)
    clinical = pd.read_csv(V3_ROOT / "topology_v7_sample_clinical_table.csv", nrows=1)
    metabolite = pd.read_csv(V3_ROOT / "topology_v7_sample_metabolite_table.csv", nrows=1)

    assert prohibited.issubset(provenance.columns)
    assert prohibited.isdisjoint(graph.columns)
    assert prohibited.isdisjoint(clinical.columns)
    assert prohibited.isdisjoint(metabolite.columns)
