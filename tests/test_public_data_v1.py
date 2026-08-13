from __future__ import annotations

from pathlib import Path

from experiments.public_data_v1.prepare_gse217490 import _taxon_label
from experiments.public_data_v1.registry import build_audit_report, load_registry


ROOT = Path(__file__).resolve().parents[1]


def _assessment(report: dict, dataset_id: str) -> dict:
    return next(item for item in report["assessments"] if item["id"] == dataset_id)


def test_registry_has_no_false_full_replacement_claim() -> None:
    report = build_audit_report(load_registry())

    assert report["strict_full_replacement_available"] is False
    assert report["strict_full_replacement_dataset_ids"] == []
    assert report["cross_cohort_patient_join_allowed"] is False


def test_russo_cohort_is_oral_gut_external_validation_not_survival() -> None:
    report = build_audit_report(load_registry())
    result = _assessment(report, "russo_crc_oral_gut_2023")

    assert result["download_ready"] is True
    assert result["exact_right_censored_survival"] is False
    assert result["missing_full_contract"] == [
        "metabolomics",
        "right_censored_time_to_event",
    ]


def test_colocare_is_survival_core_only_after_access() -> None:
    report = build_audit_report(load_registry())
    result = _assessment(report, "colocare_crc_dfs_2025")

    assert result["survival_core_satisfied"] is True
    assert result["survival_core_ready"] is False
    assert result["access_mode"] == "request_required"
    assert result["missing_full_contract"] == ["metabolomics", "oral_microbiome"]


def test_topology_v7_is_training_mainline_and_v6_is_frozen_for_deployment() -> None:
    config = (ROOT / "research_config_v2.yaml").read_text(encoding="utf-8")
    deployment_config = (ROOT / "config/releases/temporal_topology_v6.yaml").read_text(
        encoding="utf-8"
    )

    assert "data/research/topology_v7_sample_graph_table.csv" in config
    assert "data/research/topology_v7_sample_clinical_table.csv" in config
    assert "data/research/topology_v7_sample_metabolite_table.csv" in config
    assert "data/research/topology_v7_sample_label_table.csv" in config
    assert "archive/datasets/topology_v6/topology_v6_sample_graph_table.csv" in deployment_config


def test_v7_policy_keeps_generated_survival_development_only() -> None:
    registry = load_registry()
    policy = registry["safety_policy"]

    assert registry["current_dataset_policy"]["dataset_id"] == "topology_v7"
    assert policy["synthetic_survival_label_creation_allowed"] is True
    assert policy["synthetic_survival_label_use"] == "development_only"
    assert policy["synthetic_survival_claim_as_observed"] is False
    assert policy["generated_descendants_must_use_group_disjoint_splits"] is True


def test_public_taxonomy_uses_deepest_informative_rank() -> None:
    assert (
        _taxon_label("d__Bacteria; f__Fusobacteriaceae; g__Fusobacterium; s__uncultured")
        == "genus:Fusobacterium"
    )
    assert (
        _taxon_label("d__Bacteria; f__Porphyromonadaceae; g__uncultured_bacterium")
        == "family:Porphyromonadaceae"
    )
