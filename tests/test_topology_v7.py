from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research.data import split_sample_table
from research.metrics import concordance_index
from research.task import infer_dataset_origin


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "research"
ARCHIVE_ROOT = ROOT / "archive" / "datasets" / "topology_v6"
ARCHIVE_V7_V1_ROOT = ROOT / "archive" / "datasets" / "topology_v7_generator_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline=None) as handle:
        return hashlib.sha256(handle.read().encode("utf-8")).hexdigest()


def test_v7_manifest_declares_generated_not_observed_cohort() -> None:
    manifest = json.loads((DATA_ROOT / "topology_v7_manifest.json").read_text(encoding="utf-8"))

    assert manifest["dataset_version"] == "topology_v7"
    assert manifest["generator_version"] == "topology_v7_hybrid_generator_v2"
    assert manifest["sample_count"] == 3600
    assert manifest["observed_real_patient_count"] == 42
    assert manifest["observed_real_patient_rows_in_v7"] == 0
    assert manifest["dataset_class"] == "hybrid_model_generated_development_cohort"
    assert manifest["quality"]["exact_duplicate_abundance_vectors"] == 0
    assert manifest["quality"]["anchor_overlap_between_generation_groups"] == []
    assert any("not 3600 observed patients" in text for text in manifest["limitations"])


def test_v7_survival_generator_has_controlled_recoverable_signal() -> None:
    manifest = json.loads((DATA_ROOT / "topology_v7_manifest.json").read_text(encoding="utf-8"))
    survival = manifest["generation"]["survival"]

    assert survival["method"] == "controlled_multimodal_log_normal_aft_with_independent_censoring"
    assert 0.72 <= survival["deterministic_latent_risk_c_index"] <= 0.80
    assert survival["minimum_generation_group_latent_risk_c_index"] >= 0.68
    assert 1.10 <= survival["event_noise_to_signal_sd_ratio"] <= 1.40
    assert abs(survival["censor_log_time_latent_risk_correlation"]) <= 0.08
    assert survival["realized_hidden_event_time_oracle_c_index"] == 1.0


def test_v7_audit_provenance_reconstructs_labels_without_feature_leakage() -> None:
    manifest = json.loads((DATA_ROOT / "topology_v7_manifest.json").read_text(encoding="utf-8"))
    provenance = pd.read_csv(DATA_ROOT / "topology_v7_sample_provenance.csv")
    labels = pd.read_csv(DATA_ROOT / "topology_v7_sample_label_table.csv")
    clinical = pd.read_csv(DATA_ROOT / "topology_v7_sample_clinical_table.csv")
    metabolite = pd.read_csv(DATA_ROOT / "topology_v7_sample_metabolite_table.csv")
    graph = pd.read_csv(DATA_ROOT / "topology_v7_sample_graph_table.csv")
    prohibited = set(manifest["prohibited_model_features"])

    assert prohibited.issubset(provenance.columns)
    assert prohibited.isdisjoint(clinical.columns)
    assert prohibited.isdisjoint(metabolite.columns)
    assert prohibited.isdisjoint(graph.columns)

    merged = labels.merge(provenance, on="sample_id", how="inner", validate="one_to_one")
    reconstructed_event = (
        merged["survival_event_time"] <= merged["survival_censor_time"]
    ).astype(int)
    reconstructed_time = (
        merged[["survival_event_time", "survival_censor_time"]]
        .min(axis=1)
        .round()
        .clip(6, 132)
        .astype(int)
    )
    assert reconstructed_event.equals(merged["event"].astype(int))
    assert reconstructed_time.equals(merged["time"].astype(int))
    assert 0.72 <= concordance_index(
        merged["time"], merged["event"], merged["survival_latent_risk"]
    ) <= 0.80


def test_v7_tables_are_complete_unique_and_range_valid() -> None:
    graph = pd.read_csv(DATA_ROOT / "topology_v7_sample_graph_table.csv")
    clinical = pd.read_csv(DATA_ROOT / "topology_v7_sample_clinical_table.csv")
    metabolite = pd.read_csv(DATA_ROOT / "topology_v7_sample_metabolite_table.csv")
    label = pd.read_csv(DATA_ROOT / "topology_v7_sample_label_table.csv")
    provenance = pd.read_csv(DATA_ROOT / "topology_v7_sample_provenance.csv")

    assert len(graph) == 36000
    assert graph["sample_id"].nunique() == 3600
    assert len(clinical) == len(metabolite) == len(label) == len(provenance) == 3600
    assert clinical["sample_id"].is_unique
    assert metabolite["sample_id"].is_unique
    assert label["sample_id"].is_unique
    assert provenance["sample_id"].is_unique
    assert graph[["abundance", "function_score", "edge_weight"]].notna().all().all()
    assert graph["abundance"].between(0.0, 1.0).all()
    assert graph["function_score"].between(0.0, 1.0).all()
    assert graph["edge_weight"].between(0.0, 1.0).all()
    assert metabolite.drop(columns="sample_id").apply(lambda values: values.between(0.0, 1.0)).all().all()
    assert clinical["age"].between(1.0, 120.0).all()
    assert clinical["bmi"].between(5.0, 100.0).all()
    assert set(label["event"].unique()) == {0, 1}
    assert (label["time"] > 0).all()


def test_v7_generation_groups_are_disjoint_during_split() -> None:
    clinical = pd.read_csv(DATA_ROOT / "topology_v7_sample_clinical_table.csv")
    label = pd.read_csv(DATA_ROOT / "topology_v7_sample_label_table.csv")
    sample = clinical.merge(label, on="sample_id", how="inner")

    train, val, test, summary = split_sample_table(sample, seed=42, val_ratio=0.2, test_ratio=0.2)
    train_groups = set(train["generation_group_id"])
    val_groups = set(val["generation_group_id"])
    test_groups = set(test["generation_group_id"])

    assert summary["split_strategy"] == "generation_group_disjoint_train_val_test_split"
    assert not train_groups.intersection(val_groups)
    assert not train_groups.intersection(test_groups)
    assert not val_groups.intersection(test_groups)
    assert len(train) == 2160
    assert len(val) == 720
    assert len(test) == 720


def test_v7_explicit_logo_split_keeps_outer_test_group_hidden() -> None:
    clinical = pd.read_csv(DATA_ROOT / "topology_v7_sample_clinical_table.csv")
    label = pd.read_csv(DATA_ROOT / "topology_v7_sample_label_table.csv")
    sample = clinical.merge(label, on="sample_id", how="inner")

    train, val, test, summary = split_sample_table(
        sample,
        seed=42,
        val_ratio=0.2,
        test_ratio=0.2,
        validation_group=3,
        test_group=1,
    )

    assert summary["split_strategy"] == "generation_group_explicit_logo_train_val_test_split"
    assert set(train["generation_group_id"].astype(int)) == {0, 2, 4}
    assert set(val["generation_group_id"].astype(int)) == {3}
    assert set(test["generation_group_id"].astype(int)) == {1}
    assert [len(train), len(val), len(test)] == [2160, 720, 720]


def test_v7_explicit_logo_split_rejects_incomplete_or_overlapping_groups() -> None:
    clinical = pd.read_csv(DATA_ROOT / "topology_v7_sample_clinical_table.csv")
    label = pd.read_csv(DATA_ROOT / "topology_v7_sample_label_table.csv")
    sample = clinical.merge(label, on="sample_id", how="inner")

    import pytest

    with pytest.raises(ValueError, match="provided together"):
        split_sample_table(
            sample,
            seed=42,
            val_ratio=0.2,
            test_ratio=0.2,
            test_group=1,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        split_sample_table(
            sample,
            seed=42,
            val_ratio=0.2,
            test_ratio=0.2,
            validation_group=1,
            test_group=1,
        )


def test_v6_archive_matches_preserved_source_files() -> None:
    names = [
        "topology_v6_sample_graph_table.csv",
        "topology_v6_sample_clinical_table.csv",
        "topology_v6_sample_metabolite_table.csv",
        "topology_v6_sample_label_table.csv",
    ]
    for name in names:
        assert _normalized_text_sha256(DATA_ROOT / name) == _normalized_text_sha256(
            ARCHIVE_ROOT / name
        )


def test_previous_v7_generator_is_preserved_for_rollback() -> None:
    current_manifest = json.loads(
        (DATA_ROOT / "topology_v7_manifest.json").read_text(encoding="utf-8")
    )
    archived_manifest = json.loads(
        (ARCHIVE_V7_V1_ROOT / "topology_v7_manifest.json").read_text(encoding="utf-8")
    )
    archive_index = json.loads(
        (ARCHIVE_V7_V1_ROOT / "archive_manifest.json").read_text(encoding="utf-8")
    )

    assert archived_manifest["generator_version"] == "topology_v7_hybrid_generator_v1"
    assert archive_index["generator_version"] == "topology_v7_hybrid_generator_v1"
    assert archive_index["local_files_preserved"] is True
    assert current_manifest["previous_v7_archive"] == archive_index["files"]
    for original_path, expected_hash in archived_manifest["outputs"].items():
        archived_path = ARCHIVE_V7_V1_ROOT / Path(original_path).name
        assert archived_path.exists()
        assert _sha256(archived_path) == expected_hash


def test_dataset_origin_exposes_v7_generation_boundaries() -> None:
    origin = infer_dataset_origin("data/research/topology_v7_sample_graph_table.csv")

    assert origin["dataset_version"] == "topology_v7"
    assert origin["is_synthetic"] is True
    assert origin["uses_real_cohort_anchors"] is True
    assert origin["contains_model_generated_survival_labels"] is True
    assert origin["is_external_clinical_validation"] is False
