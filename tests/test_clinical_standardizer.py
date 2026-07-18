from __future__ import annotations

import pytest

from src.clinical_standardizer import standardize_raw_payload
from src.validators import validate_payload


def test_standardizer_rejects_non_numeric_value_instead_of_replacing_with_zero() -> None:
    payload = {
        "demographics": {"age": "not-a-number"},
        "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]},
    }

    with pytest.raises(ValueError, match="demographics.age"):
        standardize_raw_payload(payload)


def test_standardizer_omits_missing_optional_numeric_fields() -> None:
    payload = {
        "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]},
    }

    standardized = standardize_raw_payload(payload)

    assert standardized["clinical"] == {}
    assert standardized["metabolites"] == {}


def test_standardizer_preserves_negative_value_for_canonical_validation() -> None:
    payload = {
        "demographics": {"age": -1},
        "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": -0.2}]},
    }

    standardized = standardize_raw_payload(payload)
    ok, errors = validate_payload(standardized)

    assert ok is False
    assert any("clinical.age" in error for error in errors)
    assert any("microbes.Fusobacterium" in error for error in errors)


def test_standardizer_rejects_unknown_binary_text() -> None:
    payload = {
        "history": {"smoking": "sometimes maybe"},
        "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]},
    }

    with pytest.raises(ValueError, match="history.smoking"):
        standardize_raw_payload(payload)


def test_standardizer_rejects_non_object_raw_section() -> None:
    payload = {
        "history": ["not", "an", "object"],
        "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]},
    }

    with pytest.raises(ValueError, match="history 必须是 JSON 对象"):
        standardize_raw_payload(payload)


def test_standardizer_preserves_structured_medication_context() -> None:
    payload = {
        "demographics": {"pregnancy": "no"},
        "history": {
            "recent_antibiotics": "yes",
            "recent_probiotics": "no",
        },
        "medication_context": {
            "current_medications": "metformin 500 mg, vitamin D",
            "drug_allergies": ["penicillin: rash"],
            "renal_impairment": "no",
            "hepatic_impairment": "yes",
        },
        "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]},
    }

    standardized = standardize_raw_payload(payload)
    metadata = standardized["metadata"]

    assert metadata["current_medications"] == ["metformin 500 mg", "vitamin D"]
    assert metadata["drug_allergies"] == ["penicillin: rash"]
    assert metadata["recent_antibiotics"] == 1.0
    assert metadata["recent_probiotics"] == 0.0
    assert metadata["renal_impairment"] == 0.0
    assert metadata["hepatic_impairment"] == 1.0
    assert metadata["pregnancy"] == 0.0


def test_standardizer_treats_explicit_none_as_reported_empty_list() -> None:
    payload = {
        "medication_context": {
            "current_medications": "无",
            "drug_allergies": "no known allergies",
        },
        "oral_microbiome": {
            "taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]
        },
    }

    standardized = standardize_raw_payload(payload)

    assert standardized["metadata"]["current_medications"] == []
    assert standardized["metadata"]["drug_allergies"] == []


def test_validator_rejects_invalid_canonical_medication_context() -> None:
    payload = {
        "microbes": {"Fusobacterium": 0.2},
        "clinical": {},
        "metabolites": {},
        "metadata": {
            "current_medications": "not-a-list",
            "renal_impairment": 2,
        },
    }

    ok, errors = validate_payload(payload)

    assert ok is False
    assert any("metadata.current_medications" in error for error in errors)
    assert any("metadata.renal_impairment" in error for error in errors)
