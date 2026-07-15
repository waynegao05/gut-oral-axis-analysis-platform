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
