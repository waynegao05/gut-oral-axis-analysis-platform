from src.pipeline import run_pipeline
from src.validators import validate_payload


def test_validate_payload_success():
    payload = {
        "microbes": {"Fusobacterium": 0.1},
        "clinical": {"age": 50},
        "metabolites": {"bile_acids": 0.3},
    }
    ok, errors = validate_payload(payload)
    assert ok is True
    assert errors == []


def test_validate_payload_rejects_negative_microbe_abundance():
    payload = {
        "microbes": {"Fusobacterium": -0.1},
        "clinical": {"age": 50},
        "metabolites": {"bile_acids": 0.3},
    }

    ok, errors = validate_payload(payload)

    assert ok is False
    assert any("microbes.Fusobacterium" in error for error in errors)


def test_validate_payload_rejects_invalid_age_and_binary_fields():
    payload = {
        "microbes": {"Fusobacterium": 0.1},
        "clinical": {"age": -5, "smoking": 3},
        "metabolites": {},
    }

    ok, errors = validate_payload(payload)

    assert ok is False
    assert any("clinical.age" in error for error in errors)
    assert any("clinical.smoking" in error for error in errors)


def test_validate_payload_rejects_non_finite_and_out_of_range_values():
    payload = {
        "microbes": {"Fusobacterium": float("nan")},
        "clinical": {"bmi": float("inf")},
        "metabolites": {"scfa": 1.5},
    }

    ok, errors = validate_payload(payload)

    assert ok is False
    assert any("NaN 或 Infinity" in error for error in errors)
    assert any("metabolites.scfa" in error for error in errors)


def test_validate_payload_rejects_all_zero_microbes():
    payload = {
        "microbes": {"Fusobacterium": 0.0, "Prevotella": 0.0},
        "clinical": {},
        "metabolites": {},
    }

    ok, errors = validate_payload(payload)

    assert ok is False
    assert any("至少需要一个大于 0" in error for error in errors)


def test_pipeline_output_keys():
    payload = {
        "microbes": {
            "Fusobacterium": 0.18,
            "Porphyromonas": 0.15,
            "Prevotella": 0.10,
        },
        "clinical": {
            "age": 52,
            "bmi": 24.5,
            "smoking": 1,
        },
        "metabolites": {
            "bile_acids": 0.8,
            "scfa": 0.3,
        },
    }
    report = run_pipeline(payload)
    assert "top_microbes" in report
    assert "gnn_features" in report
    assert "risk_result" in report
    assert "recommendations" in report
