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
