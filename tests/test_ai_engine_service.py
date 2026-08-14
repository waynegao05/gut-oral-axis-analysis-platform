from __future__ import annotations

import pytest

from ai_engine.errors import (
    CapabilityUnavailableError,
    InvalidInputError,
    ModelNotLoadedError,
)
from ai_engine.runtime import validate_bind_host
from ai_engine.service import AIService
from src.pipeline import run_pipeline


def _complete_v8_payload() -> dict[str, object]:
    return {
        "microbes": {},
        "clinical": {
            "age": 62,
            "sex": "Female",
            "stage": 3,
            "path_t": 3,
            "path_n": 1,
            "path_m": 0,
            "tumor_location": "Colon Sigmoideum",
            "tumor_morphology": "Adenocarcinoma",
        },
        "metabolites": {},
        "metadata": {},
    }


def _minimal_report() -> dict[str, object]:
    return {
        "top_microbes": [],
        "gnn_features": {"edge_count": 0},
        "risk_result": {"prediction_available": True, "risk_score": 0.71},
        "general_risk_result": {},
        "recommendations": [],
        "pharmacy_assessment": {"status": "limited"},
    }


def test_service_matches_current_pipeline_for_complete_v8_input() -> None:
    payload = _complete_v8_payload()
    service = AIService()

    assert service.initialize() is True
    service_result = service.analyze(payload)
    current_result = run_pipeline(payload)

    assert service_result["report"] == current_result
    assert service_result["risk_result"] == current_result["risk_result"]
    assert service_result["standardized_payload"] == payload
    assert "saved_to" not in service_result


def test_service_rejects_invalid_input_with_structured_details() -> None:
    service = AIService(model_loader=lambda: object())
    with pytest.raises(InvalidInputError) as error:
        service.standardize(
            {
                "microbes": {},
                "clinical": {"age": -1, "sex": "Female"},
                "metabolites": {},
            }
        )

    assert error.value.error_code == "INVALID_INPUT"
    assert any("clinical.age" in item["message"] for item in error.value.details)


def test_service_translates_missing_optional_model_dependency() -> None:
    def missing_dependency(_: dict[str, object]) -> dict[str, object]:
        raise ModuleNotFoundError("No module named 'private_dependency'")

    service = AIService(
        model_loader=lambda: object(),
        pipeline_runner=missing_dependency,
    )
    assert service.initialize() is True

    with pytest.raises(CapabilityUnavailableError) as error:
        service.analyze(_complete_v8_payload())

    assert error.value.error_code == "CAPABILITY_UNAVAILABLE"
    assert "private_dependency" not in error.value.public_message


def test_health_reports_primary_and_optional_capabilities() -> None:
    service = AIService(model_loader=lambda: object())
    assert service.initialize() is True

    health = service.health()

    assert health["engine_ready"] is True
    assert health["model_loaded"] is True
    assert health["model_versions"]["pfs"] == "ac_icam_real_outcome_pfs_v8"
    assert set(health["capabilities"]) == {
        "primary_model",
        "pfs",
        "general_risk",
        "pharmacy",
        "oral_adenoma",
    }


def test_health_records_successful_lazy_general_risk_load() -> None:
    report = _minimal_report()
    report["general_risk_result"] = {
        "prediction_available": True,
        "endpoint": "research_risk_index",
    }
    service = AIService(
        model_loader=lambda: object(),
        pipeline_runner=lambda _: report,
    )

    service.analyze(_complete_v8_payload())

    general_status = service.health()["capabilities"]["general_risk"]
    assert general_status["available"] is True
    assert general_status["loaded"] is True


def test_model_initialization_failure_keeps_health_available_but_blocks_analysis() -> (
    None
):
    def fail_load() -> object:
        raise RuntimeError("broken model artifact")

    service = AIService(model_loader=fail_load)

    assert service.initialize() is False
    assert service.health()["model_loaded"] is False
    with pytest.raises(ModelNotLoadedError):
        service.analyze(_complete_v8_payload())


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "[::1]", "localhost"])
def test_runtime_accepts_only_loopback_hosts(host: str) -> None:
    assert validate_bind_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "example.com"])
def test_runtime_rejects_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_bind_host(host)
