from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from ai_engine.api import create_app
from ai_engine.service import AIService


TOKEN = "local-test-token-0123456789-abcdef"
HEADERS = {"X-GOA-Engine-Token": TOKEN}


def _payload() -> dict[str, object]:
    return {
        "microbes": {},
        "clinical": {"age": 52, "sex": "Female"},
        "metabolites": {},
    }


def _report() -> dict[str, object]:
    return {
        "top_microbes": [],
        "gnn_features": {"edge_count": 0},
        "risk_result": {"prediction_available": False, "risk_score": None},
        "general_risk_result": {},
        "recommendations": [],
        "pharmacy_assessment": {"status": "limited"},
    }


def _app(*, runner=None):
    service = AIService(
        model_loader=lambda: object(),
        pipeline_runner=runner or (lambda _: _report()),
    )
    return create_app(service=service, engine_token=TOKEN)


def test_health_requires_the_per_launch_token() -> None:
    with TestClient(_app()) as client:
        rejected = client.get("/api/v1/health")
        accepted = client.get("/api/v1/health", headers=HEADERS)

    assert rejected.status_code == 401
    assert rejected.json()["error_code"] == "AUTHENTICATION_FAILED"
    assert accepted.status_code == 200
    assert accepted.json()["model_loaded"] is True
    assert accepted.headers["X-Request-ID"] == accepted.json()["request_id"]


@pytest.mark.parametrize("token", [None, "short-token"])
def test_app_factory_rejects_missing_or_short_tokens(
    monkeypatch,
    token: str | None,
) -> None:
    monkeypatch.delenv("GOA_ENGINE_TOKEN", raising=False)
    service = AIService(model_loader=lambda: object())
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        create_app(service=service, engine_token=token)


def test_analyze_returns_structured_result_without_writing_a_report_path() -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=HEADERS,
            json=_payload(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["report"] == _report()
    assert "saved_to" not in payload


def test_standardize_uses_the_existing_validation_contract() -> None:
    invalid = _payload()
    invalid["clinical"] = {"age": -5, "sex": "Female"}

    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/standardize",
            headers=HEADERS,
            json=invalid,
        )

    payload = response.json()
    assert response.status_code == 400
    assert payload["error_code"] == "INVALID_INPUT"
    assert any("clinical.age" in item["message"] for item in payload["details"])


def test_api_does_not_expose_internal_exception_text() -> None:
    def fail(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("PRIVATE_INTERNAL_PATH_AND_DETAILS")

    with TestClient(_app(runner=fail), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=HEADERS,
            json=_payload(),
        )

    payload = response.json()
    assert response.status_code == 500
    assert payload["error_code"] == "MODEL_INFERENCE_FAILED"
    assert "PRIVATE_INTERNAL_PATH_AND_DETAILS" not in payload["message"]


def test_api_rejects_non_object_analysis_payload() -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=HEADERS,
            json=[1, 2, 3],
        )

    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_api_rejects_chunked_body_using_actual_byte_count() -> None:
    app = create_app(
        service=AIService(model_loader=lambda: object()),
        engine_token=TOKEN,
        max_request_bytes=64,
    )

    def chunks():
        yield b'{"microbes":{"Fusobacterium":"'
        yield b"x" * 80
        yield b'"}}'

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            headers={**HEADERS, "Transfer-Encoding": "chunked"},
            content=chunks(),
        )

    assert response.status_code == 413
    assert response.json()["error_code"] == "REQUEST_TOO_LARGE"
