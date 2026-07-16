from __future__ import annotations

from enhanced_app import app


def test_standardize_endpoint_rejects_negative_canonical_age() -> None:
    client = app.test_client()
    response = client.post(
        "/standardize",
        json={
            "microbes": {"Fusobacterium": 0.2},
            "clinical": {"age": -2},
            "metabolites": {},
        },
    )

    assert response.status_code == 400
    assert any("clinical.age" in error for error in response.get_json()["errors"])


def test_standardize_endpoint_rejects_non_numeric_raw_age() -> None:
    client = app.test_client()
    response = client.post(
        "/standardize",
        json={
            "demographics": {"age": "invalid"},
            "oral_microbiome": {"taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]},
        },
    )

    assert response.status_code == 400
    assert any("demographics.age" in error for error in response.get_json()["errors"])


def test_analyze_endpoint_returns_400_for_pipeline_input_error(monkeypatch) -> None:
    def reject_input(_payload):
        raise ValueError("No supported oral microbes were provided.")

    monkeypatch.setattr("enhanced_app.run_pipeline", reject_input)
    client = app.test_client()
    response = client.post(
        "/analyze",
        json={
            "microbes": {"UnknownMicrobe": 0.2},
            "clinical": {"age": 50},
            "metabolites": {},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["errors"] == ["No supported oral microbes were provided."]


def test_analyze_endpoint_uses_latest_temporal_topology_backend(monkeypatch) -> None:
    monkeypatch.setattr("enhanced_app.export_report", lambda _report: "not-written.json")
    client = app.test_client()
    response = client.post(
        "/analyze",
        json={
            "microbes": {
                "Fusobacterium": 0.18,
                "Porphyromonas": 0.15,
                "Prevotella": 0.10,
            },
            "clinical": {"age": 52, "bmi": 24.5, "smoking": 1},
            "metabolites": {"bile_acids": 0.8, "scfa": 0.3},
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["risk_result"]["backend"] == "temporal_topology_aft_cross_split_consensus"
    assert payload["report"]["gnn_features"]["topology_source"] == "inferred_from_web_inputs"
    assert payload["saved_to"] == "not-written.json"
