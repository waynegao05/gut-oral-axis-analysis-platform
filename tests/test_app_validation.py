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


def test_standardize_endpoint_rejects_non_object_raw_section() -> None:
    client = app.test_client()
    response = client.post(
        "/standardize",
        json={
            "history": ["invalid"],
            "oral_microbiome": {
                "taxa": [{"taxon": "Fusobacterium", "abundance": 0.2}]
            },
        },
    )

    assert response.status_code == 400
    assert response.get_json()["errors"] == ["history 必须是 JSON 对象。"]


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
    assert payload["pharmacy_assessment"]["engine_version"] == "pharmacy_assistance_v3"
    assert payload["pharmacy_assessment"]["status"] == "limited"
    assert payload["saved_to"] == "not-written.json"


def test_analyze_endpoint_explains_out_of_range_age_and_guideline_basis(
    monkeypatch,
) -> None:
    monkeypatch.setattr("enhanced_app.export_report", lambda _report: "not-written.json")
    client = app.test_client()
    response = client.post(
        "/analyze",
        json={
            "microbes": {
                "Fusobacterium": 0.18,
                "Porphyromonas": 0.15,
                "Prevotella": 0.10,
                "Streptococcus": 0.09,
                "Lactobacillus": 0.02,
            },
            "clinical": {
                "age": 120,
                "bmi": 24.5,
                "smoking": 1,
                "family_history": 1,
            },
            "metabolites": {
                "bile_acids": 0.8,
                "scfa": 0.3,
                "tryptophan_metabolism": 0.7,
            },
            "metadata": {
                "current_medications": [],
                "drug_allergies": [],
                "recent_antibiotics": 0,
                "recent_probiotics": 0,
                "renal_impairment": 0,
                "hepatic_impairment": 0,
                "pregnancy": 0,
                "suspected_condition": "irritable_bowel_syndrome",
            },
        },
    )

    payload = response.get_json()
    assessment = payload["pharmacy_assessment"]
    age_detail = payload["report"]["gnn_features"]["out_of_training_range_details"][0]
    probiotic_card = next(
        card
        for card in assessment["recommendations"]
        if card["category"] == "probiotic_evidence"
    )

    assert response.status_code == 200
    assert assessment["status"] == "withheld"
    assert age_detail == {
        "field": "age",
        "value": 120.0,
        "training_minimum": 22.0,
        "training_maximum": 84.0,
        "affected_split_seeds": [42, 43],
    }
    assert "年龄输入为 120" in assessment["quality"]["status_reasons"][0]["message"]
    assert probiotic_card["independent_of_model_result"] is True


def test_standardize_endpoint_rejects_invalid_medication_metadata() -> None:
    client = app.test_client()
    response = client.post(
        "/standardize",
        json={
            "microbes": {"Fusobacterium": 0.2},
            "clinical": {},
            "metabolites": {},
            "metadata": {"drug_allergies": "penicillin", "pregnancy": 3},
        },
    )

    assert response.status_code == 400
    errors = response.get_json()["errors"]
    assert any("metadata.drug_allergies" in error for error in errors)
    assert any("metadata.pregnancy" in error for error in errors)


def test_standardize_endpoint_rejects_negative_medication_quantity() -> None:
    client = app.test_client()
    response = client.post(
        "/standardize",
        json={
            "microbes": {"Fusobacterium": 0.2},
            "clinical": {},
            "metabolites": {},
            "metadata": {"current_medications": ["metformin -500 mg"]},
        },
    )

    assert response.status_code == 400
    assert any("负数剂量或规格" in error for error in response.get_json()["errors"])


def test_index_contains_drug_knowledge_and_label_evidence_panels() -> None:
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="drug-knowledge-coverage"' in html
    assert 'id="medication-label-list"' in html
    assert 'id="metadata-suspected-condition"' in html
    assert 'id="pharmacy-now-list"' in html
    assert 'id="priority-recommendation-list"' in html
    assert 'id="routine-recommendation-list"' in html
    assert 'id="risk-kicker"' in html
    assert "项待处理" in html
    assert "风险提示 + 用药核对 + 下一步行动" in html
    assert "完整返回 JSON（供研究与审计）" in html
    assert '<details class="details-panel technical-record-panel" open>' not in html
