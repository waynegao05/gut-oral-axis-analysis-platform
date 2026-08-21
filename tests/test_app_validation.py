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
            "clinical": {"age": 50, "sex": "Female"},
            "metabolites": {},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["errors"] == ["No supported oral microbes were provided."]


def test_analyze_endpoint_uses_latest_ac_icam_v8_backend(monkeypatch) -> None:
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
            "clinical": {
                "age": 52,
                "sex": "Female",
                "stage": 3,
                "path_t": 3,
                "path_n": 1,
                "path_m": 0,
                "tumor_location": "Colon Sigmoideum",
                "tumor_morphology": "Adenocarcinoma",
                "bmi": 24.5,
                "smoking": 1,
            },
            "metabolites": {"bile_acids": 0.8, "scfa": 0.3},
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["risk_result"]["backend"] == "ac_icam_real_outcome_clinical_pfs"
    assert payload["risk_result"]["model_variant"] == "clinical_core"
    assert payload["report"]["gnn_features"]["microbiome_used_for_risk"] is False
    assert payload["pharmacy_assessment"]["engine_version"] == "pharmacy_assistance_v3"
    assert payload["pharmacy_assessment"]["status"] == "limited"
    assert payload["saved_to"] == "not-written.json"


def test_v8_analyze_endpoint_does_not_require_microbes_for_pfs_risk(
    monkeypatch,
) -> None:
    monkeypatch.setattr("enhanced_app.export_report", lambda _report: "not-written.json")
    client = app.test_client()
    response = client.post(
        "/analyze",
        json={
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
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["risk_result"]["prediction_available"] is True
    assert payload["risk_result"]["model_variant"] == "clinical_core"
    assert payload["report"]["top_microbes"] == []


def test_v8_analyze_endpoint_requires_age_and_sex() -> None:
    client = app.test_client()
    response = client.post(
        "/analyze",
        json={
            "microbes": {},
            "clinical": {"age": 62},
            "metabolites": {},
        },
    )

    assert response.status_code == 400
    message = " ".join(response.get_json()["errors"])
    assert "clinical.sex" in message


def test_v8_analyze_endpoint_allows_missing_oncology_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "enhanced_app.export_report",
        lambda _report: "not-written.json",
    )
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
            "clinical": {"age": 23, "sex": "Male"},
            "metabolites": {},
            "metadata": {
                "current_medications": [],
                "drug_allergies": [],
                "suspected_condition": "gut_risk_screening",
            },
        },
    )

    payload = response.get_json()
    risk = payload["risk_result"]
    assert response.status_code == 200
    assert risk["prediction_available"] is False
    assert risk["not_available_reason"] == "missing_oncology_fields"
    assert risk["risk_score"] is None
    assert risk["pfs_probability"] == {"36": None, "60": None}
    assert payload["report"]["gnn_features"]["defaulted_inputs"] == []
    general = payload["general_risk_result"]
    assert general == payload["report"]["general_risk_result"]
    assert general["prediction_available"] is True
    assert general["endpoint"] == "research_risk_index"
    assert general["absolute_cancer_probability"] is False
    assert 0.0 <= general["risk_percentile"] <= 100.0
    assert any(
        card["recommendation_id"].startswith("risk_review_")
        and "不是结直肠癌概率" in card["rationale"]
        for card in payload["recommendations"]
    )


def test_v8_general_risk_does_not_treat_missing_microbes_as_zero(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "enhanced_app.export_report",
        lambda _report: "not-written.json",
    )
    client = app.test_client()
    response = client.post(
        "/analyze",
        json={
            "microbes": {"Fusobacterium": 0.2},
            "clinical": {"age": 23, "sex": "Male"},
            "metabolites": {},
        },
    )

    payload = response.get_json()
    general = payload["general_risk_result"]
    assert response.status_code == 200
    assert general["prediction_available"] is False
    assert general["not_available_reason"] == "incomplete_microbiome_panel"
    assert general["risk_score"] is None
    assert any(
        card["recommendation_id"]
        == "pfs_not_calculated_missing_oncology"
        for card in payload["recommendations"]
    )


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
                "age": 18,
                "sex": "Female",
                "stage": 3,
                "path_t": 3,
                "path_n": 1,
                "path_m": 0,
                "tumor_location": "Colon Sigmoideum",
                "tumor_morphology": "Adenocarcinoma",
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
        "field": "clinical.age",
        "value": 18.0,
        "training_minimum": 25.0,
        "training_maximum": 88.0,
    }
    assert "年龄输入为 18" in assessment["quality"]["status_reasons"][0]["message"]
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
    assert 'id="risk-scale-marker"' in html
    assert 'id="risk-scale-fill"' in html
    assert 'id="clinical-stage"' in html
    assert 'id="clinical-tumor-location"' in html
    assert 'id="clinical-icr-score"' in html
    assert (
        'id="clinical-age" step="1" min="18" max="75" required'
        in html
    )
    assert '<select id="clinical-sex" required>' in html
    assert '<select id="clinical-stage" required>' not in html
    assert '<select id="clinical-path-t" required>' not in html
    assert '<select id="clinical-path-n" required>' not in html
    assert '<select id="clinical-tumor-location" required>' not in html
    assert '<select id="clinical-tumor-morphology" required>' not in html
    assert "没有癌症诊断或病理资料时可以全部留空" in html
    assert "五项核心菌群全部填写后" in html
    assert "缺失菌也不会被当作 0" in html
    assert 'id="pfs-probability-36"' in html
    assert "项待处理" in html
    assert "ac_icam_real_outcome_pfs_v8" in html
    assert "研究决策支持，不用于癌症筛查、诊断或替代医生制定治疗方案。" not in html
    assert "真实随访结局" in html
    assert "完整返回 JSON（供研究与审计）" in html
    assert '<details class="details-panel technical-record-panel" open>' not in html
