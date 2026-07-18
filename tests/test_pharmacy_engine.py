from __future__ import annotations

from copy import deepcopy

import pytest

from research.rebuild_pharmacy_calibration_v2 import (
    DEFAULT_GRAPH_TABLE,
    expected_rule_thresholds,
    verify_thresholds,
)
from src.clinical_report_builder import build_clinical_report
from src.pharmacy_advice import build_pharmacy_assistance
from src.pharmacy_engine import (
    _validate_knowledge_payload,
    build_pharmacy_assessment,
    load_pharmacy_knowledge_base,
)


FULL_PANEL = {
    "Fusobacterium": 0.35,
    "Porphyromonas": 0.15,
    "Prevotella": 0.15,
    "Streptococcus": 0.20,
    "Lactobacillus": 0.15,
}


def _features(**overrides):
    features = {
        "supported_microbe_input": {
            "Fusobacterium": 0.80,
            "Porphyromonas": 0.45,
            "Prevotella": 0.50,
            "Streptococcus": 0.55,
            "Lactobacillus": 0.40,
        },
        "defaulted_inputs": [],
        "out_of_training_range_inputs": [],
        "unsupported_microbes_ignored": [],
    }
    features.update(overrides)
    return features


def _risk(**overrides):
    risk = {
        "risk_score": 70.0,
        "risk_percentile": 70.0,
        "risk_level": "high",
        "prediction_reliability": "standard",
        "split_disagreement": 0.1,
        "model_release": "test_release",
    }
    risk.update(overrides)
    return risk


def _complete_metadata(**overrides):
    metadata = {
        "current_medications": [],
        "drug_allergies": [],
        "recent_antibiotics": 0,
        "recent_probiotics": 0,
        "renal_impairment": 0,
        "hepatic_impairment": 0,
        "pregnancy": 0,
    }
    metadata.update(overrides)
    return metadata


def _assessment(
    *,
    microbes=FULL_PANEL,
    features=None,
    risk=None,
    metadata=None,
    clinical=None,
):
    return build_pharmacy_assessment(
        submitted_microbes=microbes,
        clinical=clinical or {"age": 52, "family_history": 0},
        risk_result=risk or _risk(),
        model_features=features or _features(),
        metadata=metadata if metadata is not None else _complete_metadata(),
    )


def test_knowledge_base_references_existing_sources() -> None:
    knowledge = load_pharmacy_knowledge_base()
    source_ids = {source["source_id"] for source in knowledge["evidence_sources"]}

    assert knowledge["engine_version"] == "pharmacy_assistance_v3"
    assert all(
        set(rule["evidence_source_ids"]).issubset(source_ids)
        for rule in knowledge["marker_rules"]
    )
    assert len(knowledge["knowledge_sha256"]) == 64


def test_knowledge_base_validation_rejects_invalid_operator() -> None:
    knowledge = deepcopy(load_pharmacy_knowledge_base())
    knowledge["marker_rules"][0]["operator"] = "gte"

    with pytest.raises(ValueError, match="invalid operator"):
        _validate_knowledge_payload(knowledge)


def test_marker_thresholds_reproduce_from_reference_graph_table() -> None:
    knowledge = load_pharmacy_knowledge_base()

    expected, sample_count = expected_rule_thresholds(knowledge, DEFAULT_GRAPH_TABLE)

    assert verify_thresholds(knowledge, expected, sample_count) == []


def test_missing_lactobacillus_is_not_treated_as_zero() -> None:
    assessment = _assessment(
        microbes={"Fusobacterium": 0.18},
        features=_features(
            supported_microbe_input={
                "Fusobacterium": 0.50,
                "Porphyromonas": 0.0,
                "Prevotella": 0.0,
                "Streptococcus": 0.0,
                "Lactobacillus": 0.0,
            }
        ),
    )

    ids = {card["recommendation_id"] for card in assessment["recommendations"]}
    assert assessment["status"] == "limited"
    assert "Lactobacillus" in assessment["quality"]["missing_markers"]
    assert "lactobacillus_lower_quartile_review" not in ids


def test_marker_threshold_uses_panel_composition_and_preserves_submitted_value() -> None:
    assessment = _assessment()
    card = next(
        card
        for card in assessment["recommendations"]
        if card["recommendation_id"] == "fusobacterium_upper_quartile_review"
    )

    assert assessment["status"] == "standard"
    assert card["submitted_abundance"] == 0.35
    assert card["panel_composition"] == 0.35
    assert card["trigger"]["threshold_quantile"] == "q75"
    assert card["trigger"]["value_scale"] == "five_marker_panel_composition"
    assert card["allows_medication_change"] is False
    assert card["action_steps"]
    assert card["urgency_label"] == "优先处理"


def test_marker_composition_is_invariant_to_panel_total_scale() -> None:
    scaled_panel = {marker: value * 0.5 for marker, value in FULL_PANEL.items()}

    original = _assessment()
    scaled = _assessment(microbes=scaled_panel)
    original_card = next(
        card for card in original["recommendations"] if "trigger" in card
    )
    scaled_card = next(card for card in scaled["recommendations"] if "trigger" in card)

    assert scaled_card["recommendation_id"] == original_card["recommendation_id"]
    assert scaled_card["panel_composition"] == original_card["panel_composition"]


def test_out_of_training_range_withholds_marker_driven_cards() -> None:
    assessment = _assessment(
        risk=_risk(prediction_reliability="caution_out_of_training_range"),
        features=_features(
            out_of_training_range_inputs=["age"],
            out_of_training_range_details=[
                {
                    "field": "age",
                    "value": 90.0,
                    "training_minimum": 22.0,
                    "training_maximum": 84.0,
                    "affected_split_seeds": [42, 43],
                }
            ],
        ),
    )

    assert assessment["status"] == "withheld"
    assert all("trigger" not in card for card in assessment["recommendations"])
    assert assessment["recommendations"][0]["recommendation_id"] == "model_result_withheld_review"
    assert assessment["status_label"] == "暂时无法给出菌群相关建议"
    assert assessment["plain_language_summary"]["headline"] == "年龄超出当前模型研究范围，请先核对"
    reason = assessment["quality"]["status_reasons"][0]["message"]
    assert "年龄输入为 90" in reason
    assert "22 至 84" in reason
    assert "不要为了获得结果而修改数据" in reason


def test_withheld_model_keeps_independent_safety_and_guideline_context() -> None:
    assessment = _assessment(
        risk=_risk(prediction_reliability="caution_out_of_training_range"),
        features=_features(out_of_training_range_inputs=["age"]),
        metadata=_complete_metadata(
            current_medications=["ciprofloxacin 500 mg", "tizanidine 2 mg"],
            suspected_condition="irritable_bowel_syndrome",
        ),
    )

    interaction_card = next(
        card
        for card in assessment["recommendations"]
        if card["category"] == "drug_interaction_alert"
    )
    probiotic_card = next(
        card
        for card in assessment["recommendations"]
        if card["category"] == "probiotic_evidence"
    )

    assert interaction_card["urgency"] == "priority"
    assert probiotic_card["independent_of_model_result"] is True
    assert "已确认肠易激综合征" in probiotic_card["title"]
    assert "不使用本次菌群模型分数" in probiotic_card["decision_basis"]
    assert assessment["plain_language_summary"]["independent_guidance_count"] == 1


def test_unverified_backend_reliability_withholds_marker_driven_cards() -> None:
    assessment = _assessment(risk=_risk(prediction_reliability="unknown"))

    assert assessment["status"] == "withheld"
    assert assessment["summary"]["marker_trigger_count"] == 0
    assert any(
        reason["code"] == "unverified_model_reliability"
        for reason in assessment["quality"]["status_reasons"]
    )


def test_recent_antibiotics_limits_interpretation_and_adds_stewardship_card() -> None:
    assessment = _assessment(metadata=_complete_metadata(recent_antibiotics=1))
    ids = {card["recommendation_id"] for card in assessment["recommendations"]}

    assert assessment["status"] == "limited"
    assert "recent_antibiotic_exposure_review" in ids
    assert assessment["medication_context"]["recent_antibiotics"] is True


def test_missing_medication_history_requires_reconciliation() -> None:
    assessment = _assessment(metadata={})
    ids = {card["recommendation_id"] for card in assessment["recommendations"]}

    assert "medication_reconciliation_required" in ids
    assert assessment["status"] == "limited"
    assert assessment["summary"]["medication_history_complete"] is False
    assert assessment["summary"]["medication_context_complete"] is False
    assert assessment["medication_context"]["context_completeness"] == 0.0
    assert assessment["medication_context"]["interaction_screening_performed"] is False


def test_partial_medication_context_remains_limited() -> None:
    assessment = _assessment(
        metadata={"current_medications": [], "drug_allergies": []}
    )

    assert assessment["status"] == "limited"
    assert assessment["summary"]["medication_history_complete"] is True
    assert assessment["summary"]["medication_context_complete"] is False
    assert "recent_antibiotics" in assessment["medication_context"]["missing_fields"]


def test_assessment_summary_counts_only_marker_triggers() -> None:
    assessment = _assessment()

    assert assessment["summary"]["recommendation_count"] == len(
        assessment["recommendations"]
    )
    assert assessment["summary"]["marker_trigger_count"] == 1
    assert assessment["summary"]["medication_change_allowed"] is False
    assert assessment["plain_language_summary"]["urgent_count"] >= 1
    assert assessment["plain_language_summary"]["what_to_do_now"]
    assert all(card["action_steps"] for card in assessment["recommendations"])


def test_current_medication_returns_label_evidence_without_selecting_dose() -> None:
    assessment = _assessment(
        metadata=_complete_metadata(current_medications=["metformin 500 mg twice daily"])
    )

    assert assessment["summary"]["label_lookup_performed"] is True
    assert assessment["summary"]["label_record_count"] == 1
    assert assessment["summary"]["patient_specific_dose_selected"] is False
    record = assessment["drug_knowledge"]["label_lookup"]["records"][0]
    assert record["drug_id"] == "metformin"
    assert record["dose_and_course_reference"]["patient_specific_dose_selected"] is False
    assert "并未替当前患者选择剂量" in record["dose_and_course_reference"]["interpretation"]
    assert record["review_prompt"].startswith("先核对实际药名")
    assert record["source"]["dailymed_url"].startswith("https://dailymed.nlm.nih.gov/")
    assert all(
        card["recommendation_id"] != "official_label_evidence_available"
        for card in assessment["recommendations"]
    )


def test_high_priority_interaction_match_is_limited_and_non_prescriptive() -> None:
    assessment = _assessment(
        metadata=_complete_metadata(
            current_medications=["ciprofloxacin 500 mg", "tizanidine 2 mg"]
        )
    )

    assert assessment["summary"]["interaction_screening_performed"] is True
    assert assessment["summary"]["comprehensive_interaction_screening_performed"] is False
    assert assessment["summary"]["high_priority_interaction_match_count"] == 1
    card = next(
        card
        for card in assessment["recommendations"]
        if card["category"] == "drug_interaction_alert"
    )
    assert card["urgency"] == "priority"
    assert card["allows_medication_change"] is False
    assert "ciprofloxacin 500 mg" in card["title"]
    assert "tizanidine 2 mg" in card["title"]
    assert "CYP" not in card["title"]
    assert card["action_steps"][0].startswith("尽快联系开药医生或临床药师")
    assert assessment["plain_language_summary"]["headline"] == "发现需要优先核对的用药安全提示"


def test_guarded_probiotic_options_require_exact_guideline_context() -> None:
    assessment = _assessment(
        metadata=_complete_metadata(
            recent_antibiotics=1,
            suspected_condition="c_difficile_prevention_during_antibiotics",
        )
    )

    support = assessment["drug_knowledge"]["probiotic_decision_support"]
    assert support["status"] == "guideline_options_available"
    assert support["candidate_count"] == 4
    assert all(candidate["dose_selected"] is False for candidate in support["candidates"])
    assert assessment["summary"]["probiotic_candidate_count"] == 4


def test_assessment_returns_every_used_evidence_source() -> None:
    assessment = _assessment(
        metadata=_complete_metadata(recent_antibiotics=1, recent_probiotics=1),
        clinical={"age": 55, "family_history": 1},
    )
    used_source_ids = {
        source_id
        for card in assessment["recommendations"]
        for source_id in card["evidence_source_ids"]
    }
    returned_source_ids = {
        source["source_id"] for source in assessment["evidence_sources"]
    }

    assert returned_source_ids == used_source_ids
    assert all(card["allows_medication_change"] is False for card in assessment["recommendations"])


def test_special_population_flag_requires_priority_review() -> None:
    assessment = _assessment(metadata=_complete_metadata(renal_impairment=1))
    card = next(
        card
        for card in assessment["recommendations"]
        if card["recommendation_id"] == "special_population_medication_review"
    )

    assert card["urgency"] == "priority"
    assert card["allows_medication_change"] is False


def test_screening_card_is_age_context_not_a_microbiome_indication() -> None:
    assessment = _assessment(clinical={"age": 55, "family_history": 1})
    card = next(
        card
        for card in assessment["recommendations"]
        if card["recommendation_id"] == "guideline_screening_status_review"
    )

    assert card["evidence_level"] == "preventive_guideline"
    assert "菌群结果本身不能决定" in card["action_steps"][-1]


def test_clinical_report_reuses_the_unified_assessment() -> None:
    assessment = _assessment()
    model_report = {
        "top_microbes": [["Fusobacterium", 0.18]],
        "risk_result": _risk(),
        "recommendations": assessment["recommendations"],
        "pharmacy_assessment": assessment,
    }
    standardized = {"metadata": {"sample_id": "TEST-001"}}

    advice = build_pharmacy_assistance(model_report, standardized["metadata"])
    report = build_clinical_report(standardized, model_report, advice)

    assert report["pharmacy_assessment"] == assessment
    assert report["pharmacological_assistance"] == assessment["recommendations"]
    assert report["microbiome_findings"]["rule_trigger_count"] == 1
