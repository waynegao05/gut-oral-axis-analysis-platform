from __future__ import annotations

from copy import deepcopy

import pytest

from src.drug_knowledge import (
    _validate_label_database,
    build_drug_knowledge_review,
    build_probiotic_decision_support,
    load_label_database,
    normalize_medication_inputs,
    screen_high_priority_interactions,
)


def test_generated_label_database_is_complete_and_hash_verified() -> None:
    database = load_label_database()

    assert database["dataset_id"] == "goa_openfda_label_evidence_v1"
    assert database["coverage"]["requested_medication_count"] == 46
    assert database["coverage"]["label_record_count"] == 46
    assert database["coverage"]["failure_count"] == 0
    assert database["coverage"]["complete"] is True
    assert len(database["records_sha256"]) == 64
    assert len(database["file_sha256"]) == 64


def test_label_database_hash_check_detects_modified_record() -> None:
    database = deepcopy(load_label_database())
    database["records"][0]["display_name"] = "modified"

    with pytest.raises(ValueError, match="records_sha256"):
        _validate_label_database(database)


def test_medication_normalization_handles_dose_brand_and_chinese_name() -> None:
    results = normalize_medication_inputs(
        ["metformin 500 mg twice daily", "Eliquis 5 mg BID", "二甲双胍"]
    )

    assert [item["status"] for item in results] == ["matched", "matched", "matched"]
    assert [item["drug_id"] for item in results] == ["metformin", "apixaban", "metformin"]
    assert results[0]["rxcui"] == "6809"


def test_high_priority_screen_can_match_generic_name_outside_label_seed() -> None:
    normalized = normalize_medication_inputs(["sertraline 50 mg", "phenelzine 15 mg"])
    screening = screen_high_priority_interactions(normalized)

    assert normalized[1]["status"] == "unmatched"
    assert screening["interaction_screening_performed"] is True
    assert screening["comprehensive_interaction_screening_performed"] is False
    assert screening["match_count"] == 1
    assert screening["matches"][0]["rule_id"] == "onc_ddi_08_antidepressant_maoi"
    assert screening["negative_result_excludes_other_interactions"] is False
    assert screening["note"].startswith("这里只核对了")


def test_unresolved_medication_prevents_completed_subset_screening() -> None:
    normalized = normalize_medication_inputs(["unknown brand alpha", "unknown brand beta"])
    screening = screen_high_priority_interactions(normalized)

    assert screening["interaction_screening_performed"] is False
    assert screening["screening_status"] == "incomplete_unresolved_medication_names"
    assert screening["resolved_for_subset_count"] == 0
    assert screening["comprehensive_interaction_screening_performed"] is False


def test_label_and_allergy_review_never_selects_patient_dose_or_change() -> None:
    review = build_drug_knowledge_review(
        current_medications=["metformin 500 mg twice daily"],
        drug_allergies=["metformin: rash"],
        metadata={"renal_impairment": 1},
    )

    assert review["label_lookup"]["record_count"] == 1
    label = review["label_lookup"]["records"][0]
    assert label["dose_and_course_reference"]["available"] is True
    assert label["dose_and_course_reference"]["patient_specific_dose_selected"] is False
    assert label["review_prompt"].startswith("先核对实际药名")
    assert any(
        "不可直接照此改剂量" in section["label_zh"]
        for section in label["sections"].values()
    )
    assert label["allows_medication_change"] is False
    assert review["allergy_screening"]["match_count"] == 1
    assert review["allergy_screening"]["class_cross_reactivity_screening_performed"] is False
    assert review["candidate_therapy_support"]["medication_candidates_generated"] is False
    assert review["safety_boundary"]["not_a_prescription"] is True


def test_probiotic_candidates_are_condition_specific_and_non_prescriptive() -> None:
    default = build_probiotic_decision_support(
        {"suspected_condition": "gut_risk_screening", "recent_antibiotics": 1}
    )
    eligible = build_probiotic_decision_support(
        {
            "suspected_condition": "c_difficile_prevention_during_antibiotics",
            "recent_antibiotics": 1,
        }
    )
    trial_only = build_probiotic_decision_support(
        {"suspected_condition": "irritable_bowel_syndrome"}
    )

    assert default["candidate_count"] == 0
    assert default["status"] == "not_generated_no_supported_indication"
    assert eligible["candidate_count"] == 4
    assert eligible["required_safety_review"][0].startswith("先独立确认")
    assert all(candidate["dose_selected"] is False for candidate in eligible["candidates"])
    assert all(
        candidate["allows_automatic_start_or_stop"] is False
        for candidate in eligible["candidates"]
    )
    assert trial_only["status"] == "no_routine_candidate"
    assert trial_only["stance"] == "clinical_trial_only"
