from __future__ import annotations

import pytest

from src.ac_icam_v8_bridge import (
    ACICAMV8ModelBridge,
    get_ac_icam_v8_model_bridge,
)


VALID_CLINICAL = {
    "age": 62,
    "sex": "Female",
    "stage": 3,
    "path_t": 3,
    "path_n": 1,
    "path_m": 0,
    "tumor_location": "Colon Sigmoideum",
    "tumor_morphology": "Adenocarcinoma",
}
VALID_MICROBES = {
    "Fusobacterium": 0.18,
    "Porphyromonas": 0.15,
    "Prevotella": 0.10,
    "Streptococcus": 0.09,
    "Lactobacillus": 0.02,
}


def test_v8_core_bridge_returns_pfs_prediction() -> None:
    prediction = get_ac_icam_v8_model_bridge().score(
        {},
        VALID_CLINICAL,
        {},
    )

    risk = prediction.risk_result
    features = prediction.model_features
    assert risk["backend"] == "ac_icam_real_outcome_clinical_pfs"
    assert risk["model_release"] == "ac_icam_real_outcome_pfs_v8"
    assert risk["model_variant"] == "clinical_core"
    assert risk["prediction_reliability"] == "standard"
    assert risk["prediction_available"] is True
    assert 0.0 <= risk["risk_percentile"] <= 100.0
    assert 0.0 < risk["pfs_probability"]["60"] <= risk["pfs_probability"]["36"] < 1.0
    assert features["microbiome_used_for_risk"] is False
    assert features["treatment_used_for_risk"] is False
    assert prediction.general_risk_result is None
    assert features["formal_metrics"]["ensemble_oof_c_index"] == pytest.approx(
        0.7756446991404011
    )


def test_v8_measured_icr_activates_expanded_variant() -> None:
    clinical = {**VALID_CLINICAL, "icr_score": 7.1}

    prediction = get_ac_icam_v8_model_bridge().score({}, clinical, {})

    assert prediction.risk_result["model_variant"] == "clinical_icr"
    assert prediction.model_features["icr_used_for_risk"] is True
    assert prediction.model_features["formal_metrics"]["ensemble_oof_c_index"] == pytest.approx(
        0.7845272206303725
    )


def test_v8_bridge_withholds_pfs_when_oncology_fields_are_missing() -> None:
    clinical = dict(VALID_CLINICAL)
    clinical.pop("stage")

    prediction = get_ac_icam_v8_model_bridge().score(
        VALID_MICROBES,
        clinical,
        {},
    )

    assert prediction.risk_result["prediction_available"] is False
    assert prediction.risk_result["not_available_reason"] == (
        "missing_oncology_fields"
    )
    assert prediction.risk_result["missing_oncology_fields"] == [
        "clinical.stage"
    ]
    assert prediction.risk_result["pfs_probability"] == {
        "36": None,
        "60": None,
    }
    assert prediction.model_features["pfs_model_eligible"] is False
    assert prediction.model_features["defaulted_inputs"] == []
    assert prediction.general_risk_result["prediction_available"] is True
    assert prediction.general_risk_result["endpoint"] == "research_risk_index"
    assert (
        prediction.general_risk_result["absolute_cancer_probability"]
        is False
    )
    assert (
        0.0
        <= prediction.general_risk_result["risk_percentile"]
        <= 100.0
    )
    assert prediction.general_risk_features["sex_used_for_risk"] is False


def test_v8_bridge_does_not_impute_missing_pathological_m() -> None:
    clinical = dict(VALID_CLINICAL)
    clinical.pop("path_m")

    prediction = get_ac_icam_v8_model_bridge().score({}, clinical, {})

    assert prediction.risk_result["prediction_available"] is False
    assert prediction.risk_result["missing_oncology_fields"] == [
        "clinical.path_m"
    ]
    assert prediction.model_features["defaulted_inputs"] == []
    assert (
        prediction.general_risk_result["not_available_reason"]
        == "incomplete_microbiome_panel"
    )


def test_v8_general_risk_requires_complete_five_microbe_panel() -> None:
    prediction = get_ac_icam_v8_model_bridge().score(
        {"Fusobacterium": 0.2},
        {"age": 23, "sex": "Male"},
        {},
    )

    general = prediction.general_risk_result
    assert general["prediction_available"] is False
    assert general["not_available_reason"] == "incomplete_microbiome_panel"
    assert general["missing_microbe_fields"] == [
        "microbes.Porphyromonas",
        "microbes.Prevotella",
        "microbes.Streptococcus",
        "microbes.Lactobacillus",
    ]
    assert general["risk_score"] is None


def test_v8_general_risk_withholds_out_of_reference_age() -> None:
    prediction = get_ac_icam_v8_model_bridge().score(
        VALID_MICROBES,
        {"age": 18, "sex": "Female"},
        {},
    )

    general = prediction.general_risk_result
    assert general["prediction_available"] is False
    assert general["not_available_reason"] == "out_of_training_range"
    assert general["risk_score"] is None
    assert "age" in prediction.general_risk_features[
        "out_of_training_range_inputs"
    ]


@pytest.mark.parametrize("missing_field", ["age", "sex"])
def test_v8_bridge_still_requires_age_and_sex(
    missing_field: str,
) -> None:
    clinical = dict(VALID_CLINICAL)
    clinical.pop(missing_field)

    with pytest.raises(ValueError, match=rf"clinical\.{missing_field}"):
        get_ac_icam_v8_model_bridge().score({}, clinical, {})


def test_v8_bridge_marks_training_range_violation_unavailable() -> None:
    clinical = {**VALID_CLINICAL, "age": 18}

    prediction = get_ac_icam_v8_model_bridge().score({}, clinical, {})

    assert prediction.risk_result["prediction_reliability"] == (
        "caution_out_of_training_range"
    )
    assert prediction.risk_result["prediction_available"] is False
    assert prediction.model_features["out_of_training_range_details"] == [
        {
            "field": "clinical.age",
            "value": 18.0,
            "training_minimum": 25.0,
            "training_maximum": 88.0,
        }
    ]


def test_v8_bridge_accepts_chinese_category_aliases() -> None:
    clinical = {
        **VALID_CLINICAL,
        "sex": "女",
        "tumor_location": "乙状结肠",
        "tumor_morphology": "腺癌",
    }

    prediction = ACICAMV8ModelBridge().score({}, clinical, {})

    assert prediction.model_features["used_clinical_inputs"]["sex"] == "Female"
    assert (
        prediction.model_features["used_clinical_inputs"]["tumor_location"]
        == "Colon Sigmoideum"
    )


def test_v8_pfs_score_is_independent_of_submitted_microbes() -> None:
    bridge = get_ac_icam_v8_model_bridge()

    without_microbes = bridge.score({}, VALID_CLINICAL, {})
    with_microbes = bridge.score(
        {"Fusobacterium": 0.9, "Lactobacillus": 0.1},
        VALID_CLINICAL,
        {},
    )

    assert without_microbes.risk_result == with_microbes.risk_result
    assert with_microbes.model_features["submitted_microbe_count"] == 2
    assert with_microbes.model_features["artifact_source"] == (
        "config/releases/ac_icam_real_outcome_pfs_v8.json"
    )
    assert with_microbes.model_features["artifact_sha256"] == (
        "feb8036a52d1c14327b93b6b324ff2c09bcee5b7f711ea0f16c84f44e006ab95"
    )
