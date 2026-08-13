from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from experiments.oral_adenoma_internal_v3.benchmark import candidate_configs, load_inputs, threshold_at_fpr
from experiments.oral_adenoma_internal_v3.benchmark import ROOT
from experiments.oral_adenoma_internal_v3.modeling import CLRTransformer
from experiments.oral_adenoma_internal_v3.predict import predict_frame
from experiments.oral_adenoma_internal_v3.prepare_data import assert_oral_only


def test_oral_only_real_cohort_and_features() -> None:
    frame, feature_map = load_inputs()
    assert len(frame) == 92
    assert frame["disease_group"].value_counts().to_dict() == {
        "healthy": 58,
        "adenoma": 34,
    }
    assert set(frame["sample_type"]) == {"oral_swab"}
    assert frame["sample_id"].nunique() == 92
    assert len(feature_map) == 381
    assert set(feature_map["rank"]) == {"genus"}


def test_forbidden_non_oral_sources_fail_closed() -> None:
    for source in ("stool", "fecal", "faecal", "gut", "blood", "serum", "plasma"):
        with pytest.raises(ValueError, match="Forbidden non-oral"):
            assert_oral_only([source])
    assert_oral_only(["oral_swab", "saliva", "buccal_swab"])


def test_candidate_space_is_locked_and_deterministic() -> None:
    first = candidate_configs()
    second = candidate_configs()
    assert len(first) == 16
    assert [config.config_id for config in first] == [config.config_id for config in second]
    assert len({config.config_id for config in first}) == 16


def test_clr_transform_is_finite_and_centered() -> None:
    values = np.asarray([[0.0, 1.0, 99.0], [20.0, 30.0, 50.0]])
    transformed = CLRTransformer().fit_transform(values)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        CLRTransformer().fit_transform(np.asarray([[1.0, -1.0]]))


def test_training_threshold_respects_false_positive_budget() -> None:
    y = np.asarray([0] * 58 + [1] * 34)
    probability = np.linspace(0.0, 1.0, len(y))
    result = threshold_at_fpr(y, probability, target_fpr=0.055)
    predicted = probability >= float(result["threshold"])
    assert int(np.sum(predicted[y == 0])) <= 3
    assert int(result["allowed_false_positives"]) == 3


def test_saved_bundle_predicts_oral_and_rejects_stool() -> None:
    model_path = (
        ROOT
        / "outputs"
        / "oral_adenoma_internal_v3"
        / "oral_adenoma_internal_model.joblib"
    )
    if not model_path.exists():
        pytest.skip("Formal oral model has not been trained yet.")
    bundle = joblib.load(model_path)
    frame, _ = load_inputs()
    oral = predict_frame(bundle, frame.iloc[:2].copy())
    assert len(oral) == 2
    assert oral["adenoma_probability"].between(0.0, 1.0).all()
    stool = frame.iloc[:1].copy()
    stool["sample_type"] = "stool"
    with pytest.raises(ValueError, match="Forbidden non-oral"):
        predict_frame(bundle, stool)
