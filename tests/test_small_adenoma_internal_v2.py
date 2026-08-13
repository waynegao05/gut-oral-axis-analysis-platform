from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.small_adenoma_internal_v2.benchmark import (
    CandidateConfig,
    FoldPreprocessor,
    augment_minority,
    candidate_configs,
    consensus_predictions,
    load_inputs,
    threshold_for_sensitivity,
)


def test_internal_v2_real_cohort_and_feature_counts() -> None:
    formal, transfer, feature_map = load_inputs()
    assert len(formal) == 88
    assert formal["disease_group"].value_counts().to_dict() == {
        "healthy": 61,
        "small_adenoma": 27,
    }
    assert len(transfer) == 68
    assert int((feature_map["rank"] == "genus").sum()) == 204
    assert int((feature_map["rank"] == "species").sum()) == 661
    assert formal["sample_id"].nunique() == 88
    assert formal["subject_id"].nunique() == 88


def test_locked_candidate_space_is_deterministic() -> None:
    first = candidate_configs()
    second = candidate_configs()
    assert len(first) == 112
    assert [item.config_id for item in first] == [item.config_id for item in second]
    assert len({item.config_id for item in first}) == len(first)


def test_fold_preprocessor_fits_and_transforms_without_invalid_values() -> None:
    formal, _, feature_map = load_inputs()
    config = CandidateConfig(
        feature_set="genus",
        top_k=15,
        model_name="log_l2_c01",
        include_clinical=True,
        augmentation="none",
    )
    y = formal["small_adenoma_label"].to_numpy(dtype=int)
    preprocessor = FoldPreprocessor.fit(formal.iloc[:70], y[:70], feature_map, config)
    transformed = preprocessor.transform(formal.iloc[70:])
    assert transformed.shape == (18, 18)
    assert np.isfinite(transformed).all()
    assert len(preprocessor.selected_feature_ids) == 15


def test_minority_mixup_is_training_only_and_balances_classes() -> None:
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    augmented_x, augmented_y, generated = augment_minority(
        x,
        y,
        "minority_mixup",
        seed=42,
    )
    assert generated == 4
    assert augmented_x.shape == (14, 3)
    assert int(np.sum(augmented_y == 0)) == int(np.sum(augmented_y == 1)) == 7
    np.testing.assert_array_equal(augmented_x[:10], x)
    np.testing.assert_array_equal(augmented_y[:10], y)


def test_threshold_prioritizes_sensitivity_then_specificity() -> None:
    y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probability = np.asarray([0.05, 0.10, 0.20, 0.30, 0.70, 0.25, 0.40, 0.50, 0.80, 0.90])
    result = threshold_for_sensitivity(y, probability, target_sensitivity=0.80)
    predicted = probability >= result["threshold"]
    assert np.mean(predicted[y == 1]) >= 0.80
    assert np.mean(~predicted[y == 0]) == result["inner_oof_specificity"]


def test_consensus_requires_three_of_five_cross_fitted_votes() -> None:
    rows = []
    for seed, prediction in zip((7, 21, 42, 123, 2026), (1, 1, 1, 0, 0), strict=True):
        rows.append(
            {
                "sample_id": "sample_1",
                "subject_id": "subject_1",
                "disease_group": "small_adenoma",
                "small_adenoma_label": 1,
                "seed": seed,
                "fold": 1,
                "probability": 0.7 if prediction else 0.3,
                "threshold": 0.5,
                "decision_margin": 0.2 if prediction else -0.2,
                "prediction": prediction,
            }
        )
    consensus = consensus_predictions(pd.DataFrame(rows))
    assert len(consensus) == 1
    assert int(consensus.loc[0, "positive_votes"]) == 3
    assert int(consensus.loc[0, "prediction"]) == 1
