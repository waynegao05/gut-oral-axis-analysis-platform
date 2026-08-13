from __future__ import annotations

import numpy as np

from experiments.adenoma_screening_v1.benchmark import (
    EXPECTED_COUNTS,
    binary_rates,
    consensus_predictions,
    load_screening_data,
    select_threshold,
    transform_abundance,
    wilson_interval,
)


def test_locked_zeller_dataset_counts_and_labels() -> None:
    frame = load_screening_data()
    assert len(frame) == 156
    assert frame["disease_group"].value_counts().to_dict() == EXPECTED_COUNTS
    assert int(frame["screening_label"].sum()) == 95


def test_abundance_transform_is_finite_with_zeros() -> None:
    transformed = transform_abundance(np.asarray([[0.0, 0.1, 100.0]]))
    assert transformed.shape == (1, 3)
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] < transformed[0, 1] < transformed[0, 2]


def test_threshold_rule_meets_locked_specificity_target() -> None:
    y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.asarray([0.05, 0.10, 0.20, 0.80, 0.30, 0.40, 0.70, 0.90])
    selected = select_threshold(y_true, probabilities, target_specificity=0.75)
    predictions = (probabilities >= selected["threshold"]).astype(int)
    specificity, sensitivity = binary_rates(y_true, predictions)
    assert specificity >= 0.75
    assert sensitivity == 1.0


def test_consensus_uses_five_seed_majority_vote() -> None:
    import pandas as pd

    rows = []
    for seed, prediction in zip((7, 21, 42, 123, 2026), (1, 1, 1, 0, 0), strict=True):
        rows.append(
            {
                "sample_id": "sample_1",
                "subject_id": "subject_1",
                "disease_group": "small_adenoma",
                "screening_label": 1,
                "seed": seed,
                "fold": 1,
                "probability": 0.6 if prediction else 0.4,
                "specificity_90_prediction": prediction,
                "specificity_95_prediction": prediction,
            }
        )
    consensus = consensus_predictions(pd.DataFrame(rows))
    assert len(consensus) == 1
    assert int(consensus.loc[0, "specificity_90_positive_votes"]) == 3
    assert int(consensus.loc[0, "specificity_90_prediction"]) == 1


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(8, 10)
    assert lower < 0.8 < upper
