from __future__ import annotations

import numpy as np
import pandas as pd

from ctm_fusion_experiment.utils.data_loader import (
    FoldSplit,
    FrozenFeatureScaler,
    make_cv_splits,
    prepare_fusion_arrays,
)


def _sample_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"S{idx}" for idx in range(12)],
            "time": np.arange(1, 13, dtype=float),
            "event": [0, 1] * 6,
        }
    )


def test_make_cv_splits_is_deterministic_and_disjoint() -> None:
    first = make_cv_splits(_sample_table(), folds=3, seed=42, val_ratio=0.25)
    second = make_cv_splits(_sample_table(), folds=3, seed=42, val_ratio=0.25)

    assert first == second
    assert len(first) == 3
    for fold in first:
        train = set(fold.train_ids)
        val = set(fold.val_ids)
        test = set(fold.test_ids)
        assert not train & val
        assert not train & test
        assert not val & test
        assert train | val | test == set(_sample_table()["sample_id"])


def test_frozen_feature_scaler_uses_train_statistics_only() -> None:
    scaler = FrozenFeatureScaler()
    train = np.array([[0.0, 10.0], [2.0, 14.0]])
    test = np.array([[4.0, 18.0]])

    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)

    assert np.allclose(scaled_train.mean(axis=0), [0.0, 0.0])
    assert np.allclose(scaled_test, [[3.0, 3.0]])


def test_prepare_fusion_arrays_aligns_ids_and_scales_from_train_only() -> None:
    sample_table = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "clinical": [0.0, 2.0, 4.0, 6.0],
            "metabolite": [10.0, 14.0, 18.0, 22.0],
            "time": [4.0, 3.0, 2.0, 1.0],
            "event": [0, 1, 0, 1],
        }
    )
    split = FoldSplit(fold=1, train_ids=("S1", "S2"), val_ids=("S3",), test_ids=("S4",))
    embeddings = {
        "S1": np.array([0.0, 10.0]),
        "S2": np.array([2.0, 14.0]),
        "S3": np.array([4.0, 18.0]),
        "S4": np.array([6.0, 22.0]),
    }

    arrays = prepare_fusion_arrays(
        sample_table=sample_table,
        split=split,
        graph_embeddings=embeddings,
        clinical_columns=["clinical"],
        metabolite_columns=["metabolite"],
    )

    assert arrays.train.sample_ids == ("S1", "S2")
    assert np.allclose(arrays.train.graph.mean(axis=0), [0.0, 0.0])
    assert np.allclose(arrays.test.graph, [[5.0, 5.0]])
    assert np.allclose(arrays.test.clinical, [[5.0]])
    assert np.allclose(arrays.test.metabolite, [[5.0]])
