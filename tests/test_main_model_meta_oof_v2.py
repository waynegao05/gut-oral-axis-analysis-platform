from __future__ import annotations

import numpy as np
import pytest

from research.main_model_meta_oof_v2 import (
    MetaCandidate,
    MetaConfig,
    _build_meta_features,
    _parse_floats,
    _parse_ints,
    _parse_strings,
    _select_candidate,
)
from research.structured_ctm_oof_v2 import StructuredSplit


def _split(offset: float = 0.0) -> StructuredSplit:
    n = 4
    return StructuredSplit(
        sample_ids=[f"S{i}" for i in range(n)],
        time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        event=np.asarray([1.0, 1.0, 1.0, 0.0]),
        baseline_risk=np.asarray([0.0, 0.2, 0.4, 0.6]) + offset,
        groups={
            "risk_members": np.asarray(
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                    [0.6, 0.7],
                ],
                dtype=float,
            )
            + offset,
            "risk_disagreement": np.asarray(
                [
                    [0.1, 0.0],
                    [0.2, 0.1],
                    [0.3, 0.2],
                    [0.9, 0.3],
                ],
                dtype=float,
            ),
            "graph_embedding_mean": np.arange(n * 3, dtype=float).reshape(n, 3) + offset,
            "graph_embedding_std": np.ones((n, 2), dtype=float),
            "latent_mean": np.arange(n * 2, dtype=float).reshape(n, 2) * 0.1 + offset,
        },
    )


def test_build_meta_features_standardizes_with_train_statistics() -> None:
    train_x, val_x, test_x, scaler = _build_meta_features(_split(0.0), _split(1.0), _split(2.0))

    assert train_x.dtype == np.float32
    assert train_x.shape == val_x.shape == test_x.shape
    np.testing.assert_allclose(train_x.mean(axis=0), np.zeros(train_x.shape[1]), atol=1e-6)
    assert scaler["groups"] == [
        "risk_members",
        "risk_disagreement",
        "graph_embedding_mean",
        "graph_embedding_std",
        "latent_mean",
    ]


def test_select_candidate_uses_validation_first_policy() -> None:
    train = _split(0.0)
    val = _split(0.0)
    test = _split(0.0)
    config = MetaConfig(
        name="candidate",
        seed=7,
        model_type="linear",
        max_delta=0.05,
        distillation_weight=0.1,
        delta_l2_weight=0.08,
        dropout=0.2,
    )
    candidate = MetaCandidate(
        config=config,
        oof_risk=np.asarray([0.6, 0.4, 0.2, 0.0]),
        val_risk=np.asarray([0.6, 0.4, 0.2, 0.0]),
        test_risk=np.asarray([0.6, 0.4, 0.2, 0.0]),
        oof_c_index=1.0,
        val_c_index=1.0,
        test_c_index=1.0,
        best_epochs=[1, 1],
        final_best_epoch=1,
    )

    selected = _select_candidate(
        train=train,
        val=val,
        test=test,
        candidates=[candidate],
        alpha_grid=[0.0, 1.0],
        min_oof_delta=0.0,
        min_val_delta=0.0,
        min_high_disagreement_val_delta=0.0,
        selection_policy="validation_then_oof",
    )

    assert selected["candidate_name"] == "candidate"
    assert selected["alpha"] == 1.0
    assert selected["val_delta"] > 0.0


def test_parse_helpers() -> None:
    assert _parse_ints("7, 21,,42") == [7, 21, 42]
    assert _parse_floats("0.03,0.05,,0.1") == pytest.approx([0.03, 0.05, 0.1])
    assert _parse_strings("linear, mlp,,") == ["linear", "mlp"]
