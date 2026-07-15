from __future__ import annotations

import numpy as np
import pytest
import torch

from research.main_model_hardpair_residual_v3 import (
    _build_feature_matrices,
    _hard_pair_mask,
    _parse_floats,
    _parse_ints,
)
from research.structured_ctm_oof_v2 import StructuredSplit


def _split(offset: float = 0.0) -> StructuredSplit:
    n = 4
    return StructuredSplit(
        sample_ids=[f"S{i}" for i in range(n)],
        time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        event=np.asarray([1.0, 1.0, 0.0, 1.0]),
        baseline_risk=np.asarray([0.4, 0.2, -0.1, -0.3]) + offset,
        groups={
            "risk_members": np.asarray(
                [
                    [0.4, 0.3],
                    [0.2, 0.1],
                    [-0.1, 0.0],
                    [-0.3, -0.2],
                ],
                dtype=float,
            )
            + offset,
            "risk_disagreement": np.asarray(
                [
                    [0.0, 0.1],
                    [0.2, 0.0],
                    [0.4, 0.2],
                    [0.6, 0.3],
                ],
                dtype=float,
            ),
            "graph_embedding_mean": np.arange(n * 3, dtype=float).reshape(n, 3) + offset,
            "graph_embedding_std": np.ones((n, 2), dtype=float) * (1.0 + offset),
            "latent_mean": np.arange(n * 2, dtype=float).reshape(n, 2) * 0.1 + offset,
        },
    )


def test_build_feature_matrices_standardizes_from_train_split() -> None:
    train_x, val_x, test_x, train_gate, val_gate, test_gate, scaler = _build_feature_matrices(
        _split(0.0),
        _split(1.0),
        _split(2.0),
    )

    assert train_x.dtype == np.float32
    assert train_gate.dtype == np.float32
    assert train_x.shape[0] == val_x.shape[0] == test_x.shape[0] == 4
    assert train_gate.shape[1] == 3
    np.testing.assert_allclose(train_x.mean(axis=0), np.zeros(train_x.shape[1]), atol=1e-6)
    assert scaler["feature_groups"] == [
        "risk_members",
        "risk_disagreement",
        "graph_embedding_mean",
        "graph_embedding_std",
        "latent_mean",
    ]
    assert val_gate.shape == test_gate.shape == train_gate.shape


def test_hard_pair_mask_selects_baseline_misordered_pairs() -> None:
    mask = _hard_pair_mask(
        time=np.asarray([1.0, 2.0, 3.0]),
        event=np.asarray([1.0, 1.0, 0.0]),
        baseline_risk=np.asarray([0.0, 1.0, 2.0]),
        hard_margin=0.0,
        device=torch.device("cpu"),
    )

    assert mask.dtype == torch.bool
    assert int(mask.sum().item()) == 3
    assert mask[0, 1].item() is True
    assert mask[0, 2].item() is True
    assert mask[1, 2].item() is True


def test_parse_helpers() -> None:
    assert _parse_ints("7, 21,,42") == [7, 21, 42]
    assert _parse_floats("0.05,0.1,,0.2") == pytest.approx([0.05, 0.1, 0.2])
