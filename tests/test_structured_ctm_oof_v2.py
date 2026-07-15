from __future__ import annotations

import numpy as np
import pytest
import torch

from research.structured_ctm_oof_v2 import (
    CandidateConfig,
    CandidateResult,
    StructuredCTMResidualAdapter,
    StructuredSplit,
    _fit_group_scaler,
    _groups_to_tensors,
    _select_candidate_with_alpha,
    load_structured_splits,
)


def test_load_structured_splits_reads_expected_groups(tmp_path) -> None:
    path = tmp_path / "features.npz"
    payload = {}
    for split, n in [("train", 4), ("val", 3), ("test", 2)]:
        payload[f"{split}_sample_ids"] = np.asarray([f"{split}_{i}" for i in range(n)])
        payload[f"{split}_time"] = np.arange(1, n + 1, dtype=float)
        payload[f"{split}_event"] = np.asarray([1, 0] * ((n + 1) // 2), dtype=float)[:n]
        payload[f"{split}_standardized_topk_risk"] = np.linspace(0.0, 1.0, n)
        payload[f"{split}_standardized_risk_matrix"] = np.ones((2, n))
        payload[f"{split}_risk_disagreement"] = np.ones((n, 6))
        payload[f"{split}_graph_embedding_mean"] = np.ones((n, 3))
        payload[f"{split}_graph_embedding_std"] = np.ones((n, 3))
        payload[f"{split}_latent_mean"] = np.ones((n, 2))
        payload[f"{split}_latent_std"] = np.ones((n, 2))
        payload[f"{split}_graph_target_mean"] = np.ones((n, 1))
        payload[f"{split}_graph_cluster_target_mean"] = np.ones((n, 1))
    np.savez(path, **payload)

    splits = load_structured_splits(path)

    assert splits["train"].groups["risk_members"].shape == (4, 2)
    assert splits["val"].groups["topology_targets"].shape == (3, 2)
    assert splits["test"].sample_ids == ["test_0", "test_1"]


def test_group_scaler_uses_fit_indices_only() -> None:
    split = _toy_split()
    scaler = _fit_group_scaler(split.groups, np.asarray([0, 1]))
    tensors = _groups_to_tensors(split, np.asarray([2]), scaler, torch.device("cpu"))

    np.testing.assert_allclose(scaler["risk_members"][0], [[0.5, 1.5]])
    np.testing.assert_allclose(tensors["risk_members"].numpy(), [[3.0, 3.0]])


def test_structured_ctm_residual_adapter_bounds_delta() -> None:
    group_dims = {
        "risk_members": 2,
        "risk_disagreement": 6,
        "graph_embedding_mean": 3,
        "graph_embedding_std": 3,
        "latent_mean": 2,
        "latent_std": 2,
        "topology_targets": 2,
    }
    model = StructuredCTMResidualAdapter(
        group_dims,
        d_input=8,
        d_model=8,
        n_heads=2,
        n_synch_action=4,
        n_synch_out=4,
        n_self_pairs=1,
        max_delta=0.2,
    )
    groups = {name: torch.randn(5, dim) for name, dim in group_dims.items()}
    baseline = torch.zeros(5)

    output = model(groups, baseline)

    assert output["risk"].shape == baseline.shape
    assert float(output["delta"].detach().abs().max()) <= 0.2
    assert float(output["gate"].detach().min()) >= 0.0
    assert float(output["gate"].detach().max()) <= 1.0


def test_select_candidate_with_alpha_uses_oof_and_high_disagreement_gate() -> None:
    train = _toy_split()
    val = _toy_split()
    test = _toy_split()
    config = CandidateConfig(7, 0.2, 0.1, 0.08, 0.25)
    good = CandidateResult(
        name="good",
        config=config,
        oof_risk=np.asarray([3.0, 2.0, 1.0, 0.0]),
        val_risk=np.asarray([3.0, 2.0, 1.0, 0.0]),
        test_risk=np.asarray([3.0, 2.0, 1.0, 0.0]),
        oof_c_index=1.0,
        val_c_index=1.0,
        test_c_index=1.0,
        best_epochs=[1],
        final_best_epoch=1,
    )

    selected = _select_candidate_with_alpha(
        train=train,
        val=val,
        test=test,
        candidates=[good],
        alpha_grid=[0.0, 1.0],
        min_oof_delta=0.0,
        min_val_delta=0.0,
        min_high_disagreement_val_delta=0.0,
    )

    assert selected["candidate_name"] == "good"
    assert selected["alpha"] == pytest.approx(1.0)
    assert selected["test_c_index"] == pytest.approx(1.0)


def _toy_split() -> StructuredSplit:
    n = 4
    groups = {
        "risk_members": np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        "risk_disagreement": np.asarray(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "graph_embedding_mean": np.ones((n, 3)),
        "graph_embedding_std": np.ones((n, 3)),
        "latent_mean": np.ones((n, 2)),
        "latent_std": np.ones((n, 2)),
        "topology_targets": np.ones((n, 2)),
    }
    return StructuredSplit(
        sample_ids=[str(index) for index in range(n)],
        time=np.asarray([1.0, 2.0, 3.0, 4.0]),
        event=np.asarray([1.0, 1.0, 0.0, 0.0]),
        baseline_risk=np.asarray([0.0, 1.0, 2.0, 3.0]),
        groups=groups,
    )
