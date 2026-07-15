from __future__ import annotations

import numpy as np
import pytest
import torch

from research.risk_adapter_v2 import (
    BoundedMlpResidualAdapter,
    DisagreementGatedResidualAdapter,
    _build_baseline_specs,
    _build_standardized_disagreement_splits,
    _disagreement_features,
    _parse_float_list,
    _parse_int_list,
    _parse_str_list,
    _standardize_gnn_members_by_validation,
    _validation_softmax_weights,
)


def test_disagreement_features_match_expected_columns() -> None:
    risk_matrix = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.5, 4.0],
            [4.0, 3.0, 5.0],
        ]
    )
    raw_mean = risk_matrix.mean(axis=0)

    features = _disagreement_features(risk_matrix, raw_mean, top_indices=[1, 2])

    assert features.shape == (3, 6)
    assert features[:, 0] == pytest.approx(risk_matrix.std(axis=0))
    assert features[:, 1] == pytest.approx(risk_matrix.max(axis=0) - risk_matrix.min(axis=0))
    assert features[:, 2] == pytest.approx(risk_matrix[[1, 2]].std(axis=0))
    assert features[:, 3] == pytest.approx(risk_matrix[[1, 2]].max(axis=0) - risk_matrix[[1, 2]].min(axis=0))
    assert features[:, 4] == pytest.approx(np.abs(risk_matrix[[1, 2]].mean(axis=0) - raw_mean))
    assert features[:, 5] == pytest.approx(np.max(np.abs(risk_matrix - raw_mean[None, :]), axis=0))


def test_standardized_disagreement_splits_use_train_statistics() -> None:
    class Prediction:
        def __init__(self, risk_matrix: np.ndarray) -> None:
            self.risk_matrix = risk_matrix

    train_matrix = np.asarray([[1.0, 2.0, 3.0], [3.0, 3.0, 5.0], [5.0, 4.0, 7.0]])
    val_matrix = train_matrix + 1.0
    test_matrix = train_matrix + 2.0

    names, train, val, test, scaler = _build_standardized_disagreement_splits(
        gnn_train=Prediction(train_matrix),
        gnn_val=Prediction(val_matrix),
        gnn_test=Prediction(test_matrix),
        raw_train_risk=train_matrix.mean(axis=0),
        raw_val_risk=val_matrix.mean(axis=0),
        raw_test_risk=test_matrix.mean(axis=0),
        top_indices=[0, 1],
    )

    assert names == [
        "risk_std_all",
        "risk_range_all",
        "risk_std_top3",
        "risk_range_top3",
        "abs_top3_minus_raw_mean",
        "max_abs_member_minus_raw_mean",
    ]
    assert train.mean(axis=0) == pytest.approx(np.zeros(6))
    assert scaler["feature_means"] == pytest.approx(_disagreement_features(train_matrix, train_matrix.mean(axis=0), [0, 1]).mean(axis=0))
    assert val.shape == train.shape == test.shape


def test_standardize_gnn_members_by_validation_reuses_validation_scaler_for_train_and_test() -> None:
    train = np.asarray([[2.0, 4.0], [10.0, 14.0]])
    val = np.asarray([[1.0, 3.0], [6.0, 10.0]])
    test = np.asarray([[5.0], [18.0]])

    train_scaled, val_scaled, test_scaled, scaler = _standardize_gnn_members_by_validation(
        train_matrix=train,
        val_matrix=val,
        test_matrix=test,
    )

    assert scaler["risk_means"] == pytest.approx([2.0, 8.0])
    assert scaler["risk_stds"] == pytest.approx([1.0, 2.0])
    np.testing.assert_allclose(val_scaled, [[-1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(train_scaled, [[0.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(test_scaled, [[3.0], [5.0]])


def test_bounded_mlp_residual_adapter_bounds_delta() -> None:
    model = BoundedMlpResidualAdapter(input_dim=4, max_delta=0.15)
    features = torch.randn(5, 4)
    baseline = torch.linspace(-1.0, 1.0, steps=5)

    risk, delta = model(features, baseline)

    assert risk.shape == baseline.shape
    assert delta.shape == baseline.shape
    assert float(delta.detach().abs().max()) <= 0.15


def test_disagreement_gated_adapter_bounds_delta_and_gate() -> None:
    model = DisagreementGatedResidualAdapter(input_dim=4, gate_dim=6, max_delta=0.2)
    features = torch.randn(5, 4)
    disagreement = torch.randn(5, 6)
    baseline = torch.linspace(-1.0, 1.0, steps=5)

    risk, delta, gate = model(features, disagreement, baseline)

    assert risk.shape == baseline.shape
    assert delta.shape == baseline.shape
    assert gate.shape == baseline.shape
    assert float(delta.detach().abs().max()) <= 0.2
    assert float(gate.detach().min()) >= 0.0
    assert float(gate.detach().max()) <= 1.0


def test_parse_list_helpers() -> None:
    assert _parse_int_list("7, 21,,42") == [7, 21, 42]
    assert _parse_float_list("0.2, 0.35,,0.5") == pytest.approx([0.2, 0.35, 0.5])
    assert _parse_str_list("raw_top3, standardized_top3,,") == ["raw_top3", "standardized_top3"]


def test_build_baseline_specs_rejects_unknown_mode() -> None:
    risk = np.asarray([1.0, 2.0])

    specs = _build_baseline_specs(
        baseline_modes=["raw_top3"],
        raw_top3_train_risk=risk,
        raw_top3_val_risk=risk,
        raw_top3_test_risk=risk,
        standardized_top3_train_risk=-risk,
        standardized_top3_val_risk=-risk,
        standardized_top3_test_risk=-risk,
    )

    assert specs[0]["name"] == "gnn_top3_raw"
    with pytest.raises(ValueError):
        _build_baseline_specs(
            baseline_modes=["bad_mode"],
            raw_top3_train_risk=risk,
            raw_top3_val_risk=risk,
            raw_top3_test_risk=risk,
            standardized_top3_train_risk=-risk,
            standardized_top3_val_risk=-risk,
            standardized_top3_test_risk=-risk,
        )


def test_build_baseline_specs_supports_softmax_ensemble() -> None:
    risk = np.asarray([1.0, 2.0])

    specs = _build_baseline_specs(
        baseline_modes=["softmax_ensemble"],
        raw_top3_train_risk=risk,
        raw_top3_val_risk=risk,
        raw_top3_test_risk=risk,
        standardized_top3_train_risk=risk,
        standardized_top3_val_risk=risk,
        standardized_top3_test_risk=risk,
        softmax_train_risk=-risk,
        softmax_val_risk=-risk,
        softmax_test_risk=-risk,
        softmax_temperature=0.003,
    )

    assert specs[0]["mode"] == "softmax_ensemble"
    assert specs[0]["name"] == "gnn_softmax_ensemble_t0p003"
    np.testing.assert_allclose(specs[0]["test_risk"], -risk)


def test_validation_softmax_weights_favor_best_member() -> None:
    weights = _validation_softmax_weights([0.70, 0.72, 0.71], temperature=0.003)

    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] > weights[2] > weights[0]
    with pytest.raises(ValueError, match="positive"):
        _validation_softmax_weights([0.70], temperature=0.0)
