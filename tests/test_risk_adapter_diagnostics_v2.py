from __future__ import annotations

import numpy as np
import pytest

from research.risk_adapter_diagnostics_v2 import (
    _load_summary_predictions,
    _modality_conflict_features,
    _rankdata,
    _risk_quantile_bins,
    _spearman,
    calibration_proxy,
    continuous_nri_proxy,
    pairwise_change_diagnostics,
)


def test_load_summary_predictions_prefers_selection_reference() -> None:
    predictions = _load_summary_predictions(
        {
            "test_predictions": [
                {
                    "sample_id": "a",
                    "time": 1.0,
                    "event": 1.0,
                    "raw_mean_risk": 0.1,
                    "gnn_top3_risk": 0.2,
                    "selection_reference_risk": 0.3,
                    "selected_risk": 0.4,
                }
            ]
        }
    )

    assert predictions["baseline_risk_field"] == "selection_reference_risk"
    assert predictions["selection_reference_risk"].tolist() == pytest.approx([0.3])


def test_pairwise_change_diagnostics_counts_corrected_and_harmed_pairs() -> None:
    sample_ids = ["a", "b", "c"]
    time = np.asarray([1.0, 2.0, 3.0])
    event = np.asarray([1.0, 1.0, 0.0])
    baseline = np.asarray([0.2, 0.3, 0.1])
    selected = np.asarray([0.4, 0.2, 0.1])

    summary, sample_rows = pairwise_change_diagnostics(
        sample_ids=sample_ids,
        time=time,
        event=event,
        baseline_risk=baseline,
        selected_risk=selected,
    )

    assert summary.permissible_pairs == 3
    assert summary.corrected_pairs == 1
    assert summary.harmed_pairs == 0
    assert summary.net_corrected_pairs == 1
    assert summary.baseline_c_index == pytest.approx(2.0 / 3.0)
    assert summary.selected_c_index == pytest.approx(1.0)
    assert sample_rows[0]["corrected_pairs"] == 1
    assert sample_rows[1]["corrected_pairs"] == 1


def test_pairwise_change_diagnostics_handles_ties() -> None:
    sample_ids = ["a", "b"]
    time = np.asarray([1.0, 2.0])
    event = np.asarray([1.0, 0.0])
    baseline = np.asarray([0.5, 0.5])
    selected = np.asarray([0.6, 0.5])

    summary, _ = pairwise_change_diagnostics(
        sample_ids=sample_ids,
        time=time,
        event=event,
        baseline_risk=baseline,
        selected_risk=selected,
    )

    assert summary.baseline_c_index == pytest.approx(0.5)
    assert summary.selected_c_index == pytest.approx(1.0)
    assert summary.tied_to_correct_pairs == 1


def test_rankdata_uses_average_ranks_for_ties() -> None:
    assert _rankdata(np.asarray([3.0, 1.0, 1.0, 2.0])).tolist() == pytest.approx([4.0, 1.5, 1.5, 3.0])


def test_spearman_reports_monotonic_correlation() -> None:
    x = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert _spearman(x, x) == pytest.approx(1.0)
    assert _spearman(x, -x) == pytest.approx(-1.0)


def test_modality_conflict_features_builds_gap_proxies() -> None:
    feature_names = [
        "clinical:age",
        "clinical:bmi",
        "metabolite:scfa",
        "graph:density",
        "graph:edge",
    ]
    features = np.asarray(
        [
            [1.0, 3.0, 0.0, 2.0, 4.0],
            [2.0, 2.0, 5.0, 1.0, 1.0],
        ]
    )

    result = _modality_conflict_features(feature_names, features)

    assert result["modality_score_clinical"] == pytest.approx([2.0, 2.0])
    assert result["modality_score_metabolite"] == pytest.approx([0.0, 5.0])
    assert result["modality_score_graph"] == pytest.approx([3.0, 1.0])
    assert result["abs_clinical_metabolite_gap"] == pytest.approx([2.0, 3.0])
    assert result["modality_conflict_range"] == pytest.approx([3.0, 4.0])


def test_risk_quantile_bins_assigns_ordered_bins() -> None:
    bins = _risk_quantile_bins(np.asarray([0.4, 0.1, 0.3, 0.2]), num_bins=2)

    assert bins.tolist() == [1, 0, 1, 0]


def test_calibration_proxy_reports_top_event_lift_and_monotonicity() -> None:
    time = np.asarray([10.0, 9.0, 4.0, 3.0])
    event = np.asarray([0.0, 0.0, 1.0, 1.0])
    risk = np.asarray([0.1, 0.2, 0.8, 0.9])

    result = calibration_proxy(time=time, event=event, risk=risk, num_bins=2)

    assert result["bins"][0]["event_rate"] == pytest.approx(0.0)
    assert result["bins"][1]["event_rate"] == pytest.approx(1.0)
    assert result["top_event_lift"] == pytest.approx(2.0)
    assert result["high_low_event_gap"] == pytest.approx(1.0)
    assert result["risk_event_monotonic_spearman"] == pytest.approx(1.0)
    assert result["risk_time_monotonic_spearman"] == pytest.approx(1.0)


def test_continuous_nri_proxy_rewards_event_up_and_nonevent_down() -> None:
    event = np.asarray([1.0, 1.0, 0.0, 0.0])
    baseline = np.asarray([0.2, 0.5, 0.8, 0.4])
    selected = np.asarray([0.3, 0.4, 0.7, 0.5])

    result = continuous_nri_proxy(event=event, baseline_risk=baseline, selected_risk=selected)

    assert result["event_up_rate"] == pytest.approx(0.5)
    assert result["event_down_rate"] == pytest.approx(0.5)
    assert result["nonevent_down_rate"] == pytest.approx(0.5)
    assert result["nonevent_up_rate"] == pytest.approx(0.5)
    assert result["continuous_nri"] == pytest.approx(0.0)
