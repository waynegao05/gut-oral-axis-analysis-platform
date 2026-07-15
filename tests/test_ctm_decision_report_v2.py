from __future__ import annotations

import json

import pytest

from research.ctm_decision_report_v2 import build_ctm_decision_report


def test_build_ctm_decision_report_recommends_mechanistic_role_when_ctm_is_weak(tmp_path) -> None:
    comparison = {
        "current_recommended": {
            "candidate_name": "adapter",
            "test_c_index": 0.742,
            "delta_vs_gnn_top3": 0.001,
            "delta_vs_previous_tamed_mlp": 0.0001,
        }
    }
    diagnostics = {
        "global_pair_change": {
            "corrected_pairs": 100,
            "harmed_pairs": 95,
            "net_corrected_pairs": 5,
        },
        "calibration_proxy": {
            "selected_delta_vs_baseline": {
                "top_event_lift_delta": 0.1,
                "high_low_event_gap_delta": 0.05,
                "risk_event_monotonic_spearman_delta": 0.0,
                "risk_time_monotonic_spearman_delta": 0.0,
            }
        },
        "continuous_net_reclassification_proxy": {"continuous_nri": 0.0},
        "subgroup_diagnostics": [
            {
                "feature": "risk_std_all",
                "bucket": "high_q75",
                "n": 20,
                "baseline_c_index": 0.7,
                "selected_c_index": 0.69,
                "c_index_delta": -0.01,
                "mean_abs_adapter_delta": 0.2,
            },
            {
                "feature": "risk_std_all",
                "bucket": "low_q25",
                "n": 20,
                "baseline_c_index": 0.7,
                "selected_c_index": 0.72,
                "c_index_delta": 0.02,
                "mean_abs_adapter_delta": 0.1,
            },
        ],
    }
    oof = _summary(mean_delta=0.001, p_value=0.001, fold_deltas=[0.001, 0.001])
    ctm = _summary(mean_delta=0.0002, p_value=0.5, fold_deltas=[0.001, -0.0006])
    ctm["oracle_best_seed_paired_comparison"] = ctm["selected_paired_comparison"]
    ctm["folds"] = [
        {"net_improved_pairs": 10, "validation_selected_delta": 0.01, "selected_delta": 0.001},
        {"net_improved_pairs": -8, "validation_selected_delta": 0.02, "selected_delta": -0.0006},
    ]

    paths = {
        "comparison": tmp_path / "comparison.json",
        "diagnostics": tmp_path / "diagnostics.json",
        "oof": tmp_path / "oof.json",
        "formal": tmp_path / "formal.json",
        "ctm": tmp_path / "ctm.json",
        "out": tmp_path / "out.json",
        "md": tmp_path / "out.md",
    }
    for key, payload in [
        ("comparison", comparison),
        ("diagnostics", diagnostics),
        ("oof", oof),
        ("formal", oof),
        ("ctm", ctm),
    ]:
        paths[key].write_text(json.dumps(payload), encoding="utf-8")

    report = build_ctm_decision_report(
        current_adapter_comparison_path=paths["comparison"],
        current_adapter_diagnostics_path=paths["diagnostics"],
        oof_repeated_summary_path=paths["oof"],
        oof_formal_summary_path=paths["formal"],
        ctm_residual_summary_paths=[paths["ctm"]],
        output_json_path=paths["out"],
        output_markdown_path=paths["md"],
    )

    assert report["decision"]["recommended_ctm_role"] == "mechanistic_residual_or_interpretability_component"
    assert report["decision"]["gating_requirements"]["require_disagreement_safe_gate"] is True
    assert paths["out"].exists()
    assert "CTM Decision Report" in paths["md"].read_text(encoding="utf-8")
    assert report["oof_evidence"]["repeated_cv"]["positive_delta_fraction"] == pytest.approx(1.0)


def _summary(mean_delta: float, p_value: float, fold_deltas: list[float]) -> dict:
    return {
        "num_folds": len(fold_deltas),
        "flow": "test_flow",
        "selected_paired_comparison": {
            "baseline": {"mean": 0.7, "std": 0.0},
            "ctm": {"mean": 0.7 + mean_delta, "std": 0.0},
            "fold_deltas": fold_deltas,
            "mean_delta": mean_delta,
            "paired_t_test": {"p_value": p_value},
        },
    }
