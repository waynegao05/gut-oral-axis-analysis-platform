from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def build_ctm_decision_report(
    *,
    current_adapter_comparison_path: str | Path,
    current_adapter_diagnostics_path: str | Path,
    oof_repeated_summary_path: str | Path,
    oof_formal_summary_path: str | Path,
    ctm_residual_summary_paths: Sequence[str | Path],
    output_json_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    current_comparison = _read_json(current_adapter_comparison_path)
    current_diagnostics = _read_json(current_adapter_diagnostics_path)
    oof_repeated = _read_json(oof_repeated_summary_path)
    oof_formal = _read_json(oof_formal_summary_path)
    ctm_summaries = [_read_json(path) for path in ctm_residual_summary_paths]

    current = _summarize_current_adapter(current_comparison, current_diagnostics)
    oof = {
        "repeated_cv": _summarize_paired_comparison(oof_repeated, comparison_key="selected_paired_comparison"),
        "formal_oof": _summarize_paired_comparison(oof_formal, comparison_key="selected_paired_comparison"),
    }
    ctm = [_summarize_ctm_summary(summary, str(path)) for summary, path in zip(ctm_summaries, ctm_residual_summary_paths)]
    decision = _make_decision(current=current, oof=oof, ctm=ctm)
    report = {
        "current_adapter_evidence": current,
        "oof_evidence": oof,
        "ctm_residual_evidence": ctm,
        "decision": decision,
        "next_experiment_spec": _next_experiment_spec(),
        "interpretation": (
            "This report combines fixed-split risk-adapter diagnostics, existing fold-local OOF ensemble results, "
            "and residual CTM fold summaries. It is intended to decide whether CTM should be a primary predictor "
            "or a constrained mechanistic residual module before further training is launched."
        ),
    }
    if output_json_path is not None:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if output_markdown_path is not None:
        path = Path(output_markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _summarize_current_adapter(comparison: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    selected = comparison["current_recommended"]
    pair_change = diagnostics["global_pair_change"]
    calibration_delta = diagnostics["calibration_proxy"]["selected_delta_vs_baseline"]
    nri = diagnostics["continuous_net_reclassification_proxy"]
    best_subgroups = diagnostics["subgroup_diagnostics"][:5]
    worst_subgroups = sorted(diagnostics["subgroup_diagnostics"], key=lambda row: row["c_index_delta"])[:5]
    return {
        "candidate_name": selected["candidate_name"],
        "test_c_index": selected["test_c_index"],
        "delta_vs_gnn_top3": selected["delta_vs_gnn_top3"],
        "delta_vs_previous_tamed_mlp": selected["delta_vs_previous_tamed_mlp"],
        "pair_change": pair_change,
        "calibration_delta_vs_baseline": calibration_delta,
        "continuous_nri_proxy": nri,
        "best_subgroups": _compact_subgroups(best_subgroups),
        "worst_subgroups": _compact_subgroups(worst_subgroups),
        "evidence_flags": {
            "pair_changes_mostly_cancel": pair_change["corrected_pairs"] > 0
            and pair_change["harmed_pairs"] / max(pair_change["corrected_pairs"], 1) > 0.8,
            "large_delta_bucket_is_harmful": any(
                row["feature"] == "abs_adapter_delta"
                and row["bucket"] == "high_q75"
                and row["c_index_delta"] < 0.0
                for row in diagnostics["subgroup_diagnostics"]
            ),
            "high_disagreement_is_harmful": any(
                row["feature"] in {"risk_std_all", "risk_std_top3", "risk_range_top3"}
                and row["bucket"] == "high_q75"
                and row["c_index_delta"] < 0.0
                for row in diagnostics["subgroup_diagnostics"]
            ),
        },
    }


def _summarize_paired_comparison(summary: dict[str, Any], *, comparison_key: str) -> dict[str, Any]:
    comparison = summary[comparison_key]
    fold_deltas = [float(value) for value in comparison["fold_deltas"]]
    return {
        "flow": summary.get("flow"),
        "num_folds": int(summary.get("num_repeated_cv_folds", summary.get("num_folds", len(fold_deltas)))),
        "baseline_mean": float(comparison["baseline"]["mean"]),
        "selected_mean": float(comparison["ctm"]["mean"]),
        "mean_delta": float(comparison["mean_delta"]),
        "p_value": float(comparison["paired_t_test"]["p_value"]),
        "positive_delta_fraction": float(np.mean(np.asarray(fold_deltas) > 0.0)),
        "negative_delta_fraction": float(np.mean(np.asarray(fold_deltas) < 0.0)),
        "fold_deltas": fold_deltas,
    }


def _summarize_ctm_summary(summary: dict[str, Any], source_path: str) -> dict[str, Any]:
    selected = _summarize_paired_comparison(summary, comparison_key="selected_paired_comparison")
    oracle = _summarize_paired_comparison(summary, comparison_key="oracle_best_seed_paired_comparison")
    folds = summary.get("folds", [])
    net_pairs = [float(row.get("net_improved_pairs", 0.0)) for row in folds]
    val_deltas = [float(row.get("validation_selected_delta", 0.0)) for row in folds]
    test_deltas = [float(row.get("selected_delta", 0.0)) for row in folds]
    return {
        "source_path": source_path,
        "flow": summary.get("flow"),
        "selected": selected,
        "oracle_best_seed": oracle,
        "net_pair_mean": float(np.mean(net_pairs)) if net_pairs else 0.0,
        "net_pair_sum": float(np.sum(net_pairs)) if net_pairs else 0.0,
        "validation_test_delta_spearman": _spearman(np.asarray(val_deltas), np.asarray(test_deltas)),
        "selected_positive_fold_fraction": selected["positive_delta_fraction"],
        "selected_negative_fold_fraction": selected["negative_delta_fraction"],
    }


def _make_decision(*, current: dict[str, Any], oof: dict[str, Any], ctm: list[dict[str, Any]]) -> dict[str, Any]:
    ctm_best_delta = max((row["selected"]["mean_delta"] for row in ctm), default=0.0)
    ctm_best_p = min((row["selected"]["p_value"] for row in ctm), default=1.0)
    high_disagreement_harmful = bool(current["evidence_flags"]["high_disagreement_is_harmful"])
    pair_cancel = bool(current["evidence_flags"]["pair_changes_mostly_cancel"])
    oof_stable = (
        oof["repeated_cv"]["mean_delta"] > 0.0
        and oof["repeated_cv"]["positive_delta_fraction"] >= 0.9
        and oof["repeated_cv"]["p_value"] < 0.01
    )
    if ctm_best_delta <= 0.001 or ctm_best_p >= 0.05 or high_disagreement_harmful:
        role = "mechanistic_residual_or_interpretability_component"
    else:
        role = "candidate_primary_module"
    return {
        "recommended_ctm_role": role,
        "should_ctm_replace_main_predictor_now": role == "candidate_primary_module",
        "should_continue_ctm": True,
        "rationale": [
            "OOF/shrinkage ensemble evidence is stable but the mean gain is small; this supports conservative stacking rather than aggressive selection.",
            "Current adapter gains are real but mostly local: corrected and harmed pair counts are close, so unconstrained residual capacity is risky.",
            "High-disagreement subgroups are harmed by the current adapter; the next CTM must explicitly gate or shrink residuals in these regions.",
            "Residual CTM fold summaries do not yet show a statistically reliable mean c-index gain over the concat baseline.",
        ],
        "gating_requirements": {
            "require_oof_selection": True,
            "require_disagreement_safe_gate": high_disagreement_harmful,
            "require_delta_magnitude_penalty": pair_cancel,
            "require_subgroup_noninferiority_on_high_disagreement": True,
            "minimum_evidence_to_promote_to_primary": {
                "oof_mean_delta": 0.003,
                "paired_p_value_max": 0.05,
                "high_disagreement_c_index_delta_min": 0.0,
                "net_pair_delta_min": 0.0,
            },
        },
        "evidence_summary": {
            "oof_repeated_is_stable": oof_stable,
            "best_ctm_selected_mean_delta": ctm_best_delta,
            "best_ctm_selected_p_value": ctm_best_p,
            "current_adapter_delta_vs_gnn_top3": current["delta_vs_gnn_top3"],
            "current_adapter_net_corrected_pairs": current["pair_change"]["net_corrected_pairs"],
        },
    }


def _next_experiment_spec() -> dict[str, Any]:
    return {
        "name": "oof_disagreement_safe_structured_ctm_residual",
        "training_protocol": [
            "Use fold-local OOF predictions to train and select residual policies; do not select on a single validation split.",
            "Keep an untouched outer test fold for final reporting.",
            "Allow alpha=0 fallback and require high-disagreement subgroup non-inferiority.",
        ],
        "ctm_inputs": [
            "GNN member risk vector",
            "GNN member disagreement features",
            "graph_embedding mean/std across GNN members",
            "latent fusion embedding mean/std across GNN members",
            "graph topology targets and cluster targets",
            "clinical/metabolite/graph modality conflict features",
        ],
        "metrics": [
            "global c-index",
            "subgroup c-index by disagreement/conflict strata",
            "pair correction / harm / net corrected pairs",
            "risk quantile calibration proxy",
            "continuous NRI proxy",
            "delta magnitude and high-disagreement non-inferiority",
        ],
        "model_constraints": [
            "bounded residual delta",
            "disagreement-safe gate initialized toward no correction",
            "delta L2 penalty",
            "distillation to top3/reference risk",
            "optional CTM attention only over structured tokens, not raw flattened summaries alone",
        ],
    }


def _compact_subgroups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "feature": row["feature"],
            "bucket": row["bucket"],
            "n": int(row["n"]),
            "baseline_c_index": float(row["baseline_c_index"]),
            "selected_c_index": float(row["selected_c_index"]),
            "c_index_delta": float(row["c_index_delta"]),
            "mean_abs_adapter_delta": float(row["mean_abs_adapter_delta"]),
        }
        for row in rows
    ]


def _render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    current = report["current_adapter_evidence"]
    oof = report["oof_evidence"]["repeated_cv"]
    ctm_rows = report["ctm_residual_evidence"]
    lines = [
        "# CTM Decision Report v2",
        "",
        f"Recommended CTM role: **{decision['recommended_ctm_role']}**",
        "",
        "## Current Adapter",
        "",
        f"- Test c-index: {current['test_c_index']:.9f}",
        f"- Delta vs GNN top3: {current['delta_vs_gnn_top3']:.9f}",
        f"- Net corrected pairs: {current['pair_change']['net_corrected_pairs']}",
        f"- Corrected / harmed pairs: {current['pair_change']['corrected_pairs']} / {current['pair_change']['harmed_pairs']}",
        "",
        "## OOF Evidence",
        "",
        f"- Repeated-CV OOF mean delta: {oof['mean_delta']:.9f}",
        f"- Repeated-CV paired p-value: {oof['p_value']:.3g}",
        f"- Positive fold fraction: {oof['positive_delta_fraction']:.2f}",
        "",
        "## CTM Residual Evidence",
        "",
    ]
    for row in ctm_rows:
        source = Path(row["source_path"])
        label = f"{source.parent.name}/{source.name}"
        lines.append(
            f"- {label}: selected mean delta {row['selected']['mean_delta']:.9f}, "
            f"p={row['selected']['p_value']:.3g}, net pairs sum={row['net_pair_sum']:.1f}"
        )
    lines.extend(
        [
            "",
            "## Rationale",
            "",
            *[f"- {item}" for item in decision["rationale"]],
            "",
            "## Next Experiment",
            "",
            f"- {report['next_experiment_spec']['name']}",
            "- Use OOF selection, structured GNN embeddings, disagreement-safe gating, and subgroup non-inferiority.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return 0.0
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-adapter-comparison",
        default="outputs/current_mainline_v2/risk_adapter_v2_dual_baseline/risk_adapter_v2_comparison_summary.json",
    )
    parser.add_argument(
        "--current-adapter-diagnostics",
        default="outputs/current_mainline_v2/risk_adapter_diagnostics_v2/risk_adapter_diagnostics_v2_summary.json",
    )
    parser.add_argument(
        "--oof-repeated-summary",
        default=(
            "outputs/ctm_fusion_experiment/baseline_v10_shrinkage_softmax_alpha_0p1_5seed_7_21_42_123_2026/"
            "baseline_v10_shrinkage_softmax_alpha_0p1_5seed_7_21_42_123_2026_summary.json"
        ),
    )
    parser.add_argument(
        "--oof-formal-summary",
        default="outputs/ctm_fusion_experiment/baseline_v9_oof_formal/baseline_v9_oof_summary.json",
    )
    parser.add_argument(
        "--ctm-residual-summaries",
        default=(
            "outputs/ctm_fusion_experiment/residual_v4_formal/residual_v4_comparison_summary.json,"
            "outputs/ctm_fusion_experiment/residual_v4_hardpair_formal/residual_v4_comparison_summary.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        default="outputs/current_mainline_v2/ctm_decision_report_v2/ctm_decision_report_v2.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/current_mainline_v2/ctm_decision_report_v2/ctm_decision_report_v2.md",
    )
    args = parser.parse_args()
    report = build_ctm_decision_report(
        current_adapter_comparison_path=args.current_adapter_comparison,
        current_adapter_diagnostics_path=args.current_adapter_diagnostics,
        oof_repeated_summary_path=args.oof_repeated_summary,
        oof_formal_summary_path=args.oof_formal_summary,
        ctm_residual_summary_paths=[part.strip() for part in args.ctm_residual_summaries.split(",") if part.strip()],
        output_json_path=args.output_json,
        output_markdown_path=args.output_md,
    )
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
