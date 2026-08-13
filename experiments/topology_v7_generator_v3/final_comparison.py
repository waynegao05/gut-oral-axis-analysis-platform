from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/topology_v7_generator_v3_formal"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_final_comparison(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    manifest = _read(
        ROOT / "data/research/topology_v7_generator_v3/topology_v7_manifest.json"
    )
    light = _read(output_root / "fixed_protocol_logo/logo_benchmark_summary.json")
    gnn = _read(output_root / "gnn_locked_logo/formal_logo_gnn_summary.json")
    aft = _read(output_root / "aft_fusion_logo/formal_aft_fusion_logo_summary.json")
    legacy = _read(
        ROOT
        / "outputs/current_mainline_v2/temporal_independent_v3/cross_split_consensus/cross_split_consensus_summary.json"
    )

    light_aggregates = {
        (row["model_name"], row["feature_set"]): row
        for row in light["aggregates"]
        if row["dataset"] == "candidate_v3"
    }
    gnn_macro = float(gnn["aggregate_fold_ensembles"]["macro_mean_test_c_index"])
    gnn_pooled = float(
        gnn["aggregate_fold_ensembles"]["validation_standardized_pooled_oof_c_index"]
    )
    aft_aggregate = aft["aggregate"]
    aft_deltas = [float(row["test_c_index_delta"]) for row in aft["folds"]]
    t_test = ttest_1samp(aft_deltas, popmean=0.0)

    aft_gate_checks = {
        "macro_mean_c_index_delta_at_least_0_005": bool(
            float(aft_aggregate["macro_mean_test_c_index_delta"]) >= 0.005
        ),
        "at_least_four_of_five_folds_improve": bool(
            int(aft_aggregate["num_c_index_improved_folds"]) >= 4
        ),
        "worst_fold_delta_at_least_minus_0_005": bool(min(aft_deltas) >= -0.005),
        "mean_calibrated_cox_loss_does_not_worsen": bool(
            float(aft_aggregate["macro_mean_calibrated_cox_loss_delta"]) <= 0.0
        ),
    }
    legacy_c = float(legacy["aggregate"]["mean_selected_test_c_index"])
    result = {
        "schema_version": 1,
        "status": "complete",
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "formal_seed": int(manifest["seed"]),
            "sample_count": int(manifest["sample_count"]),
            "quality_gate_passed": bool(manifest["quality_gate"]["passed"]),
            "deterministic_latent_risk_c_index": float(
                manifest["generation"]["survival"]["deterministic_latent_risk_c_index"]
            ),
            "minimum_group_latent_risk_c_index": float(
                manifest["generation"]["survival"][
                    "minimum_generation_group_latent_risk_c_index"
                ]
            ),
            "source_grouped_domain_auc": float(
                light["fixed_protocol_comparison"]["domain_shift"]["candidate_mean_auc"]
            ),
            "reference_v2_source_grouped_domain_auc": float(
                light["fixed_protocol_comparison"]["domain_shift"]["reference_mean_auc"]
            ),
        },
        "fixed_protocol_models": {
            "linear_cox_macro_c_index": float(
                light_aggregates[("linear_cox", "edge_identity")]["mean_test_c_index"]
            ),
            "mlp_cox_macro_c_index": float(
                light_aggregates[("mlp_cox", "edge_identity")]["mean_test_c_index"]
            ),
            "xgb_aft_full_topology_macro_c_index": float(
                light_aggregates[("xgb_aft", "full_topology")]["mean_test_c_index"]
            ),
            "locked_gnn_five_seed_macro_c_index": gnn_macro,
            "locked_gnn_validation_standardized_pooled_oof_c_index": gnn_pooled,
            "validation_selected_aft_fusion_macro_c_index": float(
                aft_aggregate["macro_mean_selected_test_c_index"]
            ),
            "validation_selected_aft_fusion_pooled_oof_c_index": float(
                aft_aggregate["validation_standardized_pooled_oof_selected_c_index"]
            ),
        },
        "aft_fusion_decision": {
            "adopt": all(aft_gate_checks.values()),
            "checks": aft_gate_checks,
            "fold_deltas": aft_deltas,
            "macro_mean_c_index_delta": float(
                aft_aggregate["macro_mean_test_c_index_delta"]
            ),
            "macro_mean_calibrated_cox_loss_delta": float(
                aft_aggregate["macro_mean_calibrated_cox_loss_delta"]
            ),
            "paired_fold_t_test": {
                "statistic": float(t_test.statistic),
                "p_value": float(t_test.pvalue),
                "num_pairs": len(aft_deltas),
            },
            "decision": "reject_aft_fusion_keep_locked_gnn",
        },
        "legacy_context": {
            "topology_v6_aft_consensus_c_index": legacy_c,
            "v7_v3_locked_gnn_minus_legacy": gnn_macro - legacy_c,
            "protocol_comparability": "context_only_not_paired",
            "reason": (
                "The legacy value used two split seeds on topology_v6; V7 v3 uses five "
                "explicit leave-one-generation-group-out folds. A direct superiority claim "
                "is not valid without rerunning V6 under the same five-group protocol."
            ),
            "historical_exploratory_ceiling": 0.8967,
            "historical_exploratory_ceiling_status": (
                "retained as development potential only; not reproducible formal evidence"
            ),
        },
        "current_best": {
            "model": "locked five-seed identity full-risk Cox GNN",
            "macro_c_index": gnn_macro,
            "pooled_oof_c_index": gnn_pooled,
            "deployment_status": "research_candidate_not_deployed",
        },
        "conclusion": (
            "Generator-v3 fixes the V7 domain-shift failure and restores stable learnability. "
            "The locked GNN is the best accepted V7 v3 model. Validation-selected AFT fusion "
            "does not clear the predeclared gain and loss gates and is rejected."
        ),
    }
    report_dir = output_root / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "formal_model_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# topology_v7 generator-v3 formal comparison",
        "",
        f"- Dataset quality gate: passed; latent-risk C-index {result['dataset']['deterministic_latent_risk_c_index']:.6f}.",
        f"- Source-grouped domain AUC: {result['dataset']['source_grouped_domain_auc']:.4f} (V2 {result['dataset']['reference_v2_source_grouped_domain_auc']:.4f}).",
        f"- Locked five-seed GNN: macro C-index {gnn_macro:.6f}; pooled OOF {gnn_pooled:.6f}.",
        f"- AFT fusion: macro C-index {aft_aggregate['macro_mean_selected_test_c_index']:.6f}; delta {aft_aggregate['macro_mean_test_c_index_delta']:+.6f}.",
        f"- AFT pooled OOF: {aft_aggregate['validation_standardized_pooled_oof_selected_c_index']:.6f}; calibrated Cox-loss delta {aft_aggregate['macro_mean_calibrated_cox_loss_delta']:+.6f}.",
        "- Decision: reject AFT fusion and keep the locked GNN as the V7 v3 research candidate.",
        f"- Legacy V6 context: {legacy_c:.6f}; not paired with the five-group V7 v3 protocol.",
        "",
        "The historical 0.8967 result remains an exploratory ceiling only, not reproducible formal evidence.",
    ]
    (report_dir / "formal_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    print(json.dumps(build_final_comparison(), indent=2))


if __name__ == "__main__":
    main()
