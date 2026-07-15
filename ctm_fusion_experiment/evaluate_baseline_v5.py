from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_baseline_v5_comparison_summary(output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    metadata_path = output_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    fold_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_path.glob("fold_*/fold_summary.json"))
    ]
    if not fold_summaries:
        raise ValueError(f"No fold summaries found under {output_path.as_posix()}.")

    rows = []
    for summary in fold_summaries:
        reference = float(summary["reference_baseline"]["test"]["c_index"])
        selected = float(summary["baseline_v5_selected"]["test"]["c_index"])
        candidate_scores = [float(candidate["test"]["c_index"]) for candidate in summary["candidate_models"]]
        oracle = max(candidate_scores)
        diagnostics = summary["baseline_v5_selected"]["pair_diagnostics"]
        rows.append(
            {
                "fold": int(summary["fold"]),
                "reference_c_index": reference,
                "selected_c_index": selected,
                "selected_delta": selected - reference,
                "oracle_best_single_c_index": oracle,
                "oracle_best_single_delta": oracle - reference,
                "selected_candidate": summary["baseline_v5_selected"]["candidate_name"],
                "selected_weights": json.dumps(summary["baseline_v5_selected"]["weights"]),
                "validation_reference_c_index": summary["baseline_v5_selected"]["validation_reference_c_index"],
                "validation_selected_c_index": summary["baseline_v5_selected"]["validation_c_index"],
                "validation_selected_delta": (
                    summary["baseline_v5_selected"]["validation_c_index"]
                    - summary["baseline_v5_selected"]["validation_reference_c_index"]
                ),
                "candidate_count": len(summary["candidate_models"]),
                "improved_pairs": diagnostics["improved_pairs"],
                "regressed_pairs": diagnostics["regressed_pairs"],
                "net_improved_pairs": diagnostics["net_improved_pairs"],
                "pair_credit_delta": diagnostics["pair_credit_delta"],
                "graph_encoder_seconds_total": sum(
                    float(graph.get("training_seconds", 0.0))
                    for graph in summary["graph_encoders"]
                ),
                "baseline_seconds_total": sum(
                    float(candidate.get("training_seconds", 0.0))
                    for candidate in summary["candidate_models"]
                ),
            }
        )

    reference_scores = [row["reference_c_index"] for row in rows]
    selected_scores = [row["selected_c_index"] for row in rows]
    oracle_scores = [row["oracle_best_single_c_index"] for row in rows]
    result = {
        "num_folds": len(rows),
        "selected_paired_comparison": summarize_paired_folds(
            baseline=reference_scores,
            ctm=selected_scores,
        ),
        "oracle_best_single_paired_comparison": summarize_paired_folds(
            baseline=reference_scores,
            ctm=oracle_scores,
        ),
        "folds": rows,
        "interpretation": (
            "Baseline v5 targets c-index without CTM: it trains multiple supervised graph-embedding "
            "and baseline-Cox seeds, selects standardized risk ensembles by validation c-index, "
            "and compares them against a fold-local reference concat baseline."
        ),
        **metadata,
    }
    write_json(output_path / "baseline_v5_comparison_summary.json", result)
    write_csv(output_path / "baseline_v5_fold_comparison.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/baseline_v5_formal")
    args = parser.parse_args()
    print(json.dumps(build_baseline_v5_comparison_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
