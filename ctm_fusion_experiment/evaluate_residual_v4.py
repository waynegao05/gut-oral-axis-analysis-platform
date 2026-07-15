from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_residual_v4_comparison_summary(output_dir: str | Path) -> dict[str, object]:
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
        baseline = float(summary["baseline"]["test"]["c_index"])
        selected = float(summary["residual_ctm_v4_selected"]["test"]["c_index"])
        seed_scores = [
            float(seed_summary["test"]["c_index"])
            for seed_summary in summary["residual_ctm_seeds"]
        ]
        diagnostics = summary["residual_ctm_v4_selected"]["pair_diagnostics"]
        rows.append(
            {
                "fold": int(summary["fold"]),
                "baseline_c_index": baseline,
                "selected_c_index": selected,
                "selected_delta": selected - baseline,
                "best_seed_c_index": max(seed_scores),
                "best_seed_delta": max(seed_scores) - baseline,
                "selected_candidate": summary["residual_ctm_v4_selected"]["candidate_name"],
                "selected_weights": json.dumps(summary["residual_ctm_v4_selected"]["weights"]),
                "validation_baseline_c_index": summary["residual_ctm_v4_selected"]["validation_baseline_c_index"],
                "validation_selected_c_index": summary["residual_ctm_v4_selected"]["validation_c_index"],
                "validation_selected_delta": (
                    summary["residual_ctm_v4_selected"]["validation_c_index"]
                    - summary["residual_ctm_v4_selected"]["validation_baseline_c_index"]
                ),
                "improved_pairs": diagnostics["improved_pairs"],
                "regressed_pairs": diagnostics["regressed_pairs"],
                "net_improved_pairs": diagnostics["net_improved_pairs"],
                "pair_credit_delta": diagnostics["pair_credit_delta"],
                "baseline_parameters": summary["baseline"].get("parameters"),
                "residual_parameters_per_seed": summary["residual_ctm_seeds"][0].get("parameters"),
                "num_residual_seeds": len(summary["residual_ctm_seeds"]),
                "graph_encoder_seconds": summary.get("graph_encoder", {}).get("training_seconds"),
                "baseline_seconds": summary["baseline"].get("training_seconds"),
                "residual_seconds_total": sum(
                    float(seed_summary.get("training_seconds", 0.0))
                    for seed_summary in summary["residual_ctm_seeds"]
                ),
            }
        )

    baseline_scores = [row["baseline_c_index"] for row in rows]
    selected_scores = [row["selected_c_index"] for row in rows]
    best_seed_scores = [row["best_seed_c_index"] for row in rows]
    result = {
        "num_folds": len(rows),
        "selected_paired_comparison": summarize_paired_folds(
            baseline=baseline_scores,
            ctm=selected_scores,
        ),
        "oracle_best_seed_paired_comparison": summarize_paired_folds(
            baseline=baseline_scores,
            ctm=best_seed_scores,
        ),
        "folds": rows,
        "interpretation": (
            "Residual v4 targets c-index directly: multiple residual CTM seeds are trained per fold, "
            "candidate single-seed and ensemble deltas are selected only by validation c-index, and "
            "test pair diagnostics report how many comparable pairs were fixed or regressed."
        ),
        **metadata,
    }
    write_json(output_path / "residual_v4_comparison_summary.json", result)
    write_csv(output_path / "residual_v4_fold_comparison.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/residual_v4_formal")
    args = parser.parse_args()
    print(json.dumps(build_residual_v4_comparison_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
