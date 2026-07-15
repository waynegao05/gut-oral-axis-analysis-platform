from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_residual_v2_comparison_summary(output_dir: str | Path) -> dict[str, object]:
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
        raw = float(summary["residual_ctm_raw"]["test"]["c_index"])
        calibrated = float(summary["residual_ctm_calibrated"]["test"]["c_index"])
        rows.append(
            {
                "fold": int(summary["fold"]),
                "baseline_c_index": baseline,
                "residual_raw_c_index": raw,
                "residual_calibrated_c_index": calibrated,
                "raw_delta": raw - baseline,
                "calibrated_delta": calibrated - baseline,
                "selected_alpha": summary["residual_ctm_calibrated"].get("selected_alpha"),
                "baseline_parameters": summary["baseline"].get("parameters"),
                "residual_parameters": summary["residual_ctm_raw"].get("parameters"),
                "graph_encoder_seconds": summary.get("graph_encoder", {}).get("training_seconds"),
                "baseline_seconds": summary["baseline"].get("training_seconds"),
                "residual_seconds": summary["residual_ctm_raw"].get("training_seconds"),
            }
        )

    baseline_scores = [row["baseline_c_index"] for row in rows]
    raw_scores = [row["residual_raw_c_index"] for row in rows]
    calibrated_scores = [row["residual_calibrated_c_index"] for row in rows]
    result = {
        "num_folds": len(rows),
        "raw_paired_comparison": summarize_paired_folds(baseline=baseline_scores, ctm=raw_scores),
        "calibrated_paired_comparison": summarize_paired_folds(
            baseline=baseline_scores,
            ctm=calibrated_scores,
        ),
        "folds": rows,
        "interpretation": (
            "Residual v2 trains CTM as a gated correction and then selects a safe residual "
            "alpha on each fold's validation split. alpha=0 is allowed, so calibrated "
            "performance can fall back to the concat baseline."
        ),
        **metadata,
    }
    write_json(output_path / "residual_v2_comparison_summary.json", result)
    write_csv(output_path / "residual_v2_fold_comparison.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/residual_v2_formal")
    args = parser.parse_args()
    print(json.dumps(build_residual_v2_comparison_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
