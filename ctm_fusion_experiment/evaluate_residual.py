from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_residual_comparison_summary(output_dir: str | Path) -> dict[str, object]:
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
        baseline_c_index = float(summary["baseline"]["test"]["c_index"])
        residual_c_index = float(summary["residual_ctm"]["test"]["c_index"])
        rows.append(
            {
                "fold": int(summary["fold"]),
                "baseline_c_index": baseline_c_index,
                "residual_ctm_c_index": residual_c_index,
                "delta": residual_c_index - baseline_c_index,
                "baseline_parameters": summary["baseline"].get("parameters"),
                "residual_ctm_parameters": summary["residual_ctm"].get("parameters"),
                "graph_encoder_seconds": summary.get("graph_encoder", {}).get("training_seconds"),
                "baseline_seconds": summary["baseline"].get("training_seconds"),
                "residual_ctm_seconds": summary["residual_ctm"].get("training_seconds"),
                "mean_residual_gate": summary["residual_ctm"].get("mean_residual_gate"),
            }
        )

    comparison = summarize_paired_folds(
        baseline=[row["baseline_c_index"] for row in rows],
        ctm=[row["residual_ctm_c_index"] for row in rows],
    )
    result = {
        "num_folds": len(rows),
        "paired_comparison": comparison,
        "folds": rows,
        "interpretation": (
            "Residual CTM keeps the fold-local concat-Cox baseline intact and trains CTM only as "
            "a gated correction. This run is independent from the original CTM replacement flow."
        ),
        **metadata,
    }
    write_json(output_path / "residual_comparison_summary.json", result)
    write_csv(output_path / "residual_fold_comparison.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/residual_formal")
    args = parser.parse_args()
    print(json.dumps(build_residual_comparison_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
