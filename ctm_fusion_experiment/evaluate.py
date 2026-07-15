from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_comparison_summary(output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    metadata_path = output_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    previous_summary_path = output_path / "comparison_summary.json"
    previous_summary = (
        json.loads(previous_summary_path.read_text(encoding="utf-8"))
        if previous_summary_path.exists()
        else {}
    )
    fold_summaries = []
    for fold_file in sorted(output_path.glob("fold_*/fold_summary.json")):
        fold_summaries.append(json.loads(fold_file.read_text(encoding="utf-8")))
    if not fold_summaries:
        raise ValueError(f"No fold summaries found under {output_path.as_posix()}.")

    rows = []
    for summary in fold_summaries:
        baseline_c_index = float(summary["baseline"]["test"]["c_index"])
        ctm_c_index = float(summary["ctm"]["test"]["c_index"])
        rows.append(
            {
                "fold": int(summary["fold"]),
                "baseline_c_index": baseline_c_index,
                "ctm_c_index": ctm_c_index,
                "delta": ctm_c_index - baseline_c_index,
                "baseline_parameters": summary["baseline"].get("parameters"),
                "ctm_parameters": summary["ctm"].get("parameters"),
                "graph_encoder_seconds": summary.get("graph_encoder", {}).get("training_seconds"),
                "baseline_seconds": summary["baseline"].get("training_seconds"),
                "ctm_seconds": summary["ctm"].get("training_seconds"),
            }
        )

    comparison = summarize_paired_folds(
        baseline=[row["baseline_c_index"] for row in rows],
        ctm=[row["ctm_c_index"] for row in rows],
    )
    result = {
        "num_folds": len(rows),
        "paired_comparison": comparison,
        "folds": rows,
        "interpretation": (
            "This paired comparison changes only the fusion model after fold-local graph-only "
            "GNN pretraining and frozen embedding export. topology_v6 is synthetic/noisy augmented "
            "research data and is not a clinical benchmark."
        ),
        **metadata,
    }
    if "plots" in previous_summary:
        result["plots"] = previous_summary["plots"]
    write_json(output_path / "comparison_summary.json", result)
    write_csv(output_path / "fold_comparison.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/formal")
    args = parser.parse_args()
    print(json.dumps(build_comparison_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
