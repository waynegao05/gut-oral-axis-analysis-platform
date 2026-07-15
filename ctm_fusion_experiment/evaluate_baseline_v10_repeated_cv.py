from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_repeated_cv_summary(
    runs: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    output_stem: str = "baseline_v10_5seed_repeated_cv",
    interpretation: str | None = None,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    run_names = []
    policy_name: str | None = None
    shrinkage_alpha: float | None = None

    for run_name, summary_path in runs:
        path = Path(summary_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_names.append(run_name)
        policy_name = str(summary.get("policy_name", policy_name))
        if "shrinkage_alpha" in summary:
            shrinkage_alpha = float(summary["shrinkage_alpha"])
        for row in summary["folds"]:
            rows.append(
                {
                    "run": run_name,
                    "fold": int(row["fold"]),
                    "reference_c_index": float(row["reference_c_index"]),
                    "selected_c_index": float(row["selected_c_index"]),
                    "selected_delta": float(row["selected_delta"]),
                    "policy_name": row.get("policy_name", policy_name),
                    "shrinkage_alpha": row.get("shrinkage_alpha", shrinkage_alpha),
                    "net_improved_pairs": float(row.get("net_improved_pairs", 0.0)),
                    "pair_credit_delta": float(row.get("pair_credit_delta", 0.0)),
                }
            )

    if not rows:
        raise ValueError("No fold rows found in repeated-CV summaries.")

    reference_scores = [float(row["reference_c_index"]) for row in rows]
    selected_scores = [float(row["selected_c_index"]) for row in rows]
    result = {
        "runs": run_names,
        "num_repeated_cv_folds": len(rows),
        "policy_name": policy_name,
        "shrinkage_alpha": shrinkage_alpha,
        "selected_paired_comparison": summarize_paired_folds(reference_scores, selected_scores),
        "folds": rows,
        "interpretation": interpretation
        or (
            "Repeated-CV stability summary over multiple independently seeded 5-fold runs. "
            "Folds overlap in samples, so this is a robustness audit rather than an independent-sample "
            "significance test."
        ),
    }
    write_json(output_path / f"{output_stem}_summary.json", result)
    write_csv(output_path / f"{output_stem}_folds.csv", rows)
    return result


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as name=summary.json")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", default="baseline_v10_5seed_repeated_cv")
    args = parser.parse_args()
    print(
        json.dumps(
            build_repeated_cv_summary(
                args.run,
                args.output_dir,
                output_stem=args.output_stem,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
