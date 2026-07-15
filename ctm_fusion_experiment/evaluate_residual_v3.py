from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ctm_fusion_experiment.utils.metrics import summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


UTILITY_METRICS = (
    "top_event_lift",
    "high_low_event_gap",
    "risk_std",
    "time_separation",
)


def build_residual_v3_comparison_summary(output_dir: str | Path) -> dict[str, object]:
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
        aggressive = float(summary["residual_ctm_aggressive"]["test"]["c_index"])
        row: dict[str, Any] = {
            "fold": int(summary["fold"]),
            "baseline_c_index": baseline,
            "residual_raw_c_index": raw,
            "residual_aggressive_c_index": aggressive,
            "raw_delta": raw - baseline,
            "aggressive_delta": aggressive - baseline,
            "selected_alpha": summary["residual_ctm_aggressive"].get("selected_alpha"),
            "validation_objective": summary["residual_ctm_aggressive"].get("validation_objective"),
            "validation_c_index": summary["residual_ctm_aggressive"].get("validation_c_index"),
            "baseline_parameters": summary["baseline"].get("parameters"),
            "residual_parameters": summary["residual_ctm_raw"].get("parameters"),
            "graph_encoder_seconds": summary.get("graph_encoder", {}).get("training_seconds"),
            "baseline_seconds": summary["baseline"].get("training_seconds"),
            "residual_seconds": summary["residual_ctm_raw"].get("training_seconds"),
        }
        for metric in UTILITY_METRICS:
            row[f"baseline_{metric}"] = _utility(summary, "baseline", metric)
            row[f"raw_{metric}"] = _utility(summary, "residual_ctm_raw", metric)
            row[f"aggressive_{metric}"] = _utility(summary, "residual_ctm_aggressive", metric)
            row[f"raw_{metric}_delta"] = row[f"raw_{metric}"] - row[f"baseline_{metric}"]
            row[f"aggressive_{metric}_delta"] = row[f"aggressive_{metric}"] - row[f"baseline_{metric}"]
        rows.append(row)

    baseline_scores = [row["baseline_c_index"] for row in rows]
    raw_scores = [row["residual_raw_c_index"] for row in rows]
    aggressive_scores = [row["residual_aggressive_c_index"] for row in rows]
    result = {
        "num_folds": len(rows),
        "raw_paired_comparison": summarize_paired_folds(baseline=baseline_scores, ctm=raw_scores),
        "aggressive_paired_comparison": summarize_paired_folds(
            baseline=baseline_scores,
            ctm=aggressive_scores,
        ),
        "utility_paired_comparisons": _utility_comparisons(rows),
        "folds": rows,
        "interpretation": (
            "Residual v3 keeps the residual CTM architecture independent from previous flows, "
            "but selects the validation alpha with a composite objective: c-index improvement "
            "plus top-risk event lift, high/low event-rate separation, and risk-spread pressure. "
            "It is more aggressive than v2 and reports utility metrics beyond the original c-index."
        ),
        **metadata,
    }
    write_json(output_path / "residual_v3_comparison_summary.json", result)
    write_csv(output_path / "residual_v3_fold_comparison.csv", rows)
    return result


def _utility(summary: dict[str, Any], section: str, metric: str) -> float:
    utility = summary.get(section, {}).get("test_utility", {})
    return float(utility.get(metric, 0.0))


def _utility_comparisons(rows: list[dict[str, Any]]) -> dict[str, object]:
    comparisons = {}
    for metric in UTILITY_METRICS:
        baseline = [float(row[f"baseline_{metric}"]) for row in rows]
        raw = [float(row[f"raw_{metric}"]) for row in rows]
        aggressive = [float(row[f"aggressive_{metric}"]) for row in rows]
        comparisons[metric] = {
            "raw": summarize_paired_folds(baseline=baseline, ctm=raw),
            "aggressive": summarize_paired_folds(baseline=baseline, ctm=aggressive),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/residual_v3_formal")
    args = parser.parse_args()
    print(json.dumps(build_residual_v3_comparison_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
