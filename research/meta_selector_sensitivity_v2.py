from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


HYBRID_OOF_DELTA_WEIGHT = 0.25
SELECTION_POLICIES = ("validation_then_oof", "oof_then_validation", "hybrid_validation_oof")


def build_selector_sensitivity_report(
    *,
    summary_paths: Sequence[str | Path],
    min_oof_deltas: Sequence[float] = (0.0, 0.00001, 0.00003, 0.00005, 0.0001),
    min_val_delta: float = 0.0,
    min_high_disagreement_val_delta: float = 0.0,
    selection_policy: str = "validation_then_oof",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    reports = [
        _analyze_summary(
            summary_path=Path(summary_path),
            min_oof_deltas=min_oof_deltas,
            min_val_delta=float(min_val_delta),
            min_high_disagreement_val_delta=float(min_high_disagreement_val_delta),
            selection_policy=selection_policy,
        )
        for summary_path in summary_paths
    ]
    result = {
        "selection_policy": selection_policy,
        "min_oof_deltas": [float(value) for value in min_oof_deltas],
        "min_val_delta": float(min_val_delta),
        "min_high_disagreement_val_delta": float(min_high_disagreement_val_delta),
        "reports": reports,
        "interpretation": (
            "Selector sensitivity reuses saved candidate tables and recomputes which candidate would be selected "
            "under stricter OOF/validation gates. The main_model fallback is always eligible."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _analyze_summary(
    *,
    summary_path: Path,
    min_oof_deltas: Sequence[float],
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
    selection_policy: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(summary["selected"]["candidate_table"])
    original = {
        key: summary["selected"].get(key)
        for key in (
            "candidate_name",
            "alpha",
            "oof_delta",
            "val_delta",
            "high_disagreement_val_delta",
            "test_c_index",
            "test_delta_vs_main",
        )
    }
    threshold_results = [
        _select_for_threshold(
            rows,
            min_oof_delta=float(min_oof_delta),
            min_val_delta=min_val_delta,
            min_high_disagreement_val_delta=min_high_disagreement_val_delta,
            selection_policy=selection_policy,
        )
        for min_oof_delta in min_oof_deltas
    ]
    return {
        "summary_path": str(summary_path),
        "references": summary.get("references", {}),
        "original_selected": original,
        "threshold_results": threshold_results,
    }


def _select_for_threshold(
    rows: Sequence[dict[str, Any]],
    *,
    min_oof_delta: float,
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
    selection_policy: str,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if _is_eligible(
            row,
            min_oof_delta=min_oof_delta,
            min_val_delta=min_val_delta,
            min_high_disagreement_val_delta=min_high_disagreement_val_delta,
        )
    ]
    if not eligible:
        raise ValueError("No eligible rows; candidate table does not contain a main_model fallback.")
    if selection_policy == "validation_then_oof":
        selected = max(
            eligible,
            key=lambda row: (
                float(row["val_c_index"]),
                float(row["high_disagreement_val_delta"]),
                float(row["oof_c_index"]),
            ),
        )
    elif selection_policy == "oof_then_validation":
        selected = max(
            eligible,
            key=lambda row: (
                float(row["oof_c_index"]),
                float(row["val_c_index"]),
                float(row["high_disagreement_val_delta"]),
            ),
        )
    elif selection_policy == "hybrid_validation_oof":
        selected = max(
            eligible,
            key=lambda row: (
                float(row["val_delta"]) + HYBRID_OOF_DELTA_WEIGHT * float(row["oof_delta"]),
                float(row["val_c_index"]),
                float(row["oof_c_index"]),
                float(row["high_disagreement_val_delta"]),
            ),
        )
    else:
        raise ValueError(f"Unknown selection_policy: {selection_policy}")
    selected_compact = {
        "candidate_name": selected["candidate_name"],
        "alpha": float(selected["alpha"]),
        "oof_delta": float(selected["oof_delta"]),
        "val_delta": float(selected["val_delta"]),
        "high_disagreement_val_delta": float(selected["high_disagreement_val_delta"]),
        "test_c_index": float(selected["test_c_index"]),
    }
    return {
        "min_oof_delta": float(min_oof_delta),
        "eligible_count": len(eligible),
        "selected": selected_compact,
    }


def _is_eligible(
    row: dict[str, Any],
    *,
    min_oof_delta: float,
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
) -> bool:
    if row.get("candidate_name") == "main_model":
        return True
    return (
        float(row["oof_delta"]) >= float(min_oof_delta)
        and float(row["val_delta"]) >= float(min_val_delta)
        and float(row["high_disagreement_val_delta"]) >= float(min_high_disagreement_val_delta)
    )


def _parse_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--min-oof-deltas", default="0,0.00001,0.00003,0.00005,0.0001")
    parser.add_argument("--min-val-delta", type=float, default=0.0)
    parser.add_argument("--min-high-disagreement-val-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="validation_then_oof",
    )
    parser.add_argument("--output", default="outputs/current_mainline_v2/meta_selector_sensitivity_v2/report.json")
    args = parser.parse_args()
    result = build_selector_sensitivity_report(
        summary_paths=_parse_paths(args.summaries),
        min_oof_deltas=_parse_floats(args.min_oof_deltas),
        min_val_delta=args.min_val_delta,
        min_high_disagreement_val_delta=args.min_high_disagreement_val_delta,
        selection_policy=args.selection_policy,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
