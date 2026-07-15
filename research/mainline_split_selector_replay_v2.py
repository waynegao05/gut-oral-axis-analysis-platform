from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from research.meta_selector_sensitivity_v2 import SELECTION_POLICIES, _select_for_threshold


def build_split_selector_replay_report(
    *,
    summary_paths: Sequence[str | Path],
    min_oof_delta: float = 0.00003,
    min_val_delta: float = 0.0,
    min_high_disagreement_val_delta: float = 0.0005,
    selection_policy: str = "hybrid_validation_oof",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = [
        _replay_one(
            Path(summary_path),
            min_oof_delta=float(min_oof_delta),
            min_val_delta=float(min_val_delta),
            min_high_disagreement_val_delta=float(min_high_disagreement_val_delta),
            selection_policy=selection_policy,
        )
        for summary_path in summary_paths
    ]
    deltas = [float(row["selected_test_delta_vs_main"]) for row in rows]
    result = {
        "selection_policy": selection_policy,
        "min_oof_delta": float(min_oof_delta),
        "min_val_delta": float(min_val_delta),
        "min_high_disagreement_val_delta": float(min_high_disagreement_val_delta),
        "num_splits": len(rows),
        "mean_selected_delta_vs_main": sum(deltas) / len(deltas) if deltas else None,
        "num_positive_selected_deltas": sum(1 for value in deltas if value > 0.0),
        "min_selected_delta_vs_main": min(deltas) if deltas else None,
        "max_selected_delta_vs_main": max(deltas) if deltas else None,
        "rows": rows,
        "interpretation": (
            "This report replays the selector against saved candidate tables. It does not retrain base models "
            "or meta residual models."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _replay_one(
    summary_path: Path,
    *,
    min_oof_delta: float,
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
    selection_policy: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    references = summary["references"]
    selected = _select_for_threshold(
        summary["selected"]["candidate_table"],
        min_oof_delta=min_oof_delta,
        min_val_delta=min_val_delta,
        min_high_disagreement_val_delta=min_high_disagreement_val_delta,
        selection_policy=selection_policy,
    )
    replay_selected = selected["selected"]
    test_main = float(references["test_main_c_index"])
    original_selected = summary["selected"]
    return {
        "split_seed": _infer_split_seed(summary_path),
        "source_summary_path": str(summary_path.as_posix()),
        "test_main_c_index": test_main,
        "original_selection_policy": summary.get("selection_policy"),
        "original_selected_candidate": original_selected["candidate_name"],
        "original_selected_alpha": float(original_selected["alpha"]),
        "original_selected_test_c_index": float(original_selected["test_c_index"]),
        "original_selected_test_delta_vs_main": float(original_selected["test_delta_vs_main"]),
        "eligible_count": int(selected["eligible_count"]),
        "selected_candidate": replay_selected["candidate_name"],
        "selected_alpha": float(replay_selected["alpha"]),
        "selected_oof_delta": float(replay_selected["oof_delta"]),
        "selected_val_delta": float(replay_selected["val_delta"]),
        "selected_high_disagreement_val_delta": float(replay_selected["high_disagreement_val_delta"]),
        "selected_test_c_index": float(replay_selected["test_c_index"]),
        "selected_test_delta_vs_main": float(replay_selected["test_c_index"]) - test_main,
    }


def _infer_split_seed(path: Path) -> int | None:
    match = re.search(r"split_seed_(\d+)", str(path))
    return int(match.group(1)) if match else None


def _parse_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--min-oof-delta", type=float, default=0.00003)
    parser.add_argument("--min-val-delta", type=float, default=0.0)
    parser.add_argument("--min-high-disagreement-val-delta", type=float, default=0.0005)
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="hybrid_validation_oof",
    )
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/mainline_split_selector_replay_v2/report.json",
    )
    args = parser.parse_args()
    result = build_split_selector_replay_report(
        summary_paths=_parse_paths(args.summaries),
        min_oof_delta=args.min_oof_delta,
        min_val_delta=args.min_val_delta,
        min_high_disagreement_val_delta=args.min_high_disagreement_val_delta,
        selection_policy=args.selection_policy,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
