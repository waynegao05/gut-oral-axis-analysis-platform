from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def summarize_ablation(run_specs: Sequence[str], output_path: str | Path) -> dict[str, Any]:
    rows = []
    for spec in run_specs:
        parts = spec.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("Each run spec must be split_seed:feature_set:summary_path.")
        split_seed, feature_set, summary_path = parts
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        rows.append(
            {
                "split_seed": int(split_seed),
                "feature_set": feature_set,
                "num_features": int(summary["feature_metadata"]["num_features"]),
                "selected_expert": summary["selected_expert"]["name"],
                "expert_validation_c_index": float(summary["validation"]["expert_c_index"]),
                "expert_test_c_index": float(summary["test"]["expert_c_index"]),
                "selected_alpha": float(summary["blend_selection"]["selected"]["alpha"]),
                "selected_validation_c_index": float(summary["validation"]["selected_c_index"]),
                "selected_test_c_index": float(summary["test"]["selected_c_index"]),
                "test_c_index_delta": float(summary["test"]["selected_c_index_delta"]),
                "test_calibrated_cox_loss_delta": float(summary["test"]["calibrated_cox_loss_delta"]),
                "summary_path": str(Path(summary_path).as_posix()),
            }
        )

    feature_sets = sorted({row["feature_set"] for row in rows})
    aggregate = {}
    for feature_set in feature_sets:
        selected = [row for row in rows if row["feature_set"] == feature_set]
        aggregate[feature_set] = {
            "num_splits": len(selected),
            "num_features": sorted({row["num_features"] for row in selected}),
            "mean_selected_test_c_index": float(
                np.mean([row["selected_test_c_index"] for row in selected])
            ),
            "mean_test_c_index_delta": float(np.mean([row["test_c_index_delta"] for row in selected])),
            "mean_test_calibrated_cox_loss_delta": float(
                np.mean([row["test_calibrated_cox_loss_delta"] for row in selected])
            ),
        }
    result = {
        "protocol": "seed7_feature_information_ablation",
        "selection_note": (
            "All feature sets use the same censor-aware AFT objective; differences isolate input information value."
        ),
        "rows": rows,
        "aggregate": aggregate,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize_ablation(args.runs, args.output)


if __name__ == "__main__":
    main()
