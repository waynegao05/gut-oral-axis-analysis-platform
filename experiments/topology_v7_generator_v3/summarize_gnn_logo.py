from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from research.metrics import concordance_index


SEEDS = [7, 21, 42, 123, 2026]


def summarize_logo(output_root: Path) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    pooled_time: list[float] = []
    pooled_event: list[float] = []
    pooled_standardized_risk: list[float] = []

    for test_group in range(5):
        validation_group = (test_group + 1) % 5
        fold_root = output_root / (
            f"outer_group{test_group}_val{validation_group}_five_seed"
        )
        repeat = json.loads(
            (fold_root / "research_repeat_runs_summary.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (fold_root / "val_ensemble_summary.json").read_text(encoding="utf-8")
        )
        test = json.loads(
            (fold_root / "test_ensemble_summary.json").read_text(encoding="utf-8")
        )
        if repeat["seeds"] != SEEDS:
            raise RuntimeError(f"Unexpected model seeds in outer group {test_group}.")
        if str(test["test_group"]) != str(test_group):
            raise RuntimeError(f"Test-group mismatch in outer group {test_group}.")
        if str(test["validation_group"]) != str(validation_group):
            raise RuntimeError(f"Validation-group mismatch in outer group {test_group}.")

        validation_risk = np.asarray(
            [row["ensemble_risk"] for row in validation["predictions"]], dtype=float
        )
        validation_mean = float(validation_risk.mean())
        validation_std = float(validation_risk.std())
        if validation_std < 1e-8:
            raise RuntimeError(f"Validation risk has zero variance in outer group {test_group}.")
        test_risk = np.asarray(
            [row["ensemble_risk"] for row in test["predictions"]], dtype=float
        )
        standardized_test_risk = (test_risk - validation_mean) / validation_std
        pooled_standardized_risk.extend(standardized_test_risk.tolist())
        pooled_time.extend(float(row["time"]) for row in test["predictions"])
        pooled_event.extend(float(row["event"]) for row in test["predictions"])

        all_runs.extend(repeat["runs"])
        folds.append(
            {
                "outer_test_group": test_group,
                "inner_validation_group": validation_group,
                "mean_member_test_c_index": float(repeat["mean_test_c_index"]),
                "std_member_test_c_index": float(repeat["std_test_c_index"]),
                "min_member_test_c_index": float(repeat["min_test_c_index"]),
                "max_member_test_c_index": float(repeat["max_test_c_index"]),
                "ensemble_test_c_index": float(test["ensemble_c_index"]),
                "ensemble_validation_c_index": float(validation["ensemble_c_index"]),
                "ensemble_gain_over_member_mean": float(
                    test["ensemble_c_index"] - repeat["mean_test_c_index"]
                ),
                "mean_test_loss": float(repeat["mean_test_loss"]),
            }
        )

    member_scores = [float(row["test_c_index"]) for row in all_runs]
    fold_ensemble_scores = [row["ensemble_test_c_index"] for row in folds]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "dataset": "topology_v7_generator_v3_formal_seed_20261001",
            "outer_test_groups": list(range(5)),
            "inner_validation_group_rule": "(outer_test_group + 1) modulo 5",
            "training_seeds": SEEDS,
            "num_independent_runs": len(all_runs),
            "architecture": "locked_identity_fullrisk_cox_gnn",
            "test_labels_used_for_selection": False,
            "pooled_oof_calibration": "standardize each outer test risk with its validation risk mean and standard deviation",
        },
        "folds": folds,
        "aggregate_independent_runs": {
            "mean_test_c_index": float(statistics.mean(member_scores)),
            "std_test_c_index": float(statistics.stdev(member_scores)),
            "min_test_c_index": float(min(member_scores)),
            "max_test_c_index": float(max(member_scores)),
        },
        "aggregate_fold_ensembles": {
            "macro_mean_test_c_index": float(statistics.mean(fold_ensemble_scores)),
            "macro_std_test_c_index": float(statistics.stdev(fold_ensemble_scores)),
            "minimum_fold_test_c_index": float(min(fold_ensemble_scores)),
            "maximum_fold_test_c_index": float(max(fold_ensemble_scores)),
            "validation_standardized_pooled_oof_c_index": float(
                concordance_index(
                    pooled_time,
                    pooled_event,
                    pooled_standardized_risk,
                )
            ),
        },
    }
    output_path = output_root / "formal_logo_gnn_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the locked five-group GNN protocol.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/topology_v7_generator_v3_formal/gnn_locked_logo"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize_logo(args.output_root), indent=2))


if __name__ == "__main__":
    main()
