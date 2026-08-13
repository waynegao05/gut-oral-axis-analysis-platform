from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.topology_v7_exact_edge_v5.model import (
    ExactInternalEdgeGenerator,
    evaluate_edge_emulation,
    fit_exact_edge_parameters,
)
from experiments.topology_v7_internal_relation_site_v3 import (
    runner as core,
)


ROOT = Path(__file__).resolve().parents[2]


def run_diagnostic(
    *,
    data_dir: Path,
    plan_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan = core._read_json(plan_path)
    config = core._load_config(
        ROOT / str(plan["base_config"]),
        data_dir,
    )
    site_table = core.build_site_feature_table(
        data_dir / core.ORAL_GUT_FILE
    )
    folds: list[dict[str, Any]] = []
    for outer_group in plan["outer_test_groups"]:
        bundle = core.build_refit_bundle(
            config,
            data_dir=data_dir,
            outer_test_group=int(outer_group),
            site_table=site_table,
        )
        parameters = fit_exact_edge_parameters(
            bundle.train_set,
            num_node_types=bundle.num_node_types,
        )
        generator = ExactInternalEdgeGenerator(
            num_node_types=bundle.num_node_types,
            parameters=parameters,
        )
        test_metrics = evaluate_edge_emulation(
            generator,
            bundle.test_set,
            device=torch.device("cpu"),
        )
        folds.append(
            {
                "outer_test_group": int(outer_group),
                "training_fit": parameters.fit_report,
                "held_out_test_fit": test_metrics,
                "train_count": len(bundle.train_set),
                "test_count": len(bundle.test_set),
            }
        )

    minimum_r2 = min(
        row["held_out_test_fit"]["r2"] for row in folds
    )
    maximum_mae = max(
        row["held_out_test_fit"]["mae"] for row in folds
    )
    threshold = float(
        plan["edge_fidelity_gate"][
            "minimum_held_out_r2"
        ]
    )
    summary = {
        "schema_version": 1,
        "status": (
            "passed"
            if minimum_r2 >= threshold
            else "failed"
        ),
        "scope": "outcome_free_fold_local_edge_fidelity",
        "minimum_held_out_r2": float(minimum_r2),
        "maximum_held_out_mae": float(maximum_mae),
        "required_minimum_held_out_r2": threshold,
        "uses_time_or_event": False,
        "folds": folds,
    }
    core._write_json(output_path, summary)
    if summary["status"] != "passed":
        raise RuntimeError(
            "Exact internal edge fidelity gate failed."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_diagnostic(
        data_dir=args.data_dir,
        plan_path=args.plan,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "minimum_held_out_r2": summary[
                    "minimum_held_out_r2"
                ],
                "maximum_held_out_mae": summary[
                    "maximum_held_out_mae"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
