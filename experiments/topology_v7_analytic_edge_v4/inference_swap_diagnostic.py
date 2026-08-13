from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from experiments.topology_v7_analytic_edge_v4.model import (
    AnalyticInternalRelationModel,
    evaluate_edge_emulation,
    fit_analytic_edge_parameters,
)
from experiments.topology_v7_internal_relation_site_v3.runner import (
    _load_config,
    _metric_report,
    _predict,
    build_refit_bundle,
)
from research.ensemble_v2 import build_model
from research.train_v2 import resolve_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    ROOT
    / "outputs/topology_v7_internal_relation_site_v3/cohorts/"
    "development_seed20261010"
)
DEFAULT_MODEL_ROOT = (
    ROOT
    / "outputs/topology_v7_internal_relation_site_v3/development/"
    "legacy_precomputed_edge_gnn"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/topology_v7_analytic_edge_v4/diagnostics/"
    "inference_swap_summary.json"
)
DEFAULT_PLAN = (
    ROOT
    / "experiments/topology_v7_internal_relation_site_v3/"
    "experiment_plan.json"
)


def run_diagnostic(
    *,
    data_dir: Path,
    model_root: Path,
    plan_path: Path,
    output: Path,
    device_arg: str,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = _load_config(ROOT / plan["base_config"], data_dir)
    device = resolve_device(device_arg)
    folds: list[dict[str, Any]] = []
    for outer_group in range(5):
        inner_group = (outer_group + 1) % 5
        bundle = build_refit_bundle(
            config,
            data_dir=data_dir,
            outer_test_group=outer_group,
        )
        base_model = build_model(config, bundle, device)
        checkpoint_path = (
            model_root
            / f"outer_group{outer_group}_val{inner_group}"
            / "seed42/model.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        base_model.load_state_dict(checkpoint["state_dict"])
        legacy_train = _predict(
            base_model,
            bundle.train_set,
            device=device,
        )
        legacy_test = _predict(
            base_model,
            bundle.test_set,
            device=device,
        )
        legacy_metrics = _metric_report(
            reference=legacy_train,
            evaluation=legacy_test,
            plan=plan,
        )

        parameters = fit_analytic_edge_parameters(
            bundle.train_set,
            num_node_types=bundle.num_node_types,
        )
        analytic_model = AnalyticInternalRelationModel(
            base_model,
            num_node_types=bundle.num_node_types,
            site_feature_dim=bundle.site_feature_dim,
            parameters=parameters,
            use_linear_site_residual=False,
        ).to(device)
        analytic_train = _predict(
            analytic_model,
            bundle.train_set,
            device=device,
        )
        analytic_test = _predict(
            analytic_model,
            bundle.test_set,
            device=device,
        )
        analytic_metrics = _metric_report(
            reference=analytic_train,
            evaluation=analytic_test,
            plan=plan,
        )
        edge_test = evaluate_edge_emulation(
            analytic_model.edge_generator,
            bundle.test_set,
            device=device,
        )
        folds.append(
            {
                "outer_test_group": outer_group,
                "legacy_c_index": legacy_metrics["harrell_c_index"],
                "analytic_swap_c_index": analytic_metrics[
                    "harrell_c_index"
                ],
                "c_index_delta": (
                    analytic_metrics["harrell_c_index"]
                    - legacy_metrics["harrell_c_index"]
                ),
                "legacy_integrated_auc": legacy_metrics[
                    "normalized_integrated_auc"
                ],
                "analytic_swap_integrated_auc": analytic_metrics[
                    "normalized_integrated_auc"
                ],
                "integrated_auc_delta": (
                    analytic_metrics["normalized_integrated_auc"]
                    - legacy_metrics["normalized_integrated_auc"]
                ),
                "edge_train_fit": parameters.fit_report,
                "edge_test_fit": edge_test,
            }
        )
        print(
            f"outer={outer_group} "
            f"edgeR2={edge_test['r2']:.4f} "
            f"Cdelta={folds[-1]['c_index_delta']:+.6f} "
            f"iAUCdelta={folds[-1]['integrated_auc_delta']:+.6f}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "scope": "development_checkpoint_inference_swap_diagnostic",
        "outcomes_used_to_fit_edge_layer": False,
        "model_retraining_performed": False,
        "folds": folds,
        "macro_legacy_c_index": statistics.mean(
            row["legacy_c_index"] for row in folds
        ),
        "macro_analytic_swap_c_index": statistics.mean(
            row["analytic_swap_c_index"] for row in folds
        ),
        "macro_c_index_delta": statistics.mean(
            row["c_index_delta"] for row in folds
        ),
        "macro_integrated_auc_delta": statistics.mean(
            row["integrated_auc_delta"] for row in folds
        ),
        "mean_test_edge_r2": statistics.mean(
            row["edge_test_fit"]["r2"] for row in folds
        ),
        "mean_test_edge_mae": statistics.mean(
            row["edge_test_fit"]["mae"] for row in folds
        ),
        "may_define_future_fresh_cohort_protocol": True,
        "may_not_claim_independent_validation": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    args = parser.parse_args()
    result = run_diagnostic(
        data_dir=args.data_dir.resolve(),
        model_root=args.model_root.resolve(),
        plan_path=args.plan.resolve(),
        output=args.output.resolve(),
        device_arg=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
