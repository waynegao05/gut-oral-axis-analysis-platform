from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from experiments.topology_v7_exact_edge_v5.model import (
    ExactInternalRelationModel,
    fit_exact_edge_parameters,
)
from experiments.topology_v7_internal_relation_site_v3 import (
    runner as core,
)
from experiments.topology_v7_rank_objective_v6.losses import (
    comparable_pair_logistic_loss,
    horizon_pair_logistic_loss,
)
from research.ensemble_v2 import build_model
from research.losses import cox_ph_loss
from research.train_v2 import (
    build_scheduler,
    resolve_device,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "prescreen_plan.json"


def _build_model(
    config: dict[str, Any],
    bundle: core.NestedBundle,
    device: torch.device,
) -> ExactInternalRelationModel:
    base_model = build_model(
        copy.deepcopy(config),
        bundle,
        device,
    )
    parameters = fit_exact_edge_parameters(
        bundle.train_set,
        num_node_types=bundle.num_node_types,
    )
    return ExactInternalRelationModel(
        base_model,
        num_node_types=bundle.num_node_types,
        parameters=parameters,
    ).to(device)


def _objective(
    model: torch.nn.Module,
    batch: Any,
    *,
    candidate: dict[str, Any],
    horizons: list[float],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch, compute_contrastive=False)
    cox = cox_ph_loss(
        output["risk"],
        batch.time,
        batch.event,
        ties_method="breslow",
    )
    comparable = comparable_pair_logistic_loss(
        output["risk"],
        batch.time,
        batch.event,
        temperature=temperature,
    )
    horizon = horizon_pair_logistic_loss(
        output["risk"],
        batch.time,
        batch.event,
        horizons=horizons,
        temperature=temperature,
    )
    total = (
        cox
        + float(candidate["comparable_rank_weight"])
        * comparable
        + float(candidate["horizon_rank_weight"])
        * horizon
    )
    return total, {
        "cox": float(cox.detach().item()),
        "comparable_rank": float(
            comparable.detach().item()
        ),
        "horizon_rank": float(horizon.detach().item()),
    }


def _train_inner(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    bundle: core.NestedBundle,
    candidate: dict[str, Any],
    device_arg: str,
) -> dict[str, Any]:
    training = plan["training"]
    device = resolve_device(device_arg)
    core.set_seed(int(plan["model_seed"]))
    model = _build_model(config, bundle, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=int(training["maximum_epochs"]),
        warmup_epochs=int(training["warmup_epochs"]),
    )
    loader = DataLoader(
        bundle.train_set,
        batch_size=len(bundle.train_set),
        shuffle=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_c_index = float("-inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(
        1,
        int(training["maximum_epochs"]) + 1,
    ):
        model.train()
        batch = next(iter(loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        objective, components = _objective(
            model,
            batch,
            candidate=candidate,
            horizons=[
                float(value)
                for value in plan["metrics"][
                    "report_horizons"
                ]
            ],
            temperature=float(training["temperature"]),
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(
                training["gradient_clip_norm"]
            ),
        )
        optimizer.step()
        scheduler.step()
        validation = core._predict(
            model,
            bundle.val_set,
            device=device,
        )
        validation_c_index = float(
            validation["c_index"]
        )
        history.append(
            {
                "epoch": int(epoch),
                "objective": float(
                    objective.detach().item()
                ),
                "cox": components["cox"],
                "comparable_rank": components[
                    "comparable_rank"
                ],
                "horizon_rank": components[
                    "horizon_rank"
                ],
                "validation_c_index": validation_c_index,
            }
        )
        if (
            validation_c_index
            > best_c_index
            + float(
                training[
                    "minimum_validation_c_index_delta"
                ]
            )
        ):
            best_c_index = validation_c_index
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if (
            epoch >= int(training["warmup_epochs"])
            and patience
            >= int(training["early_stop_patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("V6 prescreen produced no checkpoint.")
    model.load_state_dict(best_state)
    train_prediction = core._predict(
        model,
        bundle.train_set,
        device=device,
    )
    validation_prediction = core._predict(
        model,
        bundle.val_set,
        device=device,
    )
    metrics = core._metric_report(
        reference=train_prediction,
        evaluation=validation_prediction,
        plan=plan,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "inner_validation_group": (
            bundle.inner_validation_group
        ),
        "selection_train_groups": (
            bundle.split_summary["train_groups"]
        ),
        "best_epoch": best_epoch,
        "metrics": metrics,
        "training_seconds": float(
            time.perf_counter() - started
        ),
        "history": history,
    }


def run_prescreen(
    *,
    data_dir: Path,
    plan_path: Path,
    output_path: Path,
    device_arg: str,
) -> dict[str, Any]:
    plan = core._read_json(plan_path)
    manifest = core._read_json(
        data_dir / core.DATA_FILE_NAMES["manifest_json"]
    )
    if int(manifest["seed"]) != int(
        plan["source_development_cohort_seed"]
    ):
        raise RuntimeError("V6 prescreen cohort seed mismatch.")
    config = core._load_config(
        ROOT / str(plan["base_config"]),
        data_dir,
    )
    site_table = core.build_site_feature_table(
        data_dir / core.ORAL_GUT_FILE
    )
    outer_group = int(plan["outer_group_never_evaluated"])
    results: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        inner_runs: list[dict[str, Any]] = []
        for inner_group in plan["inner_validation_groups"]:
            bundle = core.build_nested_bundle(
                config,
                data_dir=data_dir,
                outer_test_group=outer_group,
                inner_validation_group=int(inner_group),
                site_table=site_table,
            )
            run = _train_inner(
                config=config,
                plan=plan,
                bundle=bundle,
                candidate=candidate,
                device_arg=device_arg,
            )
            inner_runs.append(run)
            print(
                f"{candidate['name']} inner={inner_group} "
                f"C={run['metrics']['harrell_c_index']:.6f} "
                f"iAUC={run['metrics']['normalized_integrated_auc']:.6f} "
                f"epoch={run['best_epoch']}",
                flush=True,
            )
        results.append(
            {
                "candidate": candidate,
                "mean_inner_validation_c_index": float(
                    statistics.mean(
                        row["metrics"]["harrell_c_index"]
                        for row in inner_runs
                    )
                ),
                "mean_inner_validation_integrated_auc": float(
                    statistics.mean(
                        row["metrics"][
                            "normalized_integrated_auc"
                        ]
                        for row in inner_runs
                    )
                ),
                "median_best_epoch": int(
                    round(
                        statistics.median(
                            row["best_epoch"]
                            for row in inner_runs
                        )
                    )
                ),
                "inner_runs": inner_runs,
            }
        )
    baseline = next(
        row
        for row in results
        if row["candidate"]["name"] == "cox_only"
    )
    eligible = [
        row
        for row in results
        if (
            row["mean_inner_validation_c_index"]
            - baseline["mean_inner_validation_c_index"]
            >= float(
                plan["selection_rule"][
                    "minimum_c_index_gain_over_cox"
                ]
            )
        )
    ]
    selected = max(
        eligible or [baseline],
        key=lambda row: (
            row["mean_inner_validation_c_index"],
            row[
                "mean_inner_validation_integrated_auc"
            ],
        ),
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "inner_validation_only_prescreen_"
            "outer_group_zero_never_evaluated"
        ),
        "plan_sha256": core._sha256(plan_path),
        "manifest_sha256": core._sha256(
            data_dir / core.DATA_FILE_NAMES[
                "manifest_json"
            ]
        ),
        "outer_group_read_for_prediction_or_metrics": False,
        "results": results,
        "selected_candidate": selected["candidate"],
    }
    core._write_json(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    args = parser.parse_args()
    result = run_prescreen(
        data_dir=args.data_dir,
        plan_path=args.plan,
        output_path=args.output,
        device_arg=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_candidate": result[
                    "selected_candidate"
                ],
                "outer_group_read_for_prediction_or_metrics": (
                    result[
                        "outer_group_read_for_prediction_or_metrics"
                    ]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
