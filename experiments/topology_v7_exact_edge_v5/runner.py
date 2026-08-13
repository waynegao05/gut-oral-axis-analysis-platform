from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import torch
from torch_geometric.loader import DataLoader

from experiments.topology_v7_exact_edge_v5.model import (
    ExactInternalRelationModel,
    fit_exact_edge_parameters,
)
from experiments.topology_v7_internal_relation_site_v3 import runner as core
from research.ensemble_v2 import build_model
from research.losses import cox_ph_loss
from research.train_v2 import build_scheduler, resolve_device


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/topology_v7_exact_edge_v5"


def _build_candidate_model(
    *,
    config: dict[str, Any],
    bundle: core.NestedBundle | core.RefitBundle,
    candidate: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float] | None]:
    run_config = copy.deepcopy(config)
    base_model = build_model(run_config, bundle, device)
    if candidate["model_family"] == "legacy":
        return base_model, None
    if candidate["model_family"] != "exact_internal_relation":
        raise ValueError(
            f"Unsupported V5 model family: {candidate['model_family']}"
        )
    parameters = fit_exact_edge_parameters(
        bundle.train_set,
        num_node_types=bundle.num_node_types,
    )
    model = ExactInternalRelationModel(
        base_model,
        num_node_types=bundle.num_node_types,
        parameters=parameters,
    ).to(device)
    return model, parameters.fit_report


def _objective(
    model: torch.nn.Module,
    batch: Any,
    *,
    candidate: dict[str, Any],
    training: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(batch, compute_contrastive=False)
    cox = cox_ph_loss(
        output["risk"],
        batch.time,
        batch.event,
        ties_method="breslow",
    )
    residual_penalty = cox.new_zeros(())
    return cox + residual_penalty, cox, residual_penalty


def train_one(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    bundle: core.NestedBundle,
    refit_bundle: core.RefitBundle,
    candidate: dict[str, Any],
    model_seed: int,
    device_arg: str,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "run_summary.json"
    checkpoint_path = output_dir / "model.pt"
    prediction_path = output_dir / "predictions.npz"
    training = plan["training"]
    fingerprint = core._fingerprint(
        {
            "runner_sha256": core._sha256(Path(__file__)),
            "model_sha256": core._sha256(EXPERIMENT_DIR / "model.py"),
            "plan_sha256": core._sha256(plan_path),
            "base_config_sha256": core._sha256(
                ROOT / str(plan["base_config"])
            ),
            "candidate": candidate,
            "model_seed": int(model_seed),
            "outer_test_group": bundle.outer_test_group,
            "inner_validation_group": bundle.inner_validation_group,
            "training": training,
            "selection_train_ids": core._sample_ids(bundle.train_set),
            "selection_validation_ids": core._sample_ids(bundle.val_set),
            "refit_train_ids": core._sample_ids(refit_bundle.train_set),
            "test_ids": core._sample_ids(refit_bundle.test_set),
            "selection_standardizers": bundle.standardizers,
            "refit_standardizers": refit_bundle.standardizers,
        }
    )
    if (
        resume
        and summary_path.exists()
        and checkpoint_path.exists()
        and prediction_path.exists()
    ):
        result = core._read_json(summary_path)
        if result.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Refusing mismatched V5 resume artifacts: {output_dir}"
            )
        return result

    device = resolve_device(device_arg)
    core.set_seed(int(model_seed))
    selection_model, selection_edge_fit = _build_candidate_model(
        config=config,
        bundle=bundle,
        candidate=candidate,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        selection_model.parameters(),
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
    best_validation_c_index = float("-inf")
    best_epoch = 0
    patience = 0
    selection_history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        selection_model.train()
        batch = next(iter(loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        objective, cox, penalty = _objective(
            selection_model,
            batch,
            candidate=candidate,
            training=training,
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            selection_model.parameters(),
            max_norm=float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        validation = core._predict(
            selection_model,
            bundle.val_set,
            device=device,
        )
        selection_history.append(
            {
                "epoch": int(epoch),
                "train_objective": float(objective.item()),
                "train_cox_loss": float(cox.item()),
                "auxiliary_penalty": float(penalty.item()),
                "validation_c_index": float(validation["c_index"]),
                "validation_cox_loss": float(
                    validation["cox_loss"]
                ),
            }
        )
        if (
            validation["c_index"]
            > best_validation_c_index
            + float(training["minimum_validation_c_index_delta"])
        ):
            best_validation_c_index = float(validation["c_index"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in selection_model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(training["early_stop_patience"]):
                break
    if best_state is None:
        raise RuntimeError("V5 selection produced no checkpoint.")
    selection_model.load_state_dict(best_state)
    selection_train = core._predict(
        selection_model,
        bundle.train_set,
        device=device,
    )
    selection_validation = core._predict(
        selection_model,
        bundle.val_set,
        device=device,
    )
    selection_metrics = core._metric_report(
        reference=selection_train,
        evaluation=selection_validation,
        plan=plan,
    )

    core.set_seed(int(model_seed))
    refit_model, refit_edge_fit = _build_candidate_model(
        config=config,
        bundle=refit_bundle,
        candidate=candidate,
        device=device,
    )
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    refit_scheduler = build_scheduler(
        refit_optimizer,
        total_epochs=int(best_epoch),
        warmup_epochs=min(
            int(training["warmup_epochs"]),
            max(1, int(best_epoch) // 3),
        ),
    )
    refit_loader = DataLoader(
        refit_bundle.train_set,
        batch_size=len(refit_bundle.train_set),
        shuffle=False,
    )
    refit_history: list[dict[str, Any]] = []
    for epoch in range(1, int(best_epoch) + 1):
        refit_model.train()
        batch = next(iter(refit_loader)).to(device)
        refit_optimizer.zero_grad(set_to_none=True)
        objective, cox, penalty = _objective(
            refit_model,
            batch,
            candidate=candidate,
            training=training,
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            refit_model.parameters(),
            max_norm=float(training["gradient_clip_norm"]),
        )
        refit_optimizer.step()
        refit_scheduler.step()
        refit_history.append(
            {
                "epoch": int(epoch),
                "train_objective": float(objective.item()),
                "train_cox_loss": float(cox.item()),
                "auxiliary_penalty": float(penalty.item()),
            }
        )
    train_predictions = core._predict(
        refit_model,
        refit_bundle.train_set,
        device=device,
    )
    test_predictions = core._predict(
        refit_model,
        refit_bundle.test_set,
        device=device,
    )
    test_metrics = core._metric_report(
        reference=train_predictions,
        evaluation=test_predictions,
        plan=plan,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in refit_model.state_dict().items()
            },
            "candidate": candidate,
            "selection_best_epoch": int(best_epoch),
            "analytic_edge_fit": refit_edge_fit,
        },
        checkpoint_path,
    )
    torch.save(
        {
            "state_dict": best_state,
            "candidate": candidate,
            "analytic_edge_fit": selection_edge_fit,
        },
        output_dir / "selection_model.pt",
    )
    core._save_refit_predictions(
        prediction_path,
        train=train_predictions,
        test=test_predictions,
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "candidate": candidate,
        "model_seed": int(model_seed),
        "outer_test_group": bundle.outer_test_group,
        "inner_validation_group": bundle.inner_validation_group,
        "best_epoch": int(best_epoch),
        "selection_validation_metrics": selection_metrics,
        "test_metrics": test_metrics,
        "selection_edge_fit": selection_edge_fit,
        "refit_edge_fit": refit_edge_fit,
        "parameter_count": int(
            sum(parameter.numel() for parameter in refit_model.parameters())
        ),
        "training_seconds": float(time.perf_counter() - started),
        "precomputed_edge_weight_used_at_inference": bool(
            candidate["model_family"] == "legacy"
        ),
        "selection_split_summary": bundle.split_summary,
        "refit_split_summary": refit_bundle.split_summary,
        "selection_history": selection_history,
        "refit_history": refit_history,
    }
    core._write_json(summary_path, summary)
    del selection_model, refit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _run_phase(
    *,
    phase: str,
    plan: dict[str, Any],
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    device_arg: str,
    resume: bool,
    candidates: Sequence[dict[str, Any]],
    model_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    config = core._load_config(
        ROOT / str(plan["base_config"]),
        data_dir,
    )
    site_table = core.build_site_feature_table(
        data_dir / core.ORAL_GUT_FILE
    )
    aggregates: list[dict[str, Any]] = []
    for candidate in candidates:
        folds: list[dict[str, Any]] = []
        for outer_group in plan["outer_test_groups"]:
            inner_group = (int(outer_group) + 1) % 5
            bundle = core.build_nested_bundle(
                config,
                data_dir=data_dir,
                outer_test_group=int(outer_group),
                inner_validation_group=inner_group,
                site_table=site_table,
            )
            refit_bundle = core.build_refit_bundle(
                config,
                data_dir=data_dir,
                outer_test_group=int(outer_group),
                site_table=site_table,
            )
            fold_root = (
                output_root
                / phase
                / candidate["name"]
                / f"outer_group{outer_group}_val{inner_group}"
            )
            run_dirs: list[Path] = []
            runs: list[dict[str, Any]] = []
            for model_seed in model_seeds:
                run_dir = fold_root / f"seed{int(model_seed)}"
                run = train_one(
                    config=config,
                    plan=plan,
                    plan_path=plan_path,
                    bundle=bundle,
                    refit_bundle=refit_bundle,
                    candidate=candidate,
                    model_seed=int(model_seed),
                    device_arg=device_arg,
                    output_dir=run_dir,
                    resume=resume,
                )
                run_dirs.append(run_dir)
                runs.append(run)
                print(
                    f"{phase} {candidate['name']} "
                    f"outer={outer_group} val={inner_group} "
                    f"seed={model_seed} "
                    f"valC={run['selection_validation_metrics']['harrell_c_index']:.6f} "
                    f"testC={run['test_metrics']['harrell_c_index']:.6f} "
                    f"iAUC={run['test_metrics']['normalized_integrated_auc']:.6f} "
                    f"epoch={run['best_epoch']}",
                    flush=True,
                )
            ensemble = core._ensemble_runs(
                run_dirs,
                plan=plan,
                output_path=fold_root / "ensemble_predictions.npz",
            )
            fold = {
                "outer_test_group": int(outer_group),
                "inner_validation_group": inner_group,
                "model_seeds": [int(value) for value in model_seeds],
                "member_best_epochs": [
                    int(run["best_epoch"]) for run in runs
                ],
                "mean_selection_validation_c_index": float(
                    statistics.mean(
                        run["selection_validation_metrics"][
                            "harrell_c_index"
                        ]
                        for run in runs
                    )
                ),
                "mean_refit_edge_r2": (
                    None
                    if runs[0]["refit_edge_fit"] is None
                    else float(
                        statistics.mean(
                            run["refit_edge_fit"]["r2"] for run in runs
                        )
                    )
                ),
                **ensemble,
            }
            core._write_json(
                fold_root / "ensemble_summary.json",
                fold,
            )
            folds.append(fold)
        aggregate = core._aggregate_candidate(candidate, folds)
        aggregates.append(aggregate)
        print(
            f"{phase} aggregate {candidate['name']} "
            f"C={aggregate['macro_mean_c_index']:.6f} "
            f"iAUC={aggregate['macro_mean_integrated_auc']:.6f} "
            f"iBrier={aggregate['macro_mean_integrated_brier']:.6f}",
            flush=True,
        )
    return aggregates


def run_development(
    *,
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    device_arg: str,
    resume: bool,
) -> dict[str, Any]:
    plan = core._read_json(plan_path)
    if plan["status"] != "locked_before_development_generation":
        raise RuntimeError("V5 development plan is not locked.")
    manifest_path = data_dir / core.DATA_FILE_NAMES["manifest_json"]
    manifest = core._read_json(manifest_path)
    if int(manifest["seed"]) != int(plan["development_generation_seed"]):
        raise RuntimeError("V5 development seed does not match the plan.")
    aggregates = _run_phase(
        phase="development",
        plan=plan,
        plan_path=plan_path,
        data_dir=data_dir,
        output_root=output_root,
        device_arg=device_arg,
        resume=resume,
        candidates=plan["candidates"],
        model_seeds=plan["development_model_seeds"],
    )
    baseline = next(
        row
        for row in aggregates
        if row["candidate"]["name"]
        == plan["development_gate"]["baseline_candidate"]
    )
    eligible = [
        row
        for row in aggregates
        if row["candidate"].get(
            "eligible_for_internal_relation_promotion",
            False,
        )
    ]
    performance_decisions = [
        core._gate_comparison(
            row,
            baseline,
            plan["development_gate"],
            audit=False,
        )
        for row in eligible
    ]
    noninferiority_decisions = [
        core._gate_comparison(
            row,
            baseline,
            plan["replacement_noninferiority_gate"],
            audit=False,
        )
        for row in eligible
    ]
    passed_names = {
        row["candidate"]
        for row in performance_decisions
        if row["passed"]
    }
    passed = [
        row
        for row in eligible
        if row["candidate"]["name"] in passed_names
    ]
    selected = (
        max(
            passed,
            key=lambda row: (
                row["macro_mean_c_index"],
                row["macro_mean_integrated_auc"],
                -row["macro_mean_integrated_brier"],
            ),
        )
        if passed
        else baseline
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "fresh_exact_edge_nested_logo_development",
        "plan_sha256": core._sha256(plan_path),
        "manifest_sha256": core._sha256(manifest_path),
        "aggregates": aggregates,
        "performance_selection_decisions": performance_decisions,
        "replacement_noninferiority_decisions": (
            noninferiority_decisions
        ),
        "selected_candidate": selected["candidate"],
        "candidate_passed_performance_gate": bool(passed),
        "audit_cohort_generated": False,
    }
    core._write_json(
        output_root / "development/development_summary.json",
        summary,
    )
    lock = {
        "schema_version": 1,
        "status": (
            "locked_after_development_before_audit_generation"
            if passed
            else "development_no_candidate_passed_performance_gate"
        ),
        "plan_sha256": core._sha256(plan_path),
        "development_manifest_sha256": core._sha256(manifest_path),
        "baseline_candidate": baseline["candidate"],
        "selected_candidate": selected["candidate"],
        "audit_generation_seed": int(plan["audit_generation_seed"]),
        "audit_cohort_generated": False,
        "performance_selection_decisions": performance_decisions,
        "replacement_noninferiority_decisions": (
            noninferiority_decisions
        ),
    }
    core._write_json(EXPERIMENT_DIR / "protocol_lock.json", lock)
    return summary


def run_audit(
    *,
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    device_arg: str,
    resume: bool,
) -> dict[str, Any]:
    plan = core._read_json(plan_path)
    lock_path = EXPERIMENT_DIR / "protocol_lock.json"
    lock = core._read_json(lock_path)
    if (
        lock["status"]
        != "locked_after_development_before_audit_generation"
    ):
        raise RuntimeError("No V5 candidate is eligible for audit.")
    manifest_path = data_dir / core.DATA_FILE_NAMES["manifest_json"]
    manifest = core._read_json(manifest_path)
    if int(manifest["seed"]) != int(lock["audit_generation_seed"]):
        raise RuntimeError("V5 audit seed does not match the lock.")
    candidate_by_name = {
        row["name"]: row for row in plan["candidates"]
    }
    baseline = candidate_by_name[
        plan["development_gate"]["baseline_candidate"]
    ]
    selected = candidate_by_name[lock["selected_candidate"]["name"]]
    aggregates = _run_phase(
        phase="audit",
        plan=plan,
        plan_path=plan_path,
        data_dir=data_dir,
        output_root=output_root,
        device_arg=device_arg,
        resume=resume,
        candidates=[baseline, selected],
        model_seeds=plan["audit_model_seeds"],
    )
    baseline_result = next(
        row
        for row in aggregates
        if row["candidate"]["name"] == baseline["name"]
    )
    selected_result = next(
        row
        for row in aggregates
        if row["candidate"]["name"] == selected["name"]
    )
    adoption = core._gate_comparison(
        selected_result,
        baseline_result,
        plan["audit_gate"],
        audit=True,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "one_time_exact_edge_nested_logo_audit",
        "plan_sha256": core._sha256(plan_path),
        "manifest_sha256": core._sha256(manifest_path),
        "aggregates": aggregates,
        "adoption_decision": adoption,
        "audit_seed_reruns_prohibited": True,
    }
    core._write_json(output_root / "audit/audit_summary.json", summary)
    lock.update(
        {
            "status": (
                "audit_passed_adoption_gate"
                if adoption["passed"]
                else "audit_rejected_candidate"
            ),
            "audit_cohort_generated": True,
            "audit_manifest_sha256": core._sha256(manifest_path),
            "audit_adoption_decision": adoption,
        }
    )
    core._write_json(lock_path, lock)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "audit"])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = {
        "plan_path": args.plan.resolve(),
        "data_dir": args.data_dir.resolve(),
        "output_root": args.output_root.resolve(),
        "device_arg": args.device,
        "resume": args.resume,
    }
    result = (
        run_development(**common)
        if args.phase == "development"
        else run_audit(**common)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "selected_candidate": result.get("selected_candidate"),
                "adoption_decision": result.get("adoption_decision"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
