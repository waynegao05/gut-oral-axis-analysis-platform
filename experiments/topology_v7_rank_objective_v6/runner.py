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

from experiments.topology_v7_internal_relation_site_v3 import (
    runner as core,
)
from experiments.topology_v7_rank_objective_v6 import (
    prescreen,
)
from research.train_v2 import build_scheduler, resolve_device


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/topology_v7_rank_objective_v6"
)


def _fingerprint(
    *,
    plan_path: Path,
    candidate: dict[str, Any],
    outer_group: int,
    selected_epoch: int,
    model_seed: int,
    refit_bundle: core.RefitBundle,
) -> dict[str, Any]:
    return {
        "runner_sha256": core._sha256(Path(__file__)),
        "losses_sha256": core._sha256(
            EXPERIMENT_DIR / "losses.py"
        ),
        "edge_model_sha256": core._sha256(
            ROOT
            / "experiments/topology_v7_exact_edge_v5/model.py"
        ),
        "plan_sha256": core._sha256(plan_path),
        "candidate": candidate,
        "outer_test_group": int(outer_group),
        "selected_epoch": int(selected_epoch),
        "model_seed": int(model_seed),
        "refit_train_ids": core._sample_ids(
            refit_bundle.train_set
        ),
        "test_ids": core._sample_ids(
            refit_bundle.test_set
        ),
        "refit_standardizers": (
            refit_bundle.standardizers
        ),
    }


def _train_refit_member(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    refit_bundle: core.RefitBundle,
    candidate: dict[str, Any],
    selected_epoch: int,
    model_seed: int,
    device_arg: str,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "run_summary.json"
    fingerprint = _fingerprint(
        plan_path=plan_path,
        candidate=candidate,
        outer_group=refit_bundle.outer_test_group,
        selected_epoch=selected_epoch,
        model_seed=model_seed,
        refit_bundle=refit_bundle,
    )
    if resume and summary_path.exists():
        existing = core._read_json(summary_path)
        if existing["run_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Refusing mismatched V6 resume artifact: {output_dir}"
            )
        return existing

    training = plan["training"]
    device = resolve_device(device_arg)
    core.set_seed(int(model_seed))
    model = prescreen._build_model(
        config,
        refit_bundle,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=int(selected_epoch),
        warmup_epochs=min(
            int(training["warmup_epochs"]),
            max(1, int(selected_epoch) // 3),
        ),
    )
    loader = DataLoader(
        refit_bundle.train_set,
        batch_size=len(refit_bundle.train_set),
        shuffle=False,
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, int(selected_epoch) + 1):
        model.train()
        batch = next(iter(loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        objective, components = prescreen._objective(
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
            }
        )

    train_prediction = core._predict(
        model,
        refit_bundle.train_set,
        device=device,
    )
    test_prediction = core._predict(
        model,
        refit_bundle.test_set,
        device=device,
    )
    test_metrics = core._metric_report(
        reference=train_prediction,
        evaluation=test_prediction,
        plan=plan,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "candidate": candidate,
            "selected_epoch": int(selected_epoch),
        },
        output_dir / "refit_model.pt",
    )
    core._save_refit_predictions(
        output_dir / "predictions.npz",
        train=train_prediction,
        test=test_prediction,
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "candidate": candidate,
        "outer_test_group": (
            refit_bundle.outer_test_group
        ),
        "selected_epoch": int(selected_epoch),
        "model_seed": int(model_seed),
        "test_metrics": test_metrics,
        "training_seconds": float(
            time.perf_counter() - started
        ),
        "parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
            )
        ),
        "precomputed_edge_weight_used_at_inference": False,
        "refit_split_summary": refit_bundle.split_summary,
        "history": history,
    }
    core._write_json(summary_path, summary)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _inner_groups(outer_group: int) -> list[int]:
    return [
        group
        for group in range(5)
        if group != int(outer_group)
    ]


def _run_candidate_outer(
    *,
    phase: str,
    config: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    site_table: Any,
    candidate: dict[str, Any],
    outer_group: int,
    model_seeds: Sequence[int],
    device_arg: str,
    resume: bool,
) -> dict[str, Any]:
    fold_root = (
        output_root
        / phase
        / candidate["name"]
        / f"outer_group{outer_group}_full_inner"
    )
    fold_summary_path = fold_root / "ensemble_summary.json"
    if resume and fold_summary_path.exists():
        return core._read_json(fold_summary_path)

    inner_runs: list[dict[str, Any]] = []
    for inner_group in _inner_groups(outer_group):
        bundle = core.build_nested_bundle(
            config,
            data_dir=data_dir,
            outer_test_group=int(outer_group),
            inner_validation_group=int(inner_group),
            site_table=site_table,
        )
        run = prescreen._train_inner(
            config=config,
            plan=plan,
            bundle=bundle,
            candidate=candidate,
            device_arg=device_arg,
        )
        inner_runs.append(run)
        print(
            f"{phase} {candidate['name']} "
            f"outer={outer_group} inner={inner_group} "
            f"valC={run['metrics']['harrell_c_index']:.6f} "
            f"iAUC={run['metrics']['normalized_integrated_auc']:.6f} "
            f"epoch={run['best_epoch']}",
            flush=True,
        )
    selected_epoch = int(
        round(
            statistics.median(
                run["best_epoch"] for run in inner_runs
            )
        )
    )
    refit_bundle = core.build_refit_bundle(
        config,
        data_dir=data_dir,
        outer_test_group=int(outer_group),
        site_table=site_table,
    )
    run_dirs: list[Path] = []
    refit_runs: list[dict[str, Any]] = []
    for model_seed in model_seeds:
        run_dir = fold_root / f"seed{int(model_seed)}"
        run = _train_refit_member(
            config=config,
            plan=plan,
            plan_path=plan_path,
            refit_bundle=refit_bundle,
            candidate=candidate,
            selected_epoch=selected_epoch,
            model_seed=int(model_seed),
            device_arg=device_arg,
            output_dir=run_dir,
            resume=resume,
        )
        run_dirs.append(run_dir)
        refit_runs.append(run)
    ensemble = core._ensemble_runs(
        run_dirs,
        plan=plan,
        output_path=fold_root / "ensemble_predictions.npz",
    )
    fold = {
        "outer_test_group": int(outer_group),
        "inner_validation_groups": _inner_groups(
            outer_group
        ),
        "inner_best_epochs": [
            int(run["best_epoch"]) for run in inner_runs
        ],
        "selected_refit_epoch": int(selected_epoch),
        "inner_mean_validation_c_index": float(
            statistics.mean(
                run["metrics"]["harrell_c_index"]
                for run in inner_runs
            )
        ),
        "inner_mean_validation_integrated_auc": float(
            statistics.mean(
                run["metrics"][
                    "normalized_integrated_auc"
                ]
                for run in inner_runs
            )
        ),
        "model_seeds": [
            int(value) for value in model_seeds
        ],
        "num_members": len(refit_runs),
        "member_directories": [
            str(path) for path in run_dirs
        ],
        "predictions_path": str(
            fold_root / "ensemble_predictions.npz"
        ),
        "test_metrics": ensemble["test_metrics"],
        "inner_runs": inner_runs,
    }
    core._write_json(fold_summary_path, fold)
    print(
        f"{phase} {candidate['name']} "
        f"outer={outer_group} testC="
        f"{fold['test_metrics']['harrell_c_index']:.6f} "
        f"iAUC="
        f"{fold['test_metrics']['normalized_integrated_auc']:.6f} "
        f"refit_epoch={selected_epoch}",
        flush=True,
    )
    return fold


def _run_phase(
    *,
    phase: str,
    plan: dict[str, Any],
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    candidates: Sequence[dict[str, Any]],
    model_seeds: Sequence[int],
    device_arg: str,
    resume: bool,
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
        folds = [
            _run_candidate_outer(
                phase=phase,
                config=config,
                plan=plan,
                plan_path=plan_path,
                data_dir=data_dir,
                output_root=output_root,
                site_table=site_table,
                candidate=candidate,
                outer_group=int(outer_group),
                model_seeds=model_seeds,
                device_arg=device_arg,
                resume=resume,
            )
            for outer_group in plan["outer_test_groups"]
        ]
        aggregate = core._aggregate_candidate(
            candidate,
            folds,
        )
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
        raise RuntimeError("V6 development plan is not locked.")
    manifest_path = (
        data_dir / core.DATA_FILE_NAMES["manifest_json"]
    )
    manifest = core._read_json(manifest_path)
    if int(manifest["seed"]) != int(
        plan["development_generation_seed"]
    ):
        raise RuntimeError("V6 development seed mismatch.")
    aggregates = _run_phase(
        phase="development",
        plan=plan,
        plan_path=plan_path,
        data_dir=data_dir,
        output_root=output_root,
        candidates=plan["candidates"],
        model_seeds=plan[
            "development_refit_model_seeds"
        ],
        device_arg=device_arg,
        resume=resume,
    )
    baseline = next(
        row
        for row in aggregates
        if row["candidate"]["name"]
        == plan["development_gate"]["baseline_candidate"]
    )
    candidate = next(
        row
        for row in aggregates
        if row["candidate"].get(
            "eligible_for_promotion",
            False,
        )
    )
    decision = core._gate_comparison(
        candidate,
        baseline,
        plan["development_gate"],
        audit=False,
    )
    selected = (
        candidate if decision["passed"] else baseline
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "fresh_full_inner_logo_rank_objective_development"
        ),
        "plan_sha256": core._sha256(plan_path),
        "manifest_sha256": core._sha256(manifest_path),
        "aggregates": aggregates,
        "development_decision": decision,
        "selected_candidate": selected["candidate"],
        "candidate_passed_development_gate": bool(
            decision["passed"]
        ),
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
            if decision["passed"]
            else "development_candidate_rejected"
        ),
        "plan_sha256": core._sha256(plan_path),
        "development_manifest_sha256": core._sha256(
            manifest_path
        ),
        "baseline_candidate": baseline["candidate"],
        "selected_candidate": selected["candidate"],
        "audit_generation_seed": int(
            plan["audit_generation_seed"]
        ),
        "audit_cohort_generated": False,
        "development_decision": decision,
    }
    core._write_json(
        EXPERIMENT_DIR / "protocol_lock.json",
        lock,
    )
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
        raise RuntimeError("No V6 candidate is eligible for audit.")
    manifest_path = (
        data_dir / core.DATA_FILE_NAMES["manifest_json"]
    )
    manifest = core._read_json(manifest_path)
    if int(manifest["seed"]) != int(
        lock["audit_generation_seed"]
    ):
        raise RuntimeError("V6 audit seed mismatch.")
    by_name = {
        row["name"]: row for row in plan["candidates"]
    }
    baseline = by_name[
        plan["development_gate"]["baseline_candidate"]
    ]
    selected = by_name[
        lock["selected_candidate"]["name"]
    ]
    aggregates = _run_phase(
        phase="audit",
        plan=plan,
        plan_path=plan_path,
        data_dir=data_dir,
        output_root=output_root,
        candidates=[baseline, selected],
        model_seeds=plan["audit_refit_model_seeds"],
        device_arg=device_arg,
        resume=resume,
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
        "scope": "one_time_full_inner_logo_rank_objective_audit",
        "aggregates": aggregates,
        "adoption_decision": adoption,
    }
    core._write_json(
        output_root / "audit/audit_summary.json",
        summary,
    )
    lock.update(
        {
            "status": (
                "audit_passed_adoption_gate"
                if adoption["passed"]
                else "audit_rejected_candidate"
            ),
            "audit_cohort_generated": True,
            "audit_manifest_sha256": core._sha256(
                manifest_path
            ),
            "audit_adoption_decision": adoption,
        }
    )
    core._write_json(lock_path, lock)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["development", "audit"],
    )
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
    function = (
        run_development
        if args.phase == "development"
        else run_audit
    )
    result = function(
        plan_path=args.plan.resolve(),
        data_dir=args.data_dir.resolve(),
        output_root=args.output_root.resolve(),
        device_arg=args.device,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "selected_candidate": result.get(
                    "selected_candidate"
                ),
                "adoption_decision": result.get(
                    "adoption_decision"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
