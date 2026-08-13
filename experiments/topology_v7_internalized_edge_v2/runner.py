from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from experiments.topology_v7_compositional_temporal_v1.data_adapter import (
    sanitize_graph_item,
)
from experiments.topology_v7_compositional_temporal_v1.runner import (
    _aggregate,
    _fold_metrics,
    _load_config,
    _predict,
    _save_predictions,
    _select,
)
from experiments.topology_v7_internalized_edge_v2.model import (
    InternalizedEdgeDropInModel,
    internalized_edge_objective,
)
from experiments.topology_v7_nested_refit_v1.runner import (
    DATA_FILE_NAMES,
    HoldoutBundle,
    build_holdout_bundle,
)
from research.data import set_seed
from research.ensemble_v2 import build_model
from research.losses import cox_ph_loss
from research.train_v2 import build_scheduler, resolve_device


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/topology_v7_internalized_edge_v2"
DEFAULT_DEVELOPMENT_DATA = (
    DEFAULT_OUTPUT_ROOT / "cohorts/development_seed20261008"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def canonical_precomputed_edge_target(
    data: Any,
    *,
    num_node_types: int,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    edge_index = data.edge_index.long()
    edge_weight = data.edge_attr.float().view(-1)
    source_type = data.node_type[edge_index[0]].long()
    target_type = data.node_type[edge_index[1]].long()
    values = edge_weight.new_zeros(
        (num_node_types, num_node_types)
    )
    counts = torch.zeros(
        (num_node_types, num_node_types), dtype=torch.long
    )
    values.index_put_(
        (source_type, target_type), edge_weight, accumulate=True
    )
    counts.index_put_(
        (source_type.cpu(), target_type.cpu()),
        torch.ones_like(source_type.cpu()),
        accumulate=True,
    )
    expected = torch.ones_like(counts)
    expected.fill_diagonal_(0)
    if not torch.equal(counts, expected):
        raise ValueError("Precomputed edge target topology is incomplete.")
    upper = torch.triu_indices(
        num_node_types, num_node_types, offset=1
    )
    forward = values[upper[0], upper[1]]
    reverse = values[upper[1], upper[0]]
    if not torch.allclose(forward, reverse, atol=tolerance, rtol=0.0):
        raise ValueError("Precomputed training edge targets are asymmetric.")
    return 0.5 * (forward + reverse)


def _internalize_data(
    items: list[Any],
    *,
    num_node_types: int,
) -> list[Any]:
    result: list[Any] = []
    for item in items:
        edge_target = canonical_precomputed_edge_target(
            item, num_node_types=num_node_types
        )
        sanitized = sanitize_graph_item(
            item, num_node_types=num_node_types
        )
        sanitized.edge_supervision_target = edge_target
        result.append(sanitized)
    return result


def _candidate_data(
    bundle: HoldoutBundle,
    candidate: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    if candidate["model_family"] == "existing_v7_gnn":
        return bundle.train_set, bundle.eval_set
    return (
        _internalize_data(
            bundle.train_set, num_node_types=bundle.num_node_types
        ),
        _internalize_data(
            bundle.eval_set, num_node_types=bundle.num_node_types
        ),
    )


def _build_model(
    *,
    config: dict[str, Any],
    bundle: HoldoutBundle,
    candidate: dict[str, Any],
    device: torch.device,
    mean_log_time: float,
) -> torch.nn.Module:
    run_config = copy.deepcopy(config)
    base_model = build_model(run_config, bundle, device)
    if candidate["model_family"] == "existing_v7_gnn":
        return base_model
    model = InternalizedEdgeDropInModel(
        base_model,
        node_feature_dim=bundle.node_feature_dim,
        clinical_dim=bundle.clinical_dim,
        metabolite_dim=bundle.metabolite_dim,
        num_node_types=bundle.num_node_types,
        edge_mode=str(candidate["edge_mode"]),
        edge_hidden_dim=32,
        node_identity_dim=8,
    ).to(device)
    model.initialize_aft_location(mean_log_time)
    return model


def _edge_fit_metrics(
    model: torch.nn.Module,
    data_set: list[Any],
    *,
    device: torch.device,
) -> dict[str, float] | None:
    if not isinstance(model, InternalizedEdgeDropInModel):
        return None
    loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        prediction = model.edge_generator(batch).pair_weights
        target = batch.edge_supervision_target.view_as(prediction)
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    residual = target_np - prediction_np
    denominator = float(
        np.sum((target_np - target_np.mean(axis=0, keepdims=True)) ** 2)
    )
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def _pretrain_edges(
    model: torch.nn.Module,
    train_set: list[Any],
    *,
    candidate: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    epochs = int(candidate.get("edge_pretrain_epochs", 0))
    if epochs <= 0:
        return []
    if not isinstance(model, InternalizedEdgeDropInModel):
        raise ValueError("Only internal-edge models support edge pretraining.")
    optimizer = torch.optim.AdamW(
        model.edge_generator.parameters(),
        lr=float(training["edge_pretrain_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = DataLoader(
        train_set, batch_size=len(train_set), shuffle=False
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.edge_generator.train()
        batch = next(iter(loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        state = model.edge_generator(batch)
        target = batch.edge_supervision_target.view_as(
            state.pair_weights
        )
        mse = torch.mean((state.pair_weights - target) ** 2)
        objective = (
            mse
            + 1e-3 * state.delta_regularization
            + 1e-4 * state.saturation_regularization
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.edge_generator.parameters(), max_norm=2.0
        )
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch),
                "mse": float(mse.item()),
                "objective": float(objective.item()),
            }
        )
    return history


def train_development_fold(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    bundle: HoldoutBundle,
    candidate: dict[str, Any],
    device_arg: str,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "run_summary.json"
    checkpoint_path = output_dir / "model.pt"
    prediction_path = output_dir / "predictions.npz"
    training = plan["development_training"]
    run_fingerprint = _fingerprint(
        {
            "runner_sha256": _sha256(Path(__file__)),
            "model_sha256": _sha256(EXPERIMENT_DIR / "model.py"),
            "plan_sha256": _sha256(plan_path),
            "candidate": candidate,
            "holdout_group": bundle.holdout_group,
            "training": training,
            "train_ids": [str(item.sample_id) for item in bundle.train_set],
            "eval_ids": [str(item.sample_id) for item in bundle.eval_set],
            "standardizer": bundle.standardizer,
        }
    )
    if resume and summary_path.exists() and checkpoint_path.exists():
        summary = _read_json(summary_path)
        if summary.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Refusing mismatched resume artifacts: {output_dir}"
            )
        return summary

    set_seed(int(training["model_seed"]))
    device = resolve_device(device_arg)
    train_set, eval_set = _candidate_data(bundle, candidate)
    train_time_cpu = torch.tensor(
        [float(item.time.item()) for item in train_set],
        dtype=torch.float32,
    )
    mean_log_time = float(torch.log(train_time_cpu).mean().item())
    model = _build_model(
        config=config,
        bundle=bundle,
        candidate=candidate,
        device=device,
        mean_log_time=mean_log_time,
    )
    started = time.perf_counter()
    pretrain_history = _pretrain_edges(
        model,
        train_set,
        candidate=candidate,
        training=training,
        device=device,
    )
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
    train_loader = DataLoader(
        train_set, batch_size=len(train_set), shuffle=False
    )
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_c_index = float("-inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        model.train()
        batch = next(iter(train_loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, compute_contrastive=False)
        cox = cox_ph_loss(
            output["risk"],
            batch.time,
            batch.event,
            ties_method="breslow",
        )
        if candidate["model_family"] == "existing_v7_gnn":
            objective = cox
            edge_reconstruction = cox.new_zeros(())
            aft = cox.new_zeros(())
        else:
            losses = internalized_edge_objective(
                output,
                time=batch.time,
                event=batch.event,
                edge_target=batch.edge_supervision_target,
                edge_reconstruction_weight=float(
                    candidate["edge_reconstruction_weight"]
                ),
                aft_weight=float(candidate["aft_weight"]),
                cox_loss=cox,
            )
            objective = losses["total"]
            edge_reconstruction = losses["edge_reconstruction"]
            aft = losses["aft"]
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        evaluation = _predict(model, eval_set, device=device)
        history.append(
            {
                "epoch": epoch,
                "objective": float(objective.item()),
                "cox_loss": float(cox.item()),
                "edge_reconstruction_loss": float(
                    edge_reconstruction.item()
                ),
                "aft_loss": float(aft.item()),
                "evaluation_c_index": float(evaluation["c_index"]),
            }
        )
        if (
            evaluation["c_index"]
            > best_c_index + float(training["minimum_c_index_delta"])
        ):
            best_c_index = float(evaluation["c_index"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(training["early_stop_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "candidate": candidate,
            "node_type_names": bundle.node_type_names,
        },
        checkpoint_path,
    )
    train_predictions = _predict(model, train_set, device=device)
    eval_predictions = _predict(model, eval_set, device=device)
    metrics = _fold_metrics(train_predictions, eval_predictions, plan)
    _save_predictions(
        prediction_path,
        train=train_predictions,
        evaluation=eval_predictions,
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "candidate": candidate,
        "holdout_group": bundle.holdout_group,
        "model_seed": int(training["model_seed"]),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "evaluation_c_index": float(eval_predictions["c_index"]),
        "evaluation_cox_loss": float(eval_predictions["cox_loss"]),
        "metrics": metrics,
        "edge_pretrain_history": pretrain_history,
        "train_edge_fit": _edge_fit_metrics(
            model, train_set, device=device
        ),
        "evaluation_edge_fit": _edge_fit_metrics(
            model, eval_set, device=device
        ),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training_seconds": float(time.perf_counter() - started),
        "precomputed_edge_weight_used_at_inference": bool(
            candidate["model_family"] == "existing_v7_gnn"
        ),
        "history": history,
    }
    _write_json(summary_path, summary)
    return summary


def run_development(
    *,
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    device_arg: str,
    resume: bool,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if plan["status"] != "candidate_grid_locked_before_development_generation":
        raise RuntimeError("Candidate grid is not locked.")
    manifest_path = data_dir / DATA_FILE_NAMES["manifest_json"]
    manifest = _read_json(manifest_path)
    expected_seed = int(
        plan["future_cohorts"]["development_generation_seed"]
    )
    if int(manifest["seed"]) != expected_seed:
        raise RuntimeError("Development cohort seed does not match the plan.")
    config = _load_config(data_dir)
    development_root = output_root / "development"
    aggregates: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        runs: list[dict[str, Any]] = []
        for holdout_group in plan["groups"]:
            bundle = build_holdout_bundle(
                config, holdout_group=int(holdout_group)
            )
            run = train_development_fold(
                config=config,
                plan=plan,
                plan_path=plan_path,
                bundle=bundle,
                candidate=candidate,
                device_arg=device_arg,
                output_dir=(
                    development_root
                    / candidate["name"]
                    / f"holdout_group{holdout_group}"
                ),
                resume=resume,
            )
            runs.append(run)
            edge_text = ""
            if run["evaluation_edge_fit"] is not None:
                edge_text = (
                    f" edgeR2={run['evaluation_edge_fit']['r2']:.4f}"
                )
            print(
                f"{candidate['name']} group={holdout_group} "
                f"C={run['metrics']['harrell_c_index']:.6f} "
                f"iAUC={run['metrics']['normalized_integrated_auc']:.6f}"
                f"{edge_text} epoch={run['best_epoch']}",
                flush=True,
            )
        aggregates.append(_aggregate(candidate, runs))
    selected, decisions = _select(plan, aggregates)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "fresh_development_only",
        "plan_sha256": _sha256(plan_path),
        "manifest_sha256": _sha256(manifest_path),
        "aggregates": aggregates,
        "selection_decisions": decisions,
        "selected_candidate": selected,
        "audit_cohort_generated": False,
    }
    _write_json(development_root / "development_summary.json", summary)
    passed = (
        selected["candidate"]["name"]
        != plan["development_gate"]["baseline_candidate"]
    )
    lock = {
        "schema_version": 1,
        "status": (
            "locked_after_development_before_audit_generation"
            if passed
            else "development_no_candidate_passed_gate"
        ),
        "plan_sha256": _sha256(plan_path),
        "development_manifest_sha256": _sha256(manifest_path),
        "baseline_candidate": plan["development_gate"][
            "baseline_candidate"
        ],
        "selected_candidate": selected["candidate"],
        "fixed_epochs": int(selected["median_best_epoch"]),
        "audit_generation_seed": int(
            plan["future_cohorts"]["audit_generation_seed"]
        ),
        "audit_cohort_generated": False,
    }
    _write_json(EXPERIMENT_DIR / "protocol_lock.json", lock)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development"])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DEVELOPMENT_DATA
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_development(
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
                "selected_candidate": result["selected_candidate"][
                    "candidate"
                ]["name"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
