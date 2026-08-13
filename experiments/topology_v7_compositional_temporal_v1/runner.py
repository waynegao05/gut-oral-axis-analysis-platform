from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from experiments.topology_v7_compositional_temporal_v1.data_adapter import (
    sanitize_graph_sequence,
)
from experiments.topology_v7_compositional_temporal_v1.losses import (
    dual_survival_objective,
    fit_discrete_time_cutpoints,
)
from experiments.topology_v7_compositional_temporal_v1.metrics import (
    _evaluate_risk_source,
)
from experiments.topology_v7_compositional_temporal_v1.model import (
    InternalEdgeGATDualSurvivalModel,
)
from experiments.topology_v7_nested_refit_v1.runner import (
    DATA_FILE_NAMES,
    HoldoutBundle,
    build_holdout_bundle,
)
from research.data import set_seed
from research.ensemble_v2 import build_model
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import build_scheduler, resolve_device


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/topology_v7_compositional_temporal_v1"
)
DEFAULT_DEVELOPMENT_DATA = (
    DEFAULT_OUTPUT_ROOT / "cohorts/development_seed20261006"
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


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_config(data_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(
        (ROOT / "research_config_v7_v3_gnn_locked.yaml").read_text(
            encoding="utf-8"
        )
    )
    for key, file_name in DATA_FILE_NAMES.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        config["paths"][key] = path.as_posix()
    return config


def _candidate_data(
    bundle: HoldoutBundle,
    candidate: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    if candidate["model_family"] == "existing_v7_gnn":
        return bundle.train_set, bundle.eval_set
    return (
        sanitize_graph_sequence(
            bundle.train_set, num_node_types=bundle.num_node_types
        ),
        sanitize_graph_sequence(
            bundle.eval_set, num_node_types=bundle.num_node_types
        ),
    )


def _build_candidate_model(
    *,
    config: dict[str, Any],
    bundle: HoldoutBundle,
    candidate: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    if candidate["model_family"] == "existing_v7_gnn":
        run_config = copy.deepcopy(config)
        run_config["seed"] = int(training["model_seed"])
        return build_model(run_config, bundle, device)
    return InternalEdgeGATDualSurvivalModel(
        node_feature_dim=bundle.node_feature_dim,
        clinical_dim=bundle.clinical_dim,
        metabolite_dim=bundle.metabolite_dim,
        num_node_types=bundle.num_node_types,
        hidden_dim=int(training["hidden_dim"]),
        heads=int(training["heads"]),
        dropout=float(training["dropout"]),
        edge_hidden_dim=int(training["edge_hidden_dim"]),
        node_identity_dim=int(training["node_identity_dim"]),
        edge_mode=str(candidate["edge_mode"]),
        num_time_bins=int(training["num_time_bins"]),
    ).to(device)


def _predict(
    model: torch.nn.Module,
    data_set: Sequence[Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        output = model(batch, compute_contrastive=False)
    sample_ids = [str(value) for value in batch.sample_id]
    time_values = batch.time.detach().cpu().numpy().astype(float)
    event_values = batch.event.detach().cpu().numpy().astype(int)
    risk_values = output["risk"].detach().cpu().numpy().astype(float)
    return {
        "sample_ids": sample_ids,
        "time": time_values,
        "event": event_values,
        "risk": risk_values,
        "c_index": float(
            concordance_index(time_values, event_values, risk_values)
        ),
        "cox_loss": float(
            cox_ph_loss(
                output["risk"],
                batch.time,
                batch.event,
                ties_method="breslow",
            ).item()
        ),
    }


def _save_predictions(
    path: Path,
    *,
    train: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_sample_ids=np.asarray(train["sample_ids"], dtype=str),
        train_time=train["time"],
        train_event=train["event"],
        train_risk=train["risk"],
        eval_sample_ids=np.asarray(evaluation["sample_ids"], dtype=str),
        eval_time=evaluation["time"],
        eval_event=evaluation["event"],
        eval_risk=evaluation["risk"],
    )


def _fold_metrics(
    train: dict[str, Any],
    evaluation: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    return _evaluate_risk_source(
        train_time=train["time"],
        train_event=train["event"],
        train_risk=train["risk"],
        eval_time=evaluation["time"],
        eval_event=evaluation["event"],
        eval_risk=evaluation["risk"],
        report_horizons=plan["metrics"]["report_horizons"],
        integration_grid=plan["metrics"]["integration_grid"],
        uno_tau=96.0,
    )


def train_development_fold(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
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
    run_fingerprint = _json_hash(
        {
            "runner_sha256": _sha256(Path(__file__)),
            "plan_sha256": _sha256(DEFAULT_PLAN),
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
    model = _build_candidate_model(
        config=config,
        bundle=bundle,
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
    if len(train_loader) != 1:
        raise RuntimeError("Development training must use one exact Cox batch.")
    train_time = torch.tensor(
        [float(item.time.item()) for item in train_set],
        dtype=torch.float32,
    )
    train_event = torch.tensor(
        [float(item.event.item()) for item in train_set],
        dtype=torch.float32,
    )
    cutpoints = fit_discrete_time_cutpoints(
        train_time,
        train_event,
        num_bins=int(training["num_time_bins"]),
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_c_index = float("-inf")
    best_epoch = 0
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        model.train()
        batch = next(iter(train_loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, compute_contrastive=False)
        if candidate["model_family"] == "existing_v7_gnn":
            cox = cox_ph_loss(
                output["risk"],
                batch.time,
                batch.event,
                ties_method="breslow",
            )
            losses = {
                "total": cox,
                "cox": cox,
                "discrete": cox.new_zeros(()),
                "edge_delta": cox.new_zeros(()),
                "edge_saturation": cox.new_zeros(()),
            }
        else:
            losses = dual_survival_objective(
                output,
                time=batch.time,
                event=batch.event,
                cutpoints=cutpoints,
                discrete_weight=float(candidate["discrete_weight"]),
            )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        evaluation = _predict(model, eval_set, device=device)
        row = {
            "epoch": epoch,
            "objective": float(losses["total"].item()),
            "cox_loss": float(losses["cox"].item()),
            "discrete_loss": float(losses["discrete"].item()),
            "evaluation_c_index": float(evaluation["c_index"]),
        }
        history.append(row)
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
            "cutpoints": cutpoints,
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
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training_seconds": float(time.perf_counter() - started),
        "precomputed_edge_weight_used": bool(
            candidate["model_family"] == "existing_v7_gnn"
        ),
        "history": history,
    }
    _write_json(summary_path, summary)
    return summary


def _aggregate(
    candidate: dict[str, Any],
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    c_values = [float(run["metrics"]["harrell_c_index"]) for run in runs]
    iauc_values = [
        float(run["metrics"]["normalized_integrated_auc"]) for run in runs
    ]
    ibrier_values = [
        float(run["metrics"]["normalized_integrated_brier_score"])
        for run in runs
    ]
    return {
        "candidate": candidate,
        "num_folds": len(runs),
        "macro_mean_c_index": float(statistics.mean(c_values)),
        "macro_std_c_index": float(statistics.stdev(c_values)),
        "minimum_group_c_index": float(min(c_values)),
        "macro_mean_integrated_auc": float(statistics.mean(iauc_values)),
        "macro_mean_integrated_brier": float(
            statistics.mean(ibrier_values)
        ),
        "median_best_epoch": int(
            round(statistics.median(run["best_epoch"] for run in runs))
        ),
        "total_training_seconds": float(
            sum(float(run["training_seconds"]) for run in runs)
        ),
        "folds": [
            {
                "holdout_group": int(run["holdout_group"]),
                "c_index": float(run["metrics"]["harrell_c_index"]),
                "integrated_auc": float(
                    run["metrics"]["normalized_integrated_auc"]
                ),
                "integrated_brier": float(
                    run["metrics"]["normalized_integrated_brier_score"]
                ),
                "best_epoch": int(run["best_epoch"]),
            }
            for run in sorted(runs, key=lambda value: value["holdout_group"])
        ],
    }


def _select(
    plan: dict[str, Any],
    aggregates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_name = {
        str(row["candidate"]["name"]): row for row in aggregates
    }
    gate = plan["development_gate"]
    baseline = by_name[str(gate["baseline_candidate"])]
    baseline_folds = {
        row["holdout_group"]: row for row in baseline["folds"]
    }
    decisions: list[dict[str, Any]] = []
    for aggregate in aggregates:
        fold_deltas = [
            float(row["c_index"])
            - float(baseline_folds[row["holdout_group"]]["c_index"])
            for row in aggregate["folds"]
        ]
        checks = {
            "macro_c_index_gain": (
                aggregate["macro_mean_c_index"]
                - baseline["macro_mean_c_index"]
                >= float(gate["minimum_macro_c_index_gain"])
            ),
            "integrated_auc_gain": (
                aggregate["macro_mean_integrated_auc"]
                - baseline["macro_mean_integrated_auc"]
                >= float(gate["minimum_integrated_auc_gain"])
            ),
            "improved_groups": (
                sum(delta > 0 for delta in fold_deltas)
                >= int(gate["minimum_improved_groups"])
            ),
            "worst_group_regression": (
                min(fold_deltas)
                >= -float(gate["maximum_worst_group_regression"])
            ),
        }
        is_baseline = (
            aggregate["candidate"]["name"]
            == baseline["candidate"]["name"]
        )
        decisions.append(
            {
                "candidate_name": aggregate["candidate"]["name"],
                "macro_c_index_delta": float(
                    aggregate["macro_mean_c_index"]
                    - baseline["macro_mean_c_index"]
                ),
                "integrated_auc_delta": float(
                    aggregate["macro_mean_integrated_auc"]
                    - baseline["macro_mean_integrated_auc"]
                ),
                "fold_c_index_deltas": fold_deltas,
                "checks": checks,
                "eligible": bool(not is_baseline and all(checks.values())),
            }
        )
    eligible = [
        row for row in decisions if bool(row["eligible"])
    ]
    if eligible:
        selected_name = max(
            eligible,
            key=lambda row: (
                row["macro_c_index_delta"],
                row["integrated_auc_delta"],
            ),
        )["candidate_name"]
    else:
        selected_name = baseline["candidate"]["name"]
    return by_name[selected_name], decisions


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
        for holdout_group in range(5):
            bundle = build_holdout_bundle(
                config, holdout_group=holdout_group
            )
            run = train_development_fold(
                config=config,
                plan=plan,
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
            print(
                f"{candidate['name']} group={holdout_group} "
                f"C={run['metrics']['harrell_c_index']:.6f} "
                f"iAUC={run['metrics']['normalized_integrated_auc']:.6f} "
                f"epoch={run['best_epoch']}",
                flush=True,
            )
        aggregates.append(_aggregate(candidate, runs))
    selected, decisions = _select(plan, aggregates)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development_only",
        "plan_sha256": _sha256(plan_path),
        "manifest_sha256": _sha256(manifest_path),
        "aggregates": aggregates,
        "selection_decisions": decisions,
        "selected_candidate": selected,
        "audit_cohort_generated": False,
    }
    _write_json(development_root / "development_summary.json", summary)
    lock = {
        "schema_version": 1,
        "status": (
            "locked_after_development_before_audit_generation"
            if selected["candidate"]["name"]
            != plan["development_gate"]["baseline_candidate"]
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
    parser.add_argument(
        "phase",
        choices=["development"],
    )
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
    if args.phase == "development":
        result = run_development(
            plan_path=args.plan.resolve(),
            data_dir=args.data_dir.resolve(),
            output_root=args.output_root.resolve(),
            device_arg=args.device,
            resume=args.resume,
        )
    else:
        raise RuntimeError("Unsupported phase.")
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
