from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from experiments.topology_v7_compositional_temporal_v1.metrics import (
    _evaluate_risk_source,
)
from experiments.topology_v7_internal_relation_site_v3.features import (
    SiteFeatureTable,
    attach_site_features,
    build_site_feature_table,
    fit_site_standardizer,
)
from experiments.topology_v7_internal_relation_site_v3.model import (
    InternalRelationSiteModel,
    internal_relation_regularization,
)
from experiments.topology_v7_internalized_edge_v2.model import (
    lognormal_aft_nll,
)
from experiments.topology_v7_nested_refit_v1.runner import (
    DATA_FILE_NAMES,
    build_holdout_bundle,
    _load_config,
    _standardize_tensor_attribute,
)
from research.data import build_dataset_from_csv, set_seed
from research.ensemble_v2 import build_model
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import build_scheduler, resolve_device


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/topology_v7_internal_relation_site_v3"
)
ORAL_GUT_FILE = "topology_v7_sample_oral_gut_table.csv"


@dataclass
class NestedBundle:
    train_set: list[Any]
    val_set: list[Any]
    test_set: list[Any]
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int
    site_feature_dim: int
    num_node_types: int
    node_type_names: list[str]
    outer_test_group: int
    inner_validation_group: int
    standardizers: dict[str, Any]
    split_summary: dict[str, Any]


@dataclass
class RefitBundle:
    train_set: list[Any]
    test_set: list[Any]
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int
    site_feature_dim: int
    num_node_types: int
    node_type_names: list[str]
    outer_test_group: int
    standardizers: dict[str, Any]
    split_summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_as_builtin(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _as_builtin(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sample_ids(items: Sequence[Any]) -> list[str]:
    return [str(item.sample_id) for item in items]


def build_nested_bundle(
    config: dict[str, Any],
    *,
    data_dir: Path,
    outer_test_group: int,
    inner_validation_group: int,
    site_table: SiteFeatureTable | None = None,
) -> NestedBundle:
    if outer_test_group == inner_validation_group:
        raise ValueError("Outer test and inner validation groups must differ.")
    graph_preprocess = config.get("graph_preprocess", {})
    raw = build_dataset_from_csv(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
        node_feature_columns=config["model"]["node_feature_columns"],
        clinical_columns=config["model"]["clinical_columns"],
        metabolite_columns=config["model"]["metabolite_columns"],
        seed=int(config["seed"]),
        split_seed=int(config["train"].get("split_seed", 42)),
        keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
        min_edge_weight=graph_preprocess.get("min_edge_weight"),
        standardize_tabular=False,
        val_ratio=float(config["train"]["val_ratio"]),
        test_ratio=float(config["train"]["test_ratio"]),
        validation_group=inner_validation_group,
        test_group=outer_test_group,
    )
    train_set = sorted(raw.train_set, key=lambda item: str(item.sample_id))
    val_set = sorted(raw.val_set, key=lambda item: str(item.sample_id))
    test_set = sorted(raw.test_set, key=lambda item: str(item.sample_id))
    if (len(train_set), len(val_set), len(test_set)) != (2160, 720, 720):
        raise RuntimeError(
            "Nested V3 requires a 2160/720/720 three-group/one-group/"
            "one-group split."
        )
    expected_train_groups = sorted(
        set(range(5)).difference(
            {int(outer_test_group), int(inner_validation_group)}
        )
    )
    actual_train_groups = sorted(
        int(value) for value in raw.split_summary["train_groups"]
    )
    actual_val_groups = sorted(
        int(value) for value in raw.split_summary["val_groups"]
    )
    actual_test_groups = sorted(
        int(value) for value in raw.split_summary["test_groups"]
    )
    if actual_train_groups != expected_train_groups:
        raise RuntimeError("Unexpected outer-training groups.")
    if actual_val_groups != [int(inner_validation_group)]:
        raise RuntimeError("Unexpected inner-validation group.")
    if actual_test_groups != [int(outer_test_group)]:
        raise RuntimeError("Unexpected outer-test group.")

    standardizers = {
        "fit_scope": "three_outer_training_groups_only",
        "clinical": _standardize_tensor_attribute(
            train_set,
            [val_set, test_set],
            "clinical",
        ),
        "metabolites": _standardize_tensor_attribute(
            train_set,
            [val_set, test_set],
            "metabolites",
        ),
    }
    if site_table is None:
        site_table = build_site_feature_table(data_dir / ORAL_GUT_FILE)
    site_standardizer = fit_site_standardizer(
        site_table,
        _sample_ids(train_set),
    )
    attach_site_features(
        [train_set, val_set, test_set],
        site_table,
        site_standardizer,
    )
    standardizers["site"] = site_standardizer

    train_ids = set(_sample_ids(train_set))
    val_ids = set(_sample_ids(val_set))
    test_ids = set(_sample_ids(test_set))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Nested split sample IDs overlap.")
    return NestedBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        node_feature_dim=raw.node_feature_dim,
        clinical_dim=raw.clinical_dim,
        metabolite_dim=raw.metabolite_dim,
        site_feature_dim=len(site_table.feature_columns),
        num_node_types=raw.num_node_types,
        node_type_names=list(raw.node_type_names),
        outer_test_group=int(outer_test_group),
        inner_validation_group=int(inner_validation_group),
        standardizers=standardizers,
        split_summary={
            **raw.split_summary,
            "outer_test_group": int(outer_test_group),
            "inner_validation_group": int(inner_validation_group),
            "generation_group_used_as_model_feature": False,
            "site_standardizer_fit_scope": (
                "three_outer_training_groups_only"
            ),
        },
    )


def build_refit_bundle(
    config: dict[str, Any],
    *,
    data_dir: Path,
    outer_test_group: int,
    site_table: SiteFeatureTable | None = None,
) -> RefitBundle:
    raw = build_holdout_bundle(
        config,
        holdout_group=int(outer_test_group),
    )
    if site_table is None:
        site_table = build_site_feature_table(data_dir / ORAL_GUT_FILE)
    site_standardizer = fit_site_standardizer(
        site_table,
        _sample_ids(raw.train_set),
    )
    attach_site_features(
        [raw.train_set, raw.eval_set],
        site_table,
        site_standardizer,
    )
    return RefitBundle(
        train_set=raw.train_set,
        test_set=raw.eval_set,
        node_feature_dim=raw.node_feature_dim,
        clinical_dim=raw.clinical_dim,
        metabolite_dim=raw.metabolite_dim,
        site_feature_dim=len(site_table.feature_columns),
        num_node_types=raw.num_node_types,
        node_type_names=list(raw.node_type_names),
        outer_test_group=int(outer_test_group),
        standardizers={
            "fit_scope": "all_four_non_test_groups",
            "clinical": raw.standardizer["clinical"],
            "metabolites": raw.standardizer["metabolites"],
            "site": site_standardizer,
        },
        split_summary={
            **raw.split_summary,
            "outer_test_group": int(outer_test_group),
            "generation_group_used_as_model_feature": False,
            "site_standardizer_fit_scope": "all_four_non_test_groups",
        },
    )


def _build_candidate_model(
    *,
    config: dict[str, Any],
    bundle: NestedBundle | RefitBundle,
    candidate: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    run_config = copy.deepcopy(config)
    base_model = build_model(run_config, bundle, device)
    if candidate["model_family"] == "legacy":
        return base_model
    if candidate["model_family"] != "internal_relation":
        raise ValueError(
            f"Unsupported model family: {candidate['model_family']}"
        )
    model = InternalRelationSiteModel(
        base_model,
        node_feature_dim=bundle.node_feature_dim,
        clinical_dim=bundle.clinical_dim,
        metabolite_dim=bundle.metabolite_dim,
        site_feature_dim=bundle.site_feature_dim,
        num_node_types=bundle.num_node_types,
        edge_mode=str(candidate["edge_mode"]),
        use_site_residual=bool(candidate["use_site_residual"]),
    ).to(device)
    mean_log_time = float(
        np.mean(
            [
                np.log(float(item.time.item()))
                for item in bundle.train_set
            ]
        )
    )
    model.initialize_aft_location(mean_log_time)
    return model


def _training_objective(
    output: dict[str, torch.Tensor],
    batch: Any,
    *,
    candidate: dict[str, Any],
    training: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cox = cox_ph_loss(
        output["risk"],
        batch.time,
        batch.event,
        ties_method="breslow",
    )
    if candidate["model_family"] == "internal_relation":
        regularization = internal_relation_regularization(
            output,
            edge_weight=float(training["edge_regularization_weight"]),
            site_residual_weight=float(
                training["site_residual_regularization_weight"]
            ),
        )
        aft_weight = float(candidate.get("aft_weight", 0.0))
        if aft_weight > 0.0:
            aft = lognormal_aft_nll(
                output["aft_location"],
                output["aft_log_scale"],
                batch.time,
                batch.event,
            )
        else:
            aft = cox.new_zeros(())
    else:
        regularization = cox.new_zeros(())
        aft = cox.new_zeros(())
        aft_weight = 0.0
    objective = cox + regularization + aft_weight * aft
    return objective, cox, regularization, aft


def _predict(
    model: torch.nn.Module,
    data_set: Sequence[Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        data_set,
        batch_size=len(data_set),
        shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        output = model(batch, compute_contrastive=False)
        risk = output["risk"].detach().cpu().numpy().astype(float)
        time_values = batch.time.detach().cpu().numpy().astype(float)
        event_values = batch.event.detach().cpu().numpy().astype(float)
        result = {
            "sample_ids": np.asarray(
                [str(value) for value in batch.sample_id],
                dtype=str,
            ),
            "time": time_values,
            "event": event_values,
            "risk": risk,
            "c_index": float(
                concordance_index(time_values, event_values, risk)
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
        if "pair_edge_weights" in output:
            edge_values = (
                output["pair_edge_weights"].detach().cpu().numpy()
            )
            result["edge_summary"] = {
                "mean": float(edge_values.mean()),
                "std": float(edge_values.std()),
                "minimum": float(edge_values.min()),
                "maximum": float(edge_values.max()),
            }
        else:
            result["edge_summary"] = None
        return result


def _metric_report(
    *,
    reference: dict[str, Any],
    evaluation: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    metrics = plan["metrics"]
    return _evaluate_risk_source(
        train_time=reference["time"],
        train_event=reference["event"].astype(int),
        train_risk=reference["risk"],
        eval_time=evaluation["time"],
        eval_event=evaluation["event"].astype(int),
        eval_risk=evaluation["risk"],
        report_horizons=[
            float(value) for value in metrics["report_horizons"]
        ],
        integration_grid=[
            float(value) for value in metrics["integration_grid"]
        ],
        uno_tau=float(metrics["uno_tau"]),
    )


def _save_refit_predictions(
    path: Path,
    *,
    train: dict[str, Any],
    test: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for split_name, values in (
        ("train", train),
        ("test", test),
    ):
        for key in ("sample_ids", "time", "event", "risk"):
            payload[f"{split_name}_{key}"] = values[key]
    np.savez_compressed(path, **payload)


def train_one(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    bundle: NestedBundle,
    refit_bundle: RefitBundle,
    candidate: dict[str, Any],
    model_seed: int,
    device_arg: str,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "run_summary.json"
    checkpoint_path = output_dir / "model.pt"
    predictions_path = output_dir / "predictions.npz"
    training = plan["training"]
    run_fingerprint = _fingerprint(
        {
            "runner_sha256": _sha256(Path(__file__)),
            "model_sha256": _sha256(EXPERIMENT_DIR / "model.py"),
            "features_sha256": _sha256(EXPERIMENT_DIR / "features.py"),
            "plan_sha256": _sha256(plan_path),
            "base_config_sha256": _sha256(
                ROOT / str(plan["base_config"])
            ),
            "candidate": candidate,
            "model_seed": int(model_seed),
            "outer_test_group": bundle.outer_test_group,
            "inner_validation_group": bundle.inner_validation_group,
            "training": training,
            "train_sample_ids": _sample_ids(bundle.train_set),
            "validation_sample_ids": _sample_ids(bundle.val_set),
            "test_sample_ids": _sample_ids(bundle.test_set),
            "standardizers": bundle.standardizers,
            "refit_train_sample_ids": _sample_ids(
                refit_bundle.train_set
            ),
            "refit_test_sample_ids": _sample_ids(
                refit_bundle.test_set
            ),
            "refit_standardizers": refit_bundle.standardizers,
        }
    )
    if (
        resume
        and summary_path.exists()
        and checkpoint_path.exists()
        and predictions_path.exists()
    ):
        summary = _read_json(summary_path)
        if summary.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Refusing mismatched resume artifacts: {output_dir}"
            )
        return summary

    set_seed(int(model_seed))
    device = resolve_device(device_arg)
    selection_model = _build_candidate_model(
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
    train_loader = DataLoader(
        bundle.train_set,
        batch_size=len(bundle.train_set),
        shuffle=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_c_index = float("-inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        selection_model.train()
        batch = next(iter(train_loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = selection_model(batch, compute_contrastive=False)
        objective, cox, regularization, aft = _training_objective(
            output,
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

        validation = _predict(
            selection_model,
            bundle.val_set,
            device=device,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_objective": float(objective.item()),
                "train_cox_loss": float(cox.item()),
                "regularization": float(regularization.item()),
                "aft_loss": float(aft.item()),
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
        raise RuntimeError("Training did not produce a checkpoint.")
    selection_model.load_state_dict(best_state)
    selection_train_predictions = _predict(
        selection_model,
        bundle.train_set,
        device=device,
    )
    selection_validation_predictions = _predict(
        selection_model,
        bundle.val_set,
        device=device,
    )
    selection_validation_metrics = _metric_report(
        reference=selection_train_predictions,
        evaluation=selection_validation_predictions,
        plan=plan,
    )

    set_seed(int(model_seed))
    refit_model = _build_candidate_model(
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
        output = refit_model(batch, compute_contrastive=False)
        objective, cox, regularization, aft = _training_objective(
            output,
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
                "regularization": float(regularization.item()),
                "aft_loss": float(aft.item()),
            }
        )

    train_predictions = _predict(
        refit_model,
        refit_bundle.train_set,
        device=device,
    )
    test_predictions = _predict(
        refit_model,
        refit_bundle.test_set,
        device=device,
    )
    test_metrics = _metric_report(
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
            "node_type_names": refit_bundle.node_type_names,
            "site_feature_columns": refit_bundle.standardizers["site"][
                "feature_columns"
            ],
            "selection_best_epoch": int(best_epoch),
            "refit_scope": "all_four_non_test_groups",
        },
        checkpoint_path,
    )
    torch.save(
        {
            "state_dict": best_state,
            "candidate": candidate,
            "selection_scope": (
                "three_training_groups_one_validation_group"
            ),
        },
        output_dir / "selection_model.pt",
    )
    _save_refit_predictions(
        predictions_path,
        train=train_predictions,
        test=test_predictions,
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "candidate": candidate,
        "model_seed": int(model_seed),
        "outer_test_group": bundle.outer_test_group,
        "inner_validation_group": bundle.inner_validation_group,
        "best_epoch": best_epoch,
        "refit_epochs": best_epoch,
        "epochs_run": len(history),
        "selection_validation_metrics": selection_validation_metrics,
        "test_metrics": test_metrics,
        "selection_validation_edge_summary": (
            selection_validation_predictions["edge_summary"]
        ),
        "test_edge_summary": test_predictions["edge_summary"],
        "parameter_count": int(
            sum(
                parameter.numel()
                for parameter in refit_model.parameters()
            )
        ),
        "training_seconds": float(time.perf_counter() - started),
        "precomputed_edge_weight_used_at_inference": bool(
            candidate["model_family"] == "legacy"
        ),
        "split_summary": bundle.split_summary,
        "refit_split_summary": refit_bundle.split_summary,
        "selection_standardizers": bundle.standardizers,
        "refit_standardizers": refit_bundle.standardizers,
        "selection_history": history,
        "refit_history": refit_history,
    }
    _write_json(summary_path, summary)
    del selection_model, refit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _load_prediction_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _ensemble_runs(
    run_dirs: Sequence[Path],
    *,
    plan: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("At least one run is required for ensembling.")
    predictions = [
        _load_prediction_file(path / "predictions.npz")
        for path in run_dirs
    ]
    reference = predictions[0]
    for values in predictions[1:]:
        for split_name in ("train", "test"):
            for key in ("sample_ids", "time", "event"):
                name = f"{split_name}_{key}"
                if not np.array_equal(values[name], reference[name]):
                    raise RuntimeError(
                        "Seed predictions are not aligned for ensembling."
                    )

    standardized: dict[str, list[np.ndarray]] = {
        "train": [],
        "test": [],
    }
    for values in predictions:
        train_risk = values["train_risk"].astype(float)
        mean = float(train_risk.mean())
        scale = float(train_risk.std())
        if scale <= 1e-8:
            raise RuntimeError("Training risk has zero variance.")
        for split_name in standardized:
            standardized[split_name].append(
                (values[f"{split_name}_risk"].astype(float) - mean) / scale
            )
    ensemble_risk = {
        split_name: np.mean(np.vstack(rows), axis=0)
        for split_name, rows in standardized.items()
    }
    train = {
        "sample_ids": reference["train_sample_ids"],
        "time": reference["train_time"],
        "event": reference["train_event"],
        "risk": ensemble_risk["train"],
    }
    test = {
        "sample_ids": reference["test_sample_ids"],
        "time": reference["test_time"],
        "event": reference["test_event"],
        "risk": ensemble_risk["test"],
    }
    test_metrics = _metric_report(
        reference=train,
        evaluation=test,
        plan=plan,
    )
    _save_refit_predictions(
        output_path,
        train=train,
        test=test,
    )
    return {
        "num_members": len(run_dirs),
        "member_directories": [
            path.relative_to(ROOT).as_posix() for path in run_dirs
        ],
        "predictions_path": output_path.relative_to(ROOT).as_posix(),
        "test_metrics": test_metrics,
    }


def _aggregate_candidate(
    candidate: dict[str, Any],
    folds: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    test_rows = [row["test_metrics"] for row in folds]
    return {
        "candidate": candidate,
        "num_outer_folds": len(folds),
        "macro_mean_c_index": float(
            statistics.mean(row["harrell_c_index"] for row in test_rows)
        ),
        "macro_std_c_index": float(
            statistics.stdev(
                row["harrell_c_index"] for row in test_rows
            )
        ),
        "minimum_group_c_index": float(
            min(row["harrell_c_index"] for row in test_rows)
        ),
        "macro_mean_integrated_auc": float(
            statistics.mean(
                row["normalized_integrated_auc"] for row in test_rows
            )
        ),
        "macro_mean_integrated_brier": float(
            statistics.mean(
                row["normalized_integrated_brier_score"]
                for row in test_rows
            )
        ),
        "folds": list(folds),
    }


def _gate_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    gate: dict[str, Any],
    *,
    audit: bool,
) -> dict[str, Any]:
    baseline_by_group = {
        int(row["outer_test_group"]): row
        for row in baseline["folds"]
    }
    deltas = [
        float(row["test_metrics"]["harrell_c_index"])
        - float(
            baseline_by_group[int(row["outer_test_group"])][
                "test_metrics"
            ]["harrell_c_index"]
        )
        for row in candidate["folds"]
    ]
    c_gain = (
        float(candidate["macro_mean_c_index"])
        - float(baseline["macro_mean_c_index"])
    )
    auc_gain = (
        float(candidate["macro_mean_integrated_auc"])
        - float(baseline["macro_mean_integrated_auc"])
    )
    brier_change = (
        float(candidate["macro_mean_integrated_brier"])
        - float(baseline["macro_mean_integrated_brier"])
    )
    num_horizons = len(
        candidate["folds"][0]["test_metrics"]["auc_by_horizon"]
    )
    horizon_auc_deltas = []
    for horizon_index in range(num_horizons):
        candidate_auc = statistics.mean(
            float(
                row["test_metrics"]["auc_by_horizon"][horizon_index][
                    "auc"
                ]
            )
            for row in candidate["folds"]
        )
        baseline_auc = statistics.mean(
            float(
                row["test_metrics"]["auc_by_horizon"][horizon_index][
                    "auc"
                ]
            )
            for row in baseline["folds"]
        )
        horizon_auc_deltas.append(float(candidate_auc - baseline_auc))
    checks = {
        "macro_c_index_gain": bool(
            c_gain >= float(gate["minimum_macro_c_index_gain"])
        ),
        "integrated_auc_gain": bool(
            auc_gain >= float(gate["minimum_integrated_auc_gain"])
        ),
        "improved_outer_groups": bool(
            sum(delta > 0.0 for delta in deltas)
            >= int(gate["minimum_improved_outer_groups"])
        ),
        "worst_group_regression": bool(
            min(deltas)
            >= -float(gate["maximum_worst_group_regression"])
        ),
        "any_horizon_auc_regression": bool(
            min(horizon_auc_deltas)
            >= -float(gate["maximum_any_horizon_auc_regression"])
        ),
    }
    if audit:
        checks["integrated_brier"] = bool(
            not gate["integrated_brier_must_not_worsen"]
            or brier_change <= 0.0
        )
    else:
        checks["integrated_brier"] = bool(
            brier_change
            <= float(gate["maximum_integrated_brier_increase"])
        )
    return {
        "candidate": candidate["candidate"]["name"],
        "passed": bool(all(checks.values())),
        "checks": checks,
        "macro_c_index_gain": c_gain,
        "macro_integrated_auc_gain": auc_gain,
        "macro_integrated_brier_change": brier_change,
        "improved_outer_groups": int(
            sum(delta > 0.0 for delta in deltas)
        ),
        "fold_c_index_deltas": deltas,
        "macro_auc_horizon_deltas": horizon_auc_deltas,
        "worst_group_delta": float(min(deltas)),
    }


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
    config = _load_config(ROOT / str(plan["base_config"]), data_dir)
    site_table = build_site_feature_table(data_dir / ORAL_GUT_FILE)
    aggregates: list[dict[str, Any]] = []
    for candidate in candidates:
        folds: list[dict[str, Any]] = []
        for outer_test_group in plan["outer_test_groups"]:
            inner_validation_group = (int(outer_test_group) + 1) % 5
            bundle = build_nested_bundle(
                config,
                data_dir=data_dir,
                outer_test_group=int(outer_test_group),
                inner_validation_group=inner_validation_group,
                site_table=site_table,
            )
            refit_bundle = build_refit_bundle(
                config,
                data_dir=data_dir,
                outer_test_group=int(outer_test_group),
                site_table=site_table,
            )
            fold_root = (
                output_root
                / phase
                / candidate["name"]
                / (
                    f"outer_group{outer_test_group}_"
                    f"val{inner_validation_group}"
                )
            )
            run_dirs: list[Path] = []
            run_summaries: list[dict[str, Any]] = []
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
                run_summaries.append(run)
                print(
                    f"{phase} {candidate['name']} "
                    f"outer={outer_test_group} "
                    f"val={inner_validation_group} seed={model_seed} "
                    f"valC={run['selection_validation_metrics']['harrell_c_index']:.6f} "
                    f"testC={run['test_metrics']['harrell_c_index']:.6f} "
                    f"iAUC={run['test_metrics']['normalized_integrated_auc']:.6f} "
                    f"epoch={run['best_epoch']}",
                    flush=True,
                )
            ensemble = _ensemble_runs(
                run_dirs,
                plan=plan,
                output_path=fold_root / "ensemble_predictions.npz",
            )
            fold = {
                "outer_test_group": int(outer_test_group),
                "inner_validation_group": inner_validation_group,
                "model_seeds": [int(value) for value in model_seeds],
                "member_best_epochs": [
                    int(run["best_epoch"]) for run in run_summaries
                ],
                "mean_selection_validation_c_index": float(
                    statistics.mean(
                        run["selection_validation_metrics"][
                            "harrell_c_index"
                        ]
                        for run in run_summaries
                    )
                ),
                **ensemble,
            }
            _write_json(fold_root / "ensemble_summary.json", fold)
            folds.append(fold)
        aggregate = _aggregate_candidate(candidate, folds)
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
    plan = _read_json(plan_path)
    if plan["status"] != "locked_before_development_generation":
        raise RuntimeError("Development plan is not locked.")
    manifest_path = data_dir / DATA_FILE_NAMES["manifest_json"]
    manifest = _read_json(manifest_path)
    if int(manifest["seed"]) != int(plan["development_generation_seed"]):
        raise RuntimeError("Development cohort seed does not match the plan.")
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
    decisions = [
        _gate_comparison(
            row,
            baseline,
            plan["development_gate"],
            audit=False,
        )
        for row in aggregates
        if row["candidate"].get(
            "eligible_for_internal_relation_promotion",
            False,
        )
    ]
    passed_names = {
        row["candidate"] for row in decisions if row["passed"]
    }
    passed = [
        row
        for row in aggregates
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
        "scope": "fresh_nested_logo_development_only",
        "plan_sha256": _sha256(plan_path),
        "manifest_sha256": _sha256(manifest_path),
        "aggregates": aggregates,
        "selection_decisions": decisions,
        "selected_candidate": selected["candidate"],
        "candidate_passed_gate": bool(passed),
        "audit_cohort_generated": False,
        "outer_test_labels_used_only_for_development_promotion": True,
    }
    _write_json(
        output_root / "development/development_summary.json",
        summary,
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
        "baseline_candidate": baseline["candidate"],
        "selected_candidate": selected["candidate"],
        "audit_generation_seed": int(plan["audit_generation_seed"]),
        "audit_cohort_generated": False,
        "selection_decisions": decisions,
    }
    _write_json(EXPERIMENT_DIR / "protocol_lock.json", lock)
    return summary


def run_audit(
    *,
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    device_arg: str,
    resume: bool,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    lock_path = EXPERIMENT_DIR / "protocol_lock.json"
    lock = _read_json(lock_path)
    if (
        lock["status"]
        != "locked_after_development_before_audit_generation"
    ):
        raise RuntimeError("No candidate is eligible for one-time audit.")
    manifest_path = data_dir / DATA_FILE_NAMES["manifest_json"]
    manifest = _read_json(manifest_path)
    if int(manifest["seed"]) != int(lock["audit_generation_seed"]):
        raise RuntimeError("Audit cohort seed does not match the lock.")
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
    adoption = _gate_comparison(
        selected_result,
        baseline_result,
        plan["audit_gate"],
        audit=True,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "one_time_nested_logo_audit",
        "plan_sha256": _sha256(plan_path),
        "protocol_lock_sha256_before_audit": _sha256(lock_path),
        "manifest_sha256": _sha256(manifest_path),
        "aggregates": aggregates,
        "adoption_decision": adoption,
        "audit_seed_reruns_prohibited": True,
        "test_labels_used_for_training_or_selection": False,
    }
    _write_json(output_root / "audit/audit_summary.json", summary)
    lock.update(
        {
            "status": (
                "audit_passed_adoption_gate"
                if adoption["passed"]
                else "audit_rejected_candidate"
            ),
            "audit_cohort_generated": True,
            "audit_manifest_sha256": _sha256(manifest_path),
            "audit_adoption_decision": adoption,
        }
    )
    _write_json(lock_path, lock)
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
        choices=["auto", "cpu", "cuda"],
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
    if args.phase == "development":
        result = run_development(**common)
    else:
        result = run_audit(**common)
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
