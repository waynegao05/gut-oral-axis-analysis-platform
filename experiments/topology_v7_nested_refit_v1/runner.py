from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from experiments.topology_v7_nested_refit_v1.model import (
    FixedEdgeResidualModel,
    fit_edge_standardizer,
)
from research.data import build_dataset_from_csv, set_seed
from research.ensemble_v2 import build_model
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import build_scheduler, resolve_device


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN_PATH = EXPERIMENT_DIR / "experiment_plan.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/topology_v7_nested_refit_v1"
DATA_FILE_NAMES = {
    "graph_csv": "topology_v7_sample_graph_table.csv",
    "clinical_csv": "topology_v7_sample_clinical_table.csv",
    "metabolite_csv": "topology_v7_sample_metabolite_table.csv",
    "label_csv": "topology_v7_sample_label_table.csv",
    "provenance_csv": "topology_v7_sample_provenance.csv",
    "manifest_json": "topology_v7_manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        _as_builtin(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


@dataclass
class HoldoutBundle:
    train_set: list[Any]
    eval_set: list[Any]
    node_feature_dim: int
    clinical_dim: int
    metabolite_dim: int
    num_node_types: int
    node_type_names: list[str]
    holdout_group: int
    train_groups: list[int]
    standardizer: dict[str, Any]
    split_summary: dict[str, Any]


@dataclass(frozen=True)
class PairCache:
    earlier: torch.Tensor
    later: torch.Tensor
    weights: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.earlier.numel())

    def sample(
        self,
        *,
        max_pairs: int,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            empty_index = torch.empty(0, dtype=torch.long, device=device)
            empty_weight = torch.empty(0, dtype=torch.float32, device=device)
            return empty_index, empty_index, empty_weight
        if self.size <= max_pairs:
            selected = torch.arange(self.size, dtype=torch.long)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            selected = torch.randint(
                low=0,
                high=self.size,
                size=(int(max_pairs),),
                generator=generator,
                dtype=torch.long,
            )
        return (
            self.earlier[selected].to(device),
            self.later[selected].to(device),
            self.weights[selected].to(device),
        )


def _load_config(base_config_path: Path, data_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    for key, file_name in DATA_FILE_NAMES.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        config["paths"][key] = str(path.as_posix())
    return config


def _group_map(provenance_csv: Path) -> dict[str, int]:
    provenance = pd.read_csv(provenance_csv)
    required = {"sample_id", "generation_group_id"}
    missing = sorted(required.difference(provenance.columns))
    if missing:
        raise ValueError(f"Provenance is missing required columns: {missing}")
    if provenance["sample_id"].duplicated().any():
        raise ValueError("Provenance contains duplicate sample_id values.")
    return {
        str(sample_id): int(group_id)
        for sample_id, group_id in provenance[
            ["sample_id", "generation_group_id"]
        ].itertuples(index=False, name=None)
    }


def _attach_groups(items: Sequence[Any], groups: dict[str, int]) -> None:
    for item in items:
        sample_id = str(item.sample_id)
        if sample_id not in groups:
            raise ValueError(f"Sample {sample_id} is absent from provenance.")
        item.generation_group_id = torch.tensor(groups[sample_id], dtype=torch.long)


def _standardize_tensor_attribute(
    train_set: Sequence[Any],
    evaluation_sets: Sequence[Sequence[Any]],
    attribute: str,
) -> dict[str, Any]:
    values = torch.stack(
        [getattr(item, attribute).detach().float().view(-1) for item in train_set],
        dim=0,
    )
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    zero_variance = scale <= 1e-12
    scale = torch.where(zero_variance, torch.ones_like(scale), scale)
    for item in [*train_set, *(item for data_set in evaluation_sets for item in data_set)]:
        raw = getattr(item, attribute).detach().float().view(-1)
        setattr(item, attribute, (raw - mean) / scale)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "zero_variance": zero_variance.tolist(),
    }


def build_holdout_bundle(
    config: dict[str, Any],
    *,
    holdout_group: int,
) -> HoldoutBundle:
    groups = [0, 1, 2, 3, 4]
    if int(holdout_group) not in groups:
        raise ValueError(f"holdout_group must be one of {groups}.")
    helper_validation_group = groups[(groups.index(int(holdout_group)) + 1) % len(groups)]
    graph_preprocess = config.get("graph_preprocess", {})
    raw_bundle = build_dataset_from_csv(
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
        validation_group=helper_validation_group,
        test_group=int(holdout_group),
    )
    train_set = sorted(
        [*raw_bundle.train_set, *raw_bundle.val_set],
        key=lambda item: str(item.sample_id),
    )
    eval_set = sorted(raw_bundle.test_set, key=lambda item: str(item.sample_id))
    groups_by_sample = _group_map(Path(config["paths"]["provenance_csv"]))
    _attach_groups(train_set, groups_by_sample)
    _attach_groups(eval_set, groups_by_sample)

    train_groups = sorted(
        {int(item.generation_group_id.item()) for item in train_set}
    )
    eval_groups = sorted({int(item.generation_group_id.item()) for item in eval_set})
    expected_train_groups = [group for group in groups if group != int(holdout_group)]
    if train_groups != expected_train_groups:
        raise RuntimeError(
            f"Expected training groups {expected_train_groups}, got {train_groups}."
        )
    if eval_groups != [int(holdout_group)]:
        raise RuntimeError(
            f"Expected evaluation group {[int(holdout_group)]}, got {eval_groups}."
        )
    if len(train_set) != 2880 or len(eval_set) != 720:
        raise RuntimeError(
            f"Expected 2880/720 refit split, got {len(train_set)}/{len(eval_set)}."
        )

    standardizer = {
        "fit_scope": "all_four_non_holdout_groups",
        "clinical": _standardize_tensor_attribute(
            train_set, [eval_set], "clinical"
        ),
        "metabolites": _standardize_tensor_attribute(
            train_set, [eval_set], "metabolites"
        ),
    }
    split_summary = {
        "strategy": "four_group_refit_one_group_holdout",
        "holdout_group": int(holdout_group),
        "helper_group_recombined_into_training": int(helper_validation_group),
        "train_groups": train_groups,
        "evaluation_groups": eval_groups,
        "num_train": len(train_set),
        "num_evaluation": len(eval_set),
        "generation_group_used_as_model_feature": False,
    }
    return HoldoutBundle(
        train_set=train_set,
        eval_set=eval_set,
        node_feature_dim=raw_bundle.node_feature_dim,
        clinical_dim=raw_bundle.clinical_dim,
        metabolite_dim=raw_bundle.metabolite_dim,
        num_node_types=raw_bundle.num_node_types,
        node_type_names=list(raw_bundle.node_type_names),
        holdout_group=int(holdout_group),
        train_groups=train_groups,
        standardizer=standardizer,
        split_summary=split_summary,
    )


def _kaplan_meier_censor_survival(
    time_values: torch.Tensor,
    event_values: torch.Tensor,
    *,
    minimum_survival: float = 0.1,
) -> torch.Tensor:
    times = time_values.detach().cpu().float()
    events = event_values.detach().cpu().float()
    result = torch.ones_like(times)
    survival = 1.0
    for observed_time in torch.unique(times, sorted=True):
        at_time = times == observed_time
        result[at_time] = max(float(survival), float(minimum_survival))
        at_risk = int((times >= observed_time).sum().item())
        censored = int((at_time & (events <= 0)).sum().item())
        if at_risk > 0 and censored > 0:
            survival *= max(0.0, 1.0 - float(censored) / float(at_risk))
    return torch.clamp(result, min=float(minimum_survival), max=1.0)


def build_pair_cache(
    time_values: torch.Tensor,
    event_values: torch.Tensor,
    *,
    use_ipcw: bool,
    maximum_weight: float = 10.0,
) -> PairCache:
    times = time_values.detach().cpu().float().view(-1)
    events = event_values.detach().cpu().float().view(-1)
    comparable = (events[:, None] > 0) & (times[:, None] < times[None, :])
    earlier, later = torch.nonzero(comparable, as_tuple=True)
    if earlier.numel() == 0:
        return PairCache(
            earlier=earlier.long(),
            later=later.long(),
            weights=torch.empty(0, dtype=torch.float32),
        )
    if use_ipcw:
        censor_survival = _kaplan_meier_censor_survival(times, events)
        sample_weights = torch.clamp(
            1.0 / censor_survival.pow(2),
            max=float(maximum_weight),
        )
        weights = sample_weights[earlier]
        weights = weights / torch.clamp(weights.mean(), min=1e-8)
    else:
        weights = torch.ones(earlier.numel(), dtype=torch.float32)
    return PairCache(
        earlier=earlier.long(),
        later=later.long(),
        weights=weights.float(),
    )


def soft_pairwise_ranking_loss(
    risk: torch.Tensor,
    *,
    earlier: torch.Tensor,
    later: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if earlier.numel() == 0:
        return torch.zeros((), device=risk.device, dtype=risk.dtype)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    differences = (risk[earlier] - risk[later]) / float(temperature)
    losses = F.softplus(-differences)
    normalized_weights = weights.to(losses.dtype)
    return (losses * normalized_weights).sum() / torch.clamp(
        normalized_weights.sum(), min=1e-8
    )


def equal_group_cox_loss(
    risk: torch.Tensor,
    time_values: torch.Tensor,
    event_values: torch.Tensor,
    group_values: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group_id in torch.unique(group_values, sorted=True):
        mask = group_values == group_id
        if int(event_values[mask].sum().item()) == 0:
            continue
        losses.append(
            cox_ph_loss(
                risk[mask],
                time_values[mask],
                event_values[mask],
                ties_method="breslow",
            )
        )
    if not losses:
        raise RuntimeError("No training group contains an observed event.")
    return torch.stack(losses).mean()


def _predict(
    model: torch.nn.Module,
    data_set: Sequence[Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(data_set, batch_size=int(batch_size), shuffle=False)
    model.eval()
    sample_ids: list[str] = []
    times: list[float] = []
    events: list[float] = []
    risks: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, compute_contrastive=False)
            sample_ids.extend(str(value) for value in batch.sample_id)
            times.extend(float(value) for value in batch.time.detach().cpu().tolist())
            events.extend(float(value) for value in batch.event.detach().cpu().tolist())
            risks.extend(float(value) for value in output["risk"].detach().cpu().tolist())
    risk_tensor = torch.tensor(risks, dtype=torch.float32, device=device)
    time_tensor = torch.tensor(times, dtype=torch.float32, device=device)
    event_tensor = torch.tensor(events, dtype=torch.float32, device=device)
    return {
        "sample_ids": sample_ids,
        "time": np.asarray(times, dtype=float),
        "event": np.asarray(events, dtype=float),
        "risk": np.asarray(risks, dtype=float),
        "c_index": float(concordance_index(times, events, risks)),
        "cox_loss": float(
            cox_ph_loss(
                risk_tensor,
                time_tensor,
                event_tensor,
                ties_method="breslow",
            ).item()
        ),
    }


def _save_predictions(
    path: Path,
    train_predictions: dict[str, Any],
    eval_predictions: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_sample_ids=np.asarray(train_predictions["sample_ids"], dtype=str),
        train_time=train_predictions["time"],
        train_event=train_predictions["event"],
        train_risk=train_predictions["risk"],
        eval_sample_ids=np.asarray(eval_predictions["sample_ids"], dtype=str),
        eval_time=eval_predictions["time"],
        eval_event=eval_predictions["event"],
        eval_risk=eval_predictions["risk"],
    )


def _training_tensors(data_set: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.tensor(
        [float(item.time.item()) for item in data_set], dtype=torch.float32
    )
    events = torch.tensor(
        [float(item.event.item()) for item in data_set], dtype=torch.float32
    )
    return times, events


def train_one(
    *,
    config: dict[str, Any],
    bundle: HoldoutBundle,
    candidate: dict[str, Any],
    seed: int,
    device_arg: str,
    output_dir: Path,
    maximum_epochs: int,
    early_stop_patience: int,
    minimum_c_index_delta: float,
    ranking_max_pairs: int,
    ranking_temperature: float,
    ranking_warmup_epochs: int,
    fixed_epochs: int | None,
    resume: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "run_summary.json"
    checkpoint_path = output_dir / "model.pt"
    predictions_path = output_dir / "predictions.npz"
    epochs_to_run = int(fixed_epochs if fixed_epochs is not None else maximum_epochs)
    scheduler_horizon_epochs = int(maximum_epochs)
    run_fingerprint_payload = {
        "runner_sha256": _sha256(Path(__file__)),
        "manifest_sha256": _sha256(Path(config["paths"]["manifest_json"])),
        "config": config,
        "candidate": candidate,
        "seed": int(seed),
        "holdout_group": int(bundle.holdout_group),
        "train_sample_ids": [str(item.sample_id) for item in bundle.train_set],
        "evaluation_sample_ids": [str(item.sample_id) for item in bundle.eval_set],
        "standardizer": bundle.standardizer,
        "epochs_to_run": epochs_to_run,
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "early_stop_patience": int(early_stop_patience),
        "minimum_c_index_delta": float(minimum_c_index_delta),
        "ranking_max_pairs": int(ranking_max_pairs),
        "ranking_temperature": float(ranking_temperature),
        "ranking_warmup_epochs": int(ranking_warmup_epochs),
        "fixed_epoch_training": fixed_epochs is not None,
    }
    run_fingerprint = _json_digest(run_fingerprint_payload)
    if (
        resume
        and summary_path.exists()
        and checkpoint_path.exists()
        and predictions_path.exists()
    ):
        existing = _read_json(summary_path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Refusing to resume mismatched run artifacts: {output_dir}"
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))
    device = resolve_device(device_arg)
    run_config = copy.deepcopy(config)
    run_config["seed"] = int(seed)
    run_config["train"]["batch_size"] = int(
        run_config["train"].get("batch_size", 4096)
    )
    model = build_model(run_config, bundle, device)
    edge_standardizer: dict[str, Any] | None = None
    if bool(candidate.get("fixed_edge_residual", False)):
        edge_mean, edge_scale = fit_edge_standardizer(
            bundle.train_set,
            num_node_types=bundle.num_node_types,
        )
        edge_standardizer = {
            "fit_scope": "all_four_non_holdout_groups",
            "mean": edge_mean.tolist(),
            "scale": edge_scale.tolist(),
        }
        model = FixedEdgeResidualModel(
            model,
            num_node_types=bundle.num_node_types,
            edge_mean=edge_mean,
            edge_scale=edge_scale,
            hidden_dim=int(candidate.get("fixed_edge_hidden_dim", 16)),
            residual_scale=float(
                candidate.get("fixed_edge_residual_scale", 0.1)
            ),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["train"]["lr"]),
        weight_decay=float(run_config["train"]["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=scheduler_horizon_epochs,
        warmup_epochs=min(
            int(run_config["train"].get("warmup_epochs", 10)),
            max(1, scheduler_horizon_epochs // 3),
        ),
    )
    batch_size = int(run_config["train"]["batch_size"])
    train_loader = DataLoader(
        bundle.train_set,
        batch_size=batch_size,
        shuffle=False,
    )
    if len(train_loader) != 1:
        raise RuntimeError(
            "Nested refit v1 requires exact full-cohort Cox training in one batch."
        )
    train_times, train_events = _training_tensors(bundle.train_set)
    pair_cache = build_pair_cache(
        train_times,
        train_events,
        use_ipcw=bool(candidate["ipcw_ranking"]),
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_c_index = float("-inf")
    best_epoch = 0
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        batch = next(iter(train_loader)).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, compute_contrastive=False)
        pooled_cox_loss = cox_ph_loss(
            output["risk"],
            batch.time,
            batch.event,
            ties_method="breslow",
        )
        if candidate["cox_mode"] == "pooled":
            cox_loss = pooled_cox_loss
        elif candidate["cox_mode"] in {"equal_group", "mixed_group"}:
            grouped_cox_loss = equal_group_cox_loss(
                output["risk"],
                batch.time,
                batch.event,
                batch.generation_group_id.view(-1),
            )
            if candidate["cox_mode"] == "equal_group":
                cox_loss = grouped_cox_loss
            else:
                cox_loss = 0.5 * pooled_cox_loss + 0.5 * grouped_cox_loss
        else:
            raise ValueError(f"Unsupported cox_mode: {candidate['cox_mode']}")

        ranking_scale = min(
            1.0,
            float(epoch) / float(max(1, int(ranking_warmup_epochs))),
        )
        effective_ranking_weight = float(candidate["ranking_weight"]) * ranking_scale
        if effective_ranking_weight > 0:
            earlier, later, pair_weights = pair_cache.sample(
                max_pairs=int(ranking_max_pairs),
                seed=int(seed) * 10000 + int(epoch),
                device=device,
            )
            standardized_risk = (
                output["risk"] - output["risk"].mean()
            ) / torch.clamp(
                output["risk"].std(unbiased=False),
                min=1e-6,
            )
            ranking_loss = soft_pairwise_ranking_loss(
                standardized_risk,
                earlier=earlier,
                later=later,
                weights=pair_weights,
                temperature=float(ranking_temperature),
            )
        else:
            ranking_loss = torch.zeros((), device=device)
        total_loss = cox_loss + effective_ranking_weight * ranking_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(run_config["train"].get("grad_clip_norm", 2.0)),
        )
        optimizer.step()
        scheduler.step()

        epoch_row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_objective": float(total_loss.item()),
            "train_cox_loss": float(cox_loss.item()),
            "train_ranking_loss": float(ranking_loss.item()),
            "effective_ranking_weight": float(effective_ranking_weight),
        }
        if fixed_epochs is None:
            validation = _predict(
                model,
                bundle.eval_set,
                device=device,
                batch_size=batch_size,
            )
            epoch_row.update(
                {
                    "validation_c_index": validation["c_index"],
                    "validation_cox_loss": validation["cox_loss"],
                }
            )
            if (
                float(validation["c_index"])
                > best_validation_c_index + float(minimum_c_index_delta)
            ):
                best_validation_c_index = float(validation["c_index"])
                best_epoch = int(epoch)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= int(early_stop_patience):
                    history.append(epoch_row)
                    break
        history.append(epoch_row)

    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("Development training did not produce a checkpoint.")
        model.load_state_dict(best_state)
    else:
        best_epoch = int(epochs_to_run)
        best_validation_c_index = float("nan")

    torch.save(model.state_dict(), checkpoint_path)
    train_predictions = _predict(
        model, bundle.train_set, device=device, batch_size=batch_size
    )
    eval_predictions = _predict(
        model, bundle.eval_set, device=device, batch_size=batch_size
    )
    _save_predictions(predictions_path, train_predictions, eval_predictions)
    summary = {
        "schema_version": 1,
        "candidate": copy.deepcopy(candidate),
        "seed": int(seed),
        "holdout_group": int(bundle.holdout_group),
        "train_groups": list(bundle.train_groups),
        "num_train": len(bundle.train_set),
        "num_evaluation": len(bundle.eval_set),
        "fixed_epoch_training": fixed_epochs is not None,
        "epochs_to_run": int(epochs_to_run),
        "scheduler_horizon_epochs": int(scheduler_horizon_epochs),
        "epochs_run": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_c_index": (
            None
            if not math.isfinite(best_validation_c_index)
            else float(best_validation_c_index)
        ),
        "train_c_index": float(train_predictions["c_index"]),
        "train_cox_loss": float(train_predictions["cox_loss"]),
        "evaluation_c_index": float(eval_predictions["c_index"]),
        "evaluation_cox_loss": float(eval_predictions["cox_loss"]),
        "num_comparable_training_pairs": int(pair_cache.size),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": float(time.perf_counter() - started),
        "split_summary": bundle.split_summary,
        "standardizer": bundle.standardizer,
        "edge_standardizer": edge_standardizer,
        "generation_group_used_as_model_feature": False,
        "run_fingerprint": run_fingerprint,
        "checkpoint_path": checkpoint_path.as_posix(),
        "predictions_path": predictions_path.as_posix(),
    }
    _write_json(output_dir / "history.json", {"epochs": history})
    _write_json(summary_path, _as_builtin(summary))
    return summary


def _aggregate_candidate_runs(
    candidate: dict[str, Any],
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scores = [float(run["evaluation_c_index"]) for run in runs]
    losses = [float(run["evaluation_cox_loss"]) for run in runs]
    best_epochs = [int(run["best_epoch"]) for run in runs]
    return {
        "candidate": copy.deepcopy(candidate),
        "num_folds": len(runs),
        "macro_mean_c_index": float(statistics.mean(scores)),
        "macro_std_c_index": float(statistics.stdev(scores)),
        "minimum_group_c_index": float(min(scores)),
        "maximum_group_c_index": float(max(scores)),
        "mean_cox_loss": float(statistics.mean(losses)),
        "median_best_epoch": int(round(statistics.median(best_epochs))),
        "folds": [
            {
                "holdout_group": int(run["holdout_group"]),
                "c_index": float(run["evaluation_c_index"]),
                "cox_loss": float(run["evaluation_cox_loss"]),
                "best_epoch": int(run["best_epoch"]),
            }
            for run in sorted(runs, key=lambda row: int(row["holdout_group"]))
        ],
    }


def _select_development_candidate(
    plan: dict[str, Any],
    aggregates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_name = {
        str(row["candidate"]["name"]): row
        for row in aggregates
    }
    fallback_name = str(plan["development_selection"]["fallback"])
    baseline = by_name[fallback_name]
    baseline_by_group = {
        int(row["holdout_group"]): row for row in baseline["folds"]
    }
    decisions: list[dict[str, Any]] = []
    for aggregate in aggregates:
        candidate_name = str(aggregate["candidate"]["name"])
        candidate_by_group = {
            int(row["holdout_group"]): row for row in aggregate["folds"]
        }
        deltas = [
            float(candidate_by_group[group]["c_index"])
            - float(baseline_by_group[group]["c_index"])
            for group in sorted(baseline_by_group)
        ]
        checks = {
            "macro_gain": bool(
                float(aggregate["macro_mean_c_index"])
                - float(baseline["macro_mean_c_index"])
                >= float(
                    plan["development_selection"][
                        "minimum_macro_c_index_gain_over_baseline"
                    ]
                )
            ),
            "improved_groups": bool(
                sum(delta > 0 for delta in deltas)
                >= int(plan["development_selection"]["minimum_improved_groups"])
            ),
            "worst_group_preserved": bool(
                float(aggregate["minimum_group_c_index"])
                >= float(baseline["minimum_group_c_index"])
                - float(
                    plan["development_selection"][
                        "maximum_worst_group_regression"
                    ]
                )
            ),
            "cox_loss_preserved": bool(
                float(aggregate["mean_cox_loss"])
                <= float(baseline["mean_cox_loss"])
                + float(
                    plan["development_selection"][
                        "maximum_mean_cox_loss_increase"
                    ]
                )
            ),
        }
        decisions.append(
            {
                "candidate_name": candidate_name,
                "fold_c_index_deltas_vs_baseline": deltas,
                "macro_c_index_delta_vs_baseline": float(
                    aggregate["macro_mean_c_index"]
                    - baseline["macro_mean_c_index"]
                ),
                "mean_cox_loss_delta_vs_baseline": float(
                    aggregate["mean_cox_loss"] - baseline["mean_cox_loss"]
                ),
                "checks": checks,
                "eligible": candidate_name != fallback_name and all(checks.values()),
            }
        )
    eligible_names = [
        row["candidate_name"] for row in decisions if bool(row["eligible"])
    ]
    if eligible_names:
        selected_name = max(
            eligible_names,
            key=lambda name: float(by_name[name]["macro_mean_c_index"]),
        )
    else:
        selected_name = fallback_name
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
    manifest_path = data_dir / DATA_FILE_NAMES["manifest_json"]
    manifest = _read_json(manifest_path)
    if int(manifest["seed"]) != int(plan["development_generation_seed"]):
        raise RuntimeError("Development dataset seed does not match the experiment plan.")
    base_config_path = ROOT / str(plan["base_config"])
    config = _load_config(base_config_path, data_dir)
    training_plan = plan["development_training"]
    development_root = output_root / "development"
    all_aggregates: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        runs: list[dict[str, Any]] = []
        for holdout_group in plan["groups"]:
            bundle = build_holdout_bundle(
                config,
                holdout_group=int(holdout_group),
            )
            run = train_one(
                config=config,
                bundle=bundle,
                candidate=candidate,
                seed=int(plan["development_model_seed"]),
                device_arg=device_arg,
                output_dir=(
                    development_root
                    / str(candidate["name"])
                    / f"holdout_group{int(holdout_group)}"
                ),
                maximum_epochs=int(training_plan["maximum_epochs"]),
                early_stop_patience=int(training_plan["early_stop_patience"]),
                minimum_c_index_delta=float(
                    training_plan["minimum_c_index_delta"]
                ),
                ranking_max_pairs=int(training_plan["ranking_max_pairs"]),
                ranking_temperature=float(training_plan["ranking_temperature"]),
                ranking_warmup_epochs=int(
                    training_plan["ranking_warmup_epochs"]
                ),
                fixed_epochs=None,
                resume=resume,
            )
            runs.append(run)
            print(
                f"development {candidate['name']} holdout={holdout_group} "
                f"c={run['evaluation_c_index']:.6f} epoch={run['best_epoch']}",
                flush=True,
            )
        all_aggregates.append(_aggregate_candidate_runs(candidate, runs))

    selected, decisions = _select_development_candidate(plan, all_aggregates)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "selection_scope": "development_cohort_only",
        "plan_path": plan_path.as_posix(),
        "plan_sha256": _sha256(plan_path),
        "development_manifest_path": manifest_path.as_posix(),
        "development_manifest_sha256": _sha256(manifest_path),
        "aggregates": all_aggregates,
        "selection_decisions": decisions,
        "selected_candidate": selected,
    }
    _write_json(development_root / "development_summary.json", _as_builtin(summary))

    audit_data_dir = output_root / "cohorts" / (
        f"audit_seed{int(plan['audit_generation_seed'])}"
    )
    lock_path = EXPERIMENT_DIR / "protocol_lock.json"
    if lock_path.exists():
        existing = _read_json(lock_path)
        if (
            existing.get("plan_sha256") != _sha256(plan_path)
            or existing.get("development_manifest_sha256") != _sha256(manifest_path)
        ):
            raise RuntimeError("Existing protocol lock does not match this development run.")
    else:
        if audit_data_dir.exists():
            raise RuntimeError(
                "Audit data already exists before the protocol was locked."
            )
        fixed_epochs = {
            str(row["candidate"]["name"]): int(row["median_best_epoch"])
            for row in all_aggregates
        }
        lock = {
            "schema_version": 1,
            "status": "locked_after_development_before_audit_generation",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_path": plan_path.relative_to(ROOT).as_posix(),
            "plan_sha256": _sha256(plan_path),
            "development_manifest_path": manifest_path.as_posix(),
            "development_manifest_sha256": _sha256(manifest_path),
            "development_summary_path": (
                development_root / "development_summary.json"
            ).as_posix(),
            "development_summary_sha256": _sha256(
                development_root / "development_summary.json"
            ),
            "baseline_candidate": next(
                candidate
                for candidate in plan["candidates"]
                if candidate["name"] == plan["development_selection"]["fallback"]
            ),
            "selected_candidate": selected["candidate"],
            "fixed_epochs_by_candidate": fixed_epochs,
            "audit_generation_seed": int(plan["audit_generation_seed"]),
            "audit_model_seeds": [
                int(seed) for seed in plan["audit_model_seeds"]
            ],
            "audit_generation_seed_reruns_prohibited": True,
            "audit_test_labels_used_for_selection": False,
        }
        _write_json(lock_path, _as_builtin(lock))
    return summary


def _load_prediction_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {key: values[key].copy() for key in values.files}


def _ensemble_candidate(
    *,
    candidate: dict[str, Any],
    runs_by_group: dict[int, list[dict[str, Any]]],
    predictions_path: Path,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    pooled_sample_ids: list[str] = []
    pooled_group: list[int] = []
    pooled_time: list[float] = []
    pooled_event: list[float] = []
    pooled_risk: list[float] = []
    all_member_scores: list[float] = []
    for holdout_group in sorted(runs_by_group):
        runs = runs_by_group[holdout_group]
        predictions = [
            _load_prediction_file(Path(run["predictions_path"])) for run in runs
        ]
        reference = predictions[0]
        for item in predictions[1:]:
            if not np.array_equal(item["eval_sample_ids"], reference["eval_sample_ids"]):
                raise RuntimeError("Audit evaluation predictions are not aligned.")
            if not np.array_equal(item["train_sample_ids"], reference["train_sample_ids"]):
                raise RuntimeError("Audit training predictions are not aligned.")
        eval_risk = np.mean(
            np.stack([item["eval_risk"] for item in predictions], axis=0),
            axis=0,
        )
        train_risk = np.mean(
            np.stack([item["train_risk"] for item in predictions], axis=0),
            axis=0,
        )
        eval_time = reference["eval_time"].astype(float)
        eval_event = reference["eval_event"].astype(float)
        fold_c_index = float(
            concordance_index(eval_time, eval_event, eval_risk)
        )
        fold_cox_loss = float(
            cox_ph_loss(
                torch.tensor(eval_risk, dtype=torch.float32),
                torch.tensor(eval_time, dtype=torch.float32),
                torch.tensor(eval_event, dtype=torch.float32),
                ties_method="breslow",
            ).item()
        )
        train_mean = float(train_risk.mean())
        train_std = float(train_risk.std())
        if train_std <= 1e-8:
            raise RuntimeError("Audit ensemble training risk has zero variance.")
        standardized_eval_risk = (eval_risk - train_mean) / train_std
        pooled_sample_ids.extend(reference["eval_sample_ids"].astype(str).tolist())
        pooled_group.extend([int(holdout_group)] * len(eval_time))
        pooled_time.extend(eval_time.tolist())
        pooled_event.extend(eval_event.tolist())
        pooled_risk.extend(standardized_eval_risk.tolist())
        member_scores = [float(run["evaluation_c_index"]) for run in runs]
        all_member_scores.extend(member_scores)
        folds.append(
            {
                "holdout_group": int(holdout_group),
                "num_models": len(runs),
                "member_mean_c_index": float(statistics.mean(member_scores)),
                "member_std_c_index": float(statistics.stdev(member_scores)),
                "ensemble_c_index": fold_c_index,
                "ensemble_cox_loss": fold_cox_loss,
                "ensemble_gain_over_member_mean": float(
                    fold_c_index - statistics.mean(member_scores)
                ),
            }
        )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        sample_ids=np.asarray(pooled_sample_ids, dtype=str),
        holdout_group=np.asarray(pooled_group, dtype=np.int64),
        time=np.asarray(pooled_time, dtype=np.float64),
        event=np.asarray(pooled_event, dtype=np.float64),
        train_standardized_risk=np.asarray(pooled_risk, dtype=np.float64),
    )
    fold_scores = [float(row["ensemble_c_index"]) for row in folds]
    fold_losses = [float(row["ensemble_cox_loss"]) for row in folds]
    return {
        "candidate": copy.deepcopy(candidate),
        "folds": folds,
        "ensemble_predictions_path": predictions_path.as_posix(),
        "aggregate": {
            "member_mean_c_index": float(statistics.mean(all_member_scores)),
            "member_std_c_index": float(statistics.stdev(all_member_scores)),
            "macro_mean_ensemble_c_index": float(statistics.mean(fold_scores)),
            "macro_std_ensemble_c_index": float(statistics.stdev(fold_scores)),
            "minimum_group_ensemble_c_index": float(min(fold_scores)),
            "maximum_group_ensemble_c_index": float(max(fold_scores)),
            "mean_ensemble_cox_loss": float(statistics.mean(fold_losses)),
            "train_standardized_pooled_oof_c_index": float(
                concordance_index(pooled_time, pooled_event, pooled_risk)
            ),
        },
    }


def _cluster_pair_matrices(
    *,
    time_values: np.ndarray,
    event_values: np.ndarray,
    risks: dict[str, np.ndarray],
    cluster_index: np.ndarray,
    group_values: np.ndarray,
    num_clusters: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[int, np.ndarray],
    dict[str, dict[int, np.ndarray]],
]:
    time_values = np.asarray(time_values, dtype=float)
    event_values = np.asarray(event_values, dtype=float)
    cluster_index = np.asarray(cluster_index, dtype=np.int64)
    group_values = np.asarray(group_values, dtype=np.int64)
    risk_arrays = {
        str(name): np.asarray(values, dtype=float) for name, values in risks.items()
    }
    n = len(time_values)
    if not all(
        len(values) == n
        for values in [
            event_values,
            cluster_index,
            group_values,
            *risk_arrays.values(),
        ]
    ):
        raise ValueError("Cluster-bootstrap arrays must have equal length.")
    if np.any(cluster_index < 0) or np.any(cluster_index >= int(num_clusters)):
        raise ValueError("Cluster index falls outside the declared cluster count.")

    denominator = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    numerators = {
        name: np.zeros_like(denominator) for name in risk_arrays
    }
    groups = [int(value) for value in sorted(np.unique(group_values))]
    group_denominators = {
        group: np.zeros_like(denominator) for group in groups
    }
    group_numerators = {
        name: {
            group: np.zeros_like(denominator) for group in groups
        }
        for name in risk_arrays
    }

    for left in range(n - 1):
        right = np.arange(left + 1, n, dtype=np.int64)
        left_is_earlier = (
            (time_values[left] < time_values[right])
            & (event_values[left] == 1)
        )
        right_is_earlier = (
            (time_values[right] < time_values[left])
            & (event_values[right] == 1)
        )
        permissible = left_is_earlier | right_is_earlier
        if not np.any(permissible):
            continue
        selected_right = right[permissible]
        selected_left_is_earlier = left_is_earlier[permissible]
        left_clusters = np.full(
            selected_right.shape,
            cluster_index[left],
            dtype=np.int64,
        )
        right_clusters = cluster_index[selected_right]
        np.add.at(
            denominator,
            (left_clusters, right_clusters),
            1.0,
        )
        same_group = group_values[selected_right] == group_values[left]
        if np.any(same_group):
            np.add.at(
                group_denominators[int(group_values[left])],
                (
                    left_clusters[same_group],
                    right_clusters[same_group],
                ),
                1.0,
            )

        for name, risk in risk_arrays.items():
            left_risk = np.full(selected_right.shape, risk[left], dtype=float)
            right_risk = risk[selected_right]
            earlier_risk = np.where(
                selected_left_is_earlier,
                left_risk,
                right_risk,
            )
            later_risk = np.where(
                selected_left_is_earlier,
                right_risk,
                left_risk,
            )
            concordance = (earlier_risk > later_risk).astype(float)
            concordance += 0.5 * (earlier_risk == later_risk)
            np.add.at(
                numerators[name],
                (left_clusters, right_clusters),
                concordance,
            )
            if np.any(same_group):
                np.add.at(
                    group_numerators[name][int(group_values[left])],
                    (
                        left_clusters[same_group],
                        right_clusters[same_group],
                    ),
                    concordance[same_group],
                )
    return denominator, numerators, group_denominators, group_numerators


def _weighted_cluster_c_index(
    counts: np.ndarray,
    *,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    weighted_numerator = np.einsum(
        "bi,ij,bj->b",
        counts,
        numerator,
        counts,
        optimize=True,
    )
    weighted_denominator = np.einsum(
        "bi,ij,bj->b",
        counts,
        denominator,
        counts,
        optimize=True,
    )
    return np.divide(
        weighted_numerator,
        weighted_denominator,
        out=np.full_like(weighted_numerator, np.nan),
        where=weighted_denominator > 0,
    )


def _interval(values: np.ndarray, confidence_level: float) -> list[float]:
    alpha = (1.0 - float(confidence_level)) / 2.0
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    ]


def _paired_anchor_cluster_bootstrap(
    *,
    plan: dict[str, Any],
    provenance_path: Path,
    baseline_predictions_path: Path,
    selected_predictions_path: Path,
) -> dict[str, Any]:
    uncertainty = plan["audit_uncertainty"]
    baseline = _load_prediction_file(baseline_predictions_path)
    selected = _load_prediction_file(selected_predictions_path)
    for field in ("sample_ids", "holdout_group", "time", "event"):
        if not np.array_equal(baseline[field], selected[field]):
            raise RuntimeError(f"Audit prediction field is misaligned: {field}")

    provenance = pd.read_csv(provenance_path)
    cluster_column = str(uncertainty["cluster_column"])
    required_columns = {"sample_id", cluster_column}
    if not required_columns.issubset(provenance.columns):
        raise RuntimeError(
            f"Audit provenance is missing columns: {sorted(required_columns)}"
        )
    if provenance["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Audit provenance contains duplicate sample IDs.")
    cluster_by_sample = provenance.set_index(
        provenance["sample_id"].astype(str)
    )[cluster_column].astype(str)
    sample_ids = baseline["sample_ids"].astype(str)
    missing = sorted(set(sample_ids).difference(cluster_by_sample.index))
    if missing:
        raise RuntimeError(
            f"Audit provenance is missing {len(missing)} prediction sample IDs."
        )
    cluster_labels = cluster_by_sample.loc[sample_ids].to_numpy(dtype=str)
    unique_clusters = np.asarray(sorted(np.unique(cluster_labels)), dtype=str)
    cluster_lookup = {
        value: index for index, value in enumerate(unique_clusters.tolist())
    }
    cluster_index = np.asarray(
        [cluster_lookup[value] for value in cluster_labels],
        dtype=np.int64,
    )
    risks = {
        "baseline": baseline["train_standardized_risk"],
        "selected": selected["train_standardized_risk"],
    }
    (
        denominator,
        numerators,
        group_denominators,
        group_numerators,
    ) = _cluster_pair_matrices(
        time_values=baseline["time"],
        event_values=baseline["event"],
        risks=risks,
        cluster_index=cluster_index,
        group_values=baseline["holdout_group"],
        num_clusters=len(unique_clusters),
    )

    replicates = int(uncertainty["replicates"])
    rng = np.random.default_rng(int(uncertainty["seed"]))
    cluster_counts = rng.multinomial(
        len(unique_clusters),
        np.full(len(unique_clusters), 1.0 / len(unique_clusters)),
        size=replicates,
    ).astype(np.float64)
    observed_counts = np.ones((1, len(unique_clusters)), dtype=np.float64)
    confidence_level = float(uncertainty["confidence_level"])

    pooled_bootstrap = {
        name: _weighted_cluster_c_index(
            cluster_counts,
            numerator=numerators[name],
            denominator=denominator,
        )
        for name in risks
    }
    pooled_observed = {
        name: float(
            _weighted_cluster_c_index(
                observed_counts,
                numerator=numerators[name],
                denominator=denominator,
            )[0]
        )
        for name in risks
    }

    macro_bootstrap: dict[str, np.ndarray] = {}
    macro_observed: dict[str, float] = {}
    for name in risks:
        group_bootstrap = np.stack(
            [
                _weighted_cluster_c_index(
                    cluster_counts,
                    numerator=group_numerators[name][group],
                    denominator=group_denominators[group],
                )
                for group in sorted(group_denominators)
            ],
            axis=1,
        )
        group_observed = [
            float(
                _weighted_cluster_c_index(
                    observed_counts,
                    numerator=group_numerators[name][group],
                    denominator=group_denominators[group],
                )[0]
            )
            for group in sorted(group_denominators)
        ]
        macro_bootstrap[name] = np.nanmean(group_bootstrap, axis=1)
        macro_observed[name] = float(np.nanmean(group_observed))

    pooled_delta = pooled_bootstrap["selected"] - pooled_bootstrap["baseline"]
    macro_delta = macro_bootstrap["selected"] - macro_bootstrap["baseline"]
    return {
        "method": str(uncertainty["method"]),
        "cluster_column": cluster_column,
        "num_clusters": int(len(unique_clusters)),
        "num_samples": int(len(sample_ids)),
        "replicates": replicates,
        "seed": int(uncertainty["seed"]),
        "confidence_level": confidence_level,
        "pooled": {
            "baseline_observed_c_index": pooled_observed["baseline"],
            "selected_observed_c_index": pooled_observed["selected"],
            "observed_delta": float(
                pooled_observed["selected"] - pooled_observed["baseline"]
            ),
            "delta_percentile_interval": _interval(
                pooled_delta,
                confidence_level,
            ),
            "bootstrap_probability_delta_positive": float(
                np.nanmean(pooled_delta > 0)
            ),
        },
        "macro": {
            "baseline_observed_c_index": macro_observed["baseline"],
            "selected_observed_c_index": macro_observed["selected"],
            "observed_delta": float(
                macro_observed["selected"] - macro_observed["baseline"]
            ),
            "delta_percentile_interval": _interval(
                macro_delta,
                confidence_level,
            ),
            "bootstrap_probability_delta_positive": float(
                np.nanmean(macro_delta > 0)
            ),
        },
    }


def _audit_comparison(
    plan: dict[str, Any],
    baseline: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_group = {
        int(row["holdout_group"]): row for row in baseline["folds"]
    }
    selected_by_group = {
        int(row["holdout_group"]): row for row in selected["folds"]
    }
    deltas = [
        float(selected_by_group[group]["ensemble_c_index"])
        - float(baseline_by_group[group]["ensemble_c_index"])
        for group in sorted(baseline_by_group)
    ]
    macro_delta = float(
        selected["aggregate"]["macro_mean_ensemble_c_index"]
        - baseline["aggregate"]["macro_mean_ensemble_c_index"]
    )
    loss_delta = float(
        selected["aggregate"]["mean_ensemble_cox_loss"]
        - baseline["aggregate"]["mean_ensemble_cox_loss"]
    )
    distinct_candidate = (
        selected["candidate"]["name"] != baseline["candidate"]["name"]
    )
    checks = {
        "distinct_candidate": distinct_candidate,
        "macro_gain": bool(
            macro_delta
            >= float(plan["audit_adoption_gate"]["minimum_macro_c_index_gain"])
        ),
        "improved_groups": bool(
            sum(delta > 0 for delta in deltas)
            >= int(plan["audit_adoption_gate"]["minimum_improved_groups"])
        ),
        "worst_group_delta": bool(
            min(deltas)
            >= -float(
                plan["audit_adoption_gate"][
                    "maximum_worst_group_delta_regression"
                ]
            )
        ),
        "cox_loss": bool(
            loss_delta
            <= float(
                plan["audit_adoption_gate"]["maximum_mean_cox_loss_increase"]
            )
        ),
    }
    return {
        "baseline_candidate": baseline["candidate"]["name"],
        "selected_candidate": selected["candidate"]["name"],
        "fold_c_index_deltas": deltas,
        "macro_c_index_delta": macro_delta,
        "mean_cox_loss_delta": loss_delta,
        "checks": checks,
        "adopt_selected_candidate": all(checks.values()),
    }


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
    if not lock_path.exists():
        raise RuntimeError("Development protocol must be locked before audit.")
    lock = _read_json(lock_path)
    if lock["status"] != "locked_after_development_before_audit_generation":
        raise RuntimeError("Unexpected audit protocol-lock status.")
    if lock["plan_sha256"] != _sha256(plan_path):
        raise RuntimeError("Audit plan hash does not match the protocol lock.")
    manifest_path = data_dir / DATA_FILE_NAMES["manifest_json"]
    manifest = _read_json(manifest_path)
    if int(manifest["seed"]) != int(lock["audit_generation_seed"]):
        raise RuntimeError("Audit dataset seed does not match the protocol lock.")

    config = _load_config(ROOT / str(plan["base_config"]), data_dir)
    baseline_candidate = lock["baseline_candidate"]
    selected_candidate = lock["selected_candidate"]
    candidates = [baseline_candidate]
    if selected_candidate["name"] != baseline_candidate["name"]:
        candidates.append(selected_candidate)
    training_plan = plan["development_training"]
    audit_root = output_root / "audit"
    candidate_results: dict[str, Any] = {}
    for candidate in candidates:
        runs_by_group: dict[int, list[dict[str, Any]]] = {}
        fixed_epochs = int(
            lock["fixed_epochs_by_candidate"][str(candidate["name"])]
        )
        for holdout_group in plan["groups"]:
            bundle = build_holdout_bundle(
                config,
                holdout_group=int(holdout_group),
            )
            group_runs: list[dict[str, Any]] = []
            for seed in lock["audit_model_seeds"]:
                run = train_one(
                    config=config,
                    bundle=bundle,
                    candidate=candidate,
                    seed=int(seed),
                    device_arg=device_arg,
                    output_dir=(
                        audit_root
                        / str(candidate["name"])
                        / f"holdout_group{int(holdout_group)}"
                        / f"seed{int(seed)}"
                    ),
                    maximum_epochs=int(training_plan["maximum_epochs"]),
                    early_stop_patience=int(
                        training_plan["early_stop_patience"]
                    ),
                    minimum_c_index_delta=float(
                        training_plan["minimum_c_index_delta"]
                    ),
                    ranking_max_pairs=int(training_plan["ranking_max_pairs"]),
                    ranking_temperature=float(
                        training_plan["ranking_temperature"]
                    ),
                    ranking_warmup_epochs=int(
                        training_plan["ranking_warmup_epochs"]
                    ),
                    fixed_epochs=fixed_epochs,
                    resume=resume,
                )
                group_runs.append(run)
                print(
                    f"audit {candidate['name']} holdout={holdout_group} "
                    f"seed={seed} c={run['evaluation_c_index']:.6f}",
                    flush=True,
                )
            runs_by_group[int(holdout_group)] = group_runs
        candidate_results[str(candidate["name"])] = _ensemble_candidate(
            candidate=candidate,
            runs_by_group=runs_by_group,
            predictions_path=(
                audit_root
                / str(candidate["name"])
                / "ensemble_oof_predictions.npz"
            ),
        )

    baseline_result = candidate_results[str(baseline_candidate["name"])]
    selected_result = candidate_results[str(selected_candidate["name"])]
    comparison = _audit_comparison(
        plan,
        baseline_result,
        selected_result,
    )
    uncertainty = _paired_anchor_cluster_bootstrap(
        plan=plan,
        provenance_path=Path(config["paths"]["provenance_csv"]),
        baseline_predictions_path=Path(
            baseline_result["ensemble_predictions_path"]
        ),
        selected_predictions_path=Path(
            selected_result["ensemble_predictions_path"]
        ),
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "plan_path": plan_path.as_posix(),
            "plan_sha256": _sha256(plan_path),
            "lock_path": lock_path.as_posix(),
            "lock_sha256": _sha256(lock_path),
            "audit_manifest_path": manifest_path.as_posix(),
            "audit_manifest_sha256": _sha256(manifest_path),
            "audit_generation_seed": int(manifest["seed"]),
            "training_groups_per_fold": 4,
            "training_samples_per_fold": 2880,
            "held_out_samples_per_fold": 720,
            "audit_test_labels_used_for_selection": False,
            "generation_group_used_as_model_feature": False,
        },
        "candidate_results": candidate_results,
        "comparison": comparison,
        "paired_cluster_bootstrap": uncertainty,
        "decision": (
            "adopt_selected_candidate"
            if comparison["adopt_selected_candidate"]
            else "keep_baseline_refit_candidate"
        ),
    }
    _write_json(audit_root / "audit_summary.json", _as_builtin(summary))
    return summary


def run_smoke(
    *,
    plan_path: Path,
    data_dir: Path,
    output_root: Path,
    device_arg: str,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    config = _load_config(ROOT / str(plan["base_config"]), data_dir)
    candidate = plan["candidates"][0]
    bundle = build_holdout_bundle(config, holdout_group=0)
    return train_one(
        config=config,
        bundle=bundle,
        candidate=candidate,
        seed=int(plan["development_model_seed"]),
        device_arg=device_arg,
        output_dir=output_root / "smoke",
        maximum_epochs=2,
        early_stop_patience=1,
        minimum_c_index_delta=0.0,
        ranking_max_pairs=4096,
        ranking_temperature=1.0,
        ranking_warmup_epochs=1,
        fixed_epochs=2,
        resume=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated topology_v7 nested-refit experiment."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "development", "audit"],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    plan_path = args.plan.resolve()
    if args.mode == "smoke":
        result = run_smoke(
            plan_path=plan_path,
            data_dir=data_dir,
            output_root=output_root,
            device_arg=args.device,
        )
    elif args.mode == "development":
        result = run_development(
            plan_path=plan_path,
            data_dir=data_dir,
            output_root=output_root,
            device_arg=args.device,
            resume=args.resume,
        )
    else:
        result = run_audit(
            plan_path=plan_path,
            data_dir=data_dir,
            output_root=output_root,
            device_arg=args.device,
            resume=args.resume,
        )
    print(json.dumps(_as_builtin(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
