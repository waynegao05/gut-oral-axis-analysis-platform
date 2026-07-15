from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from research.data import build_dataset_from_csv, set_seed
from research.ensemble_stack_v2 import _apply_weights, _cohort_cox_loss, _fit_cox_risk_scale
from research.ensemble_v2 import build_model, load_checkpoints
from research.losses import cox_ph_loss, pairwise_ranking_loss
from research.metrics import concordance_index
from research.train_v2 import resolve_device


@dataclass(frozen=True)
class CachedHeadSplit:
    sample_ids: list[str]
    time: np.ndarray
    event: np.ndarray
    graph_embedding: np.ndarray
    clinical: np.ndarray
    metabolites: np.ndarray
    latent: np.ndarray
    base_risk: np.ndarray


@dataclass(frozen=True)
class RefinedPrediction:
    mode: str
    best_epoch: int
    train_risk: np.ndarray
    val_risk: np.ndarray
    test_risk: np.ndarray
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float | int]]


@dataclass(frozen=True)
class ValidationCandidate:
    name: str
    val_c_index: float
    calibrated_val_cox_loss: float


class CachedCoxHeadRefiner(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        fusion: nn.Module,
        risk_head: nn.Module,
    ) -> None:
        super().__init__()
        if mode not in {"head_only", "fusion_head"}:
            raise ValueError(f"Unsupported refinement mode: {mode}")
        self.mode = mode
        self.fusion = fusion
        self.risk_head = risk_head

    def forward(
        self,
        *,
        graph_embedding: torch.Tensor,
        clinical: torch.Tensor,
        metabolites: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "head_only":
            refined_latent = latent
        else:
            refined_latent = self.fusion(torch.cat([graph_embedding, clinical, metabolites], dim=1))
        return self.risk_head(refined_latent).squeeze(-1)


def run_full_risk_head_refinement(
    *,
    config_path: str,
    checkpoint_glob: str,
    split_seed: int,
    output_path: str | Path,
    device_arg: str = "cuda",
    checkpoint_names: Sequence[str] = (),
    modes: Sequence[str] = ("head_only", "fusion_head"),
    epochs: int = 160,
    patience: int = 25,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.02,
    distillation_weight: float = 0.10,
    parameter_anchor_weight: float = 0.01,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
    grad_clip_norm: float = 1.0,
    softmax_temperature: float = 0.003,
    min_validation_delta: float = 0.0003,
    max_validation_cox_loss_increase: float = 0.0,
    refiner_seed: int = 42,
    cache_batch_size: int | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = _filter_checkpoints(load_checkpoints(checkpoint_glob), checkpoint_names)
    selected_modes = _validate_modes(modes)
    device = resolve_device(device_arg)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / f"{output.stem}_cache"
    artifact_dir = output.parent / f"{output.stem}_artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(config, split_seed=split_seed)
    split_sets = {
        "train": dataset.train_set,
        "val": dataset.val_set,
        "test": dataset.test_set,
    }
    effective_cache_batch_size = int(cache_batch_size or config["train"]["batch_size"])

    baseline_matrices: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    refined_matrices: dict[str, dict[str, list[np.ndarray]]] = {
        mode: {"train": [], "val": [], "test": []} for mode in selected_modes
    }
    member_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    aligned: dict[str, CachedHeadSplit] = {}

    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        model = build_model(config, dataset, device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        fusion_template = copy.deepcopy(model.fusion).cpu()
        risk_head_template = copy.deepcopy(model.risk_head).cpu()

        cached_splits: dict[str, CachedHeadSplit] = {}
        for split_name, split_items in split_sets.items():
            cache_path = cache_dir / f"{checkpoint_path.parent.name}_{split_name}.npz"
            cached_splits[split_name] = _load_or_create_cache(
                cache_path=cache_path,
                checkpoint_path=checkpoint_path,
                split_seed=split_seed,
                split_name=split_name,
                model=model,
                split_items=split_items,
                batch_size=effective_cache_batch_size,
                device=device,
            )
            if split_name not in aligned:
                aligned[split_name] = cached_splits[split_name]
            else:
                _assert_aligned(aligned[split_name], cached_splits[split_name], split_name)
            baseline_matrices[split_name].append(cached_splits[split_name].base_risk)

        member_row: dict[str, Any] = {
            "checkpoint": str(checkpoint_path.as_posix()),
            "checkpoint_name": checkpoint_path.parent.name,
            "baseline_validation_c_index": concordance_index(
                cached_splits["val"].time,
                cached_splits["val"].event,
                cached_splits["val"].base_risk,
            ),
            "refinements": {},
        }
        for mode_index, mode in enumerate(selected_modes):
            prediction = fit_cached_refiner(
                mode=mode,
                fusion_template=fusion_template,
                risk_head_template=risk_head_template,
                train_split=cached_splits["train"],
                val_split=cached_splits["val"],
                test_split=cached_splits["test"],
                device=device,
                seed=int(refiner_seed) + checkpoint_index * 100 + mode_index,
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                distillation_weight=distillation_weight,
                parameter_anchor_weight=parameter_anchor_weight,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
                grad_clip_norm=grad_clip_norm,
            )
            refined_matrices[mode]["train"].append(prediction.train_risk)
            refined_matrices[mode]["val"].append(prediction.val_risk)
            refined_matrices[mode]["test"].append(prediction.test_risk)
            refined_val_c_index = concordance_index(
                cached_splits["val"].time,
                cached_splits["val"].event,
                prediction.val_risk,
            )
            member_row["refinements"][mode] = {
                "best_epoch": prediction.best_epoch,
                "validation_c_index": refined_val_c_index,
                "validation_delta": refined_val_c_index - member_row["baseline_validation_c_index"],
                "history_epochs": len(prediction.history),
            }
            artifact_path = artifact_dir / f"{checkpoint_path.parent.name}_{mode}.pt"
            history_path = artifact_dir / f"{checkpoint_path.parent.name}_{mode}_history.json"
            history_path.write_text(json.dumps(prediction.history, indent=2), encoding="utf-8")
            member_row["refinements"][mode]["history_path"] = str(history_path.as_posix())
            torch.save(
                {
                    "format_version": 1,
                    "mode": mode,
                    "source_checkpoint": str(checkpoint_path.as_posix()),
                    "split_seed": int(split_seed),
                    "state_dict": prediction.state_dict,
                    "best_epoch": int(prediction.best_epoch),
                    "training": {
                        "epochs": int(epochs),
                        "patience": int(patience),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "distillation_weight": float(distillation_weight),
                        "parameter_anchor_weight": float(parameter_anchor_weight),
                        "ranking_weight": float(ranking_weight),
                        "ranking_margin": float(ranking_margin),
                    },
                },
                artifact_path,
            )
            artifacts.append(
                {
                    "checkpoint_name": checkpoint_path.parent.name,
                    "mode": mode,
                    "path": str(artifact_path.as_posix()),
                }
            )
        member_rows.append(member_row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_arrays = {name: np.asarray(rows, dtype=float) for name, rows in baseline_matrices.items()}
    reference_risks, reference_weights, reference_scaler = _softmax_ensemble_risks(
        train_matrix=baseline_arrays["train"],
        val_matrix=baseline_arrays["val"],
        test_matrix=baseline_arrays["test"],
        val_time=aligned["val"].time,
        val_event=aligned["val"].event,
        temperature=softmax_temperature,
    )
    candidates: dict[str, dict[str, Any]] = {
        "reference": {
            "risks": reference_risks,
            "weights": reference_weights,
            "scaler": reference_scaler,
            "metadata": {"mode": "reference"},
        }
    }
    for mode in selected_modes:
        mode_arrays = {
            name: np.asarray(rows, dtype=float) for name, rows in refined_matrices[mode].items()
        }
        risks, weights, scaler = _softmax_ensemble_risks(
            train_matrix=mode_arrays["train"],
            val_matrix=mode_arrays["val"],
            test_matrix=mode_arrays["test"],
            val_time=aligned["val"].time,
            val_event=aligned["val"].event,
            temperature=softmax_temperature,
        )
        candidates[mode] = {
            "risks": risks,
            "weights": weights,
            "scaler": scaler,
            "metadata": {"mode": mode},
        }

    validation_rows: list[dict[str, Any]] = []
    validation_candidates: list[ValidationCandidate] = []
    for name, candidate in candidates.items():
        val_risk = candidate["risks"]["val"]
        calibration = _fit_cox_risk_scale(val_risk, aligned["val"].time, aligned["val"].event)
        row = {
            "name": name,
            "validation_c_index": concordance_index(
                aligned["val"].time,
                aligned["val"].event,
                val_risk,
            ),
            "validation_cox_scale_calibration": calibration,
            "weights": [float(value) for value in candidate["weights"]],
        }
        validation_rows.append(row)
        validation_candidates.append(
            ValidationCandidate(
                name=name,
                val_c_index=float(row["validation_c_index"]),
                calibrated_val_cox_loss=float(calibration["calibrated_validation_cox_loss"]),
            )
        )

    selected_name = select_validation_candidate(
        validation_candidates,
        reference_name="reference",
        min_validation_delta=min_validation_delta,
        max_validation_cox_loss_increase=max_validation_cox_loss_increase,
    )
    reference_summary = _test_summary(
        name="reference",
        candidate=candidates["reference"],
        validation_rows=validation_rows,
        val_split=aligned["val"],
        test_split=aligned["test"],
    )
    selected_summary = _test_summary(
        name=selected_name,
        candidate=candidates[selected_name],
        validation_rows=validation_rows,
        val_split=aligned["val"],
        test_split=aligned["test"],
    )
    selected_summary["test_c_index_delta"] = (
        selected_summary["test_c_index"] - reference_summary["test_c_index"]
    )
    selected_summary["calibrated_test_cox_loss_delta"] = (
        selected_summary["calibrated_test_cox_loss"] - reference_summary["calibrated_test_cox_loss"]
    )

    predictions_path = output.with_name(f"{output.stem}_selected_predictions.npz")
    if selected_name == "reference":
        selected_member_arrays = baseline_arrays
    else:
        selected_member_arrays = {
            name: np.asarray(rows, dtype=float)
            for name, rows in refined_matrices[selected_name].items()
        }
    np.savez_compressed(
        predictions_path,
        checkpoint_names=np.asarray([path.parent.name for path in checkpoints]),
        selected_name=np.asarray(selected_name),
        train_sample_ids=np.asarray(aligned["train"].sample_ids),
        train_time=aligned["train"].time,
        train_event=aligned["train"].event,
        train_reference_risk=candidates["reference"]["risks"]["train"],
        train_selected_risk=candidates[selected_name]["risks"]["train"],
        train_reference_member_risk_matrix=baseline_arrays["train"],
        train_selected_member_risk_matrix=selected_member_arrays["train"],
        val_sample_ids=np.asarray(aligned["val"].sample_ids),
        val_time=aligned["val"].time,
        val_event=aligned["val"].event,
        val_reference_risk=candidates["reference"]["risks"]["val"],
        val_selected_risk=candidates[selected_name]["risks"]["val"],
        val_reference_member_risk_matrix=baseline_arrays["val"],
        val_selected_member_risk_matrix=selected_member_arrays["val"],
        test_sample_ids=np.asarray(aligned["test"].sample_ids),
        test_time=aligned["test"].time,
        test_event=aligned["test"].event,
        test_reference_risk=candidates["reference"]["risks"]["test"],
        test_selected_risk=candidates[selected_name]["risks"]["test"],
        test_reference_member_risk_matrix=baseline_arrays["test"],
        test_selected_member_risk_matrix=selected_member_arrays["test"],
    )

    result = {
        "config_path": str(Path(config_path).as_posix()),
        "checkpoint_glob": checkpoint_glob,
        "checkpoints": [str(path.as_posix()) for path in checkpoints],
        "checkpoint_names": [path.parent.name for path in checkpoints],
        "split_seed": int(split_seed),
        "device": str(device),
        "modes": selected_modes,
        "selection_policy": {
            "reference_name": "reference",
            "min_validation_delta": float(min_validation_delta),
            "max_validation_cox_loss_increase": float(max_validation_cox_loss_increase),
            "test_metrics_used_for_selection": False,
        },
        "training": {
            "epochs": int(epochs),
            "patience": int(patience),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "distillation_weight": float(distillation_weight),
            "parameter_anchor_weight": float(parameter_anchor_weight),
            "ranking_weight": float(ranking_weight),
            "ranking_margin": float(ranking_margin),
            "softmax_temperature": float(softmax_temperature),
        },
        "cache_dir": str(cache_dir.as_posix()),
        "artifacts": artifacts,
        "member_validation": member_rows,
        "candidate_validation": validation_rows,
        "reference": reference_summary,
        "selected": selected_summary,
        "selected_predictions": str(predictions_path.as_posix()),
        "interpretation": (
            "Frozen GNN representations are cached once. Only copied fusion/Cox heads are refined with the full "
            "training risk set. Validation c-index and calibrated validation Cox loss control fallback; test metrics "
            "are computed only after selection."
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def fit_cached_refiner(
    *,
    mode: str,
    fusion_template: nn.Module,
    risk_head_template: nn.Module,
    train_split: CachedHeadSplit,
    val_split: CachedHeadSplit,
    test_split: CachedHeadSplit,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    distillation_weight: float,
    parameter_anchor_weight: float,
    ranking_weight: float,
    ranking_margin: float,
    grad_clip_norm: float,
) -> RefinedPrediction:
    _validate_cached_split(train_split)
    _validate_cached_split(val_split)
    _validate_cached_split(test_split)
    set_seed(seed)
    model = CachedCoxHeadRefiner(
        mode=mode,
        fusion=copy.deepcopy(fusion_template),
        risk_head=copy.deepcopy(risk_head_template),
    ).to(device)
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_tensors = _split_tensors(train_split, device)
    val_tensors = _split_tensors(val_split, device)
    test_tensors = _split_tensors(test_split, device)
    train_time = torch.as_tensor(train_split.time, dtype=torch.float32, device=device)
    train_event = torch.as_tensor(train_split.event, dtype=torch.float32, device=device)
    train_base_risk = torch.as_tensor(train_split.base_risk, dtype=torch.float32, device=device)

    best_state = _cpu_state_dict(model)
    best_epoch = 0
    best_val_risk = _predict_refiner(model, val_tensors)
    best_val_c_index = concordance_index(val_split.time, val_split.event, best_val_risk)
    best_val_cox_loss = _cohort_cox_loss(best_val_risk, val_split.time, val_split.event)
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(**train_tensors)
        cox_loss = cox_ph_loss(risk, train_time, train_event)
        distillation = F.mse_loss(risk, train_base_risk)
        parameter_anchor = _parameter_anchor_loss(model, initial_parameters)
        if ranking_weight > 0.0:
            ranking_loss = pairwise_ranking_loss(
                risk=risk,
                time=train_time,
                event=train_event,
                margin=float(ranking_margin),
            )
        else:
            ranking_loss = torch.zeros((), dtype=risk.dtype, device=risk.device)
        loss = (
            cox_loss
            + float(distillation_weight) * distillation
            + float(parameter_anchor_weight) * parameter_anchor
            + float(ranking_weight) * ranking_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        val_risk = _predict_refiner(model, val_tensors)
        val_c_index = concordance_index(val_split.time, val_split.event, val_risk)
        val_cox_loss = _cohort_cox_loss(val_risk, val_split.time, val_split.event)
        history.append(
            {
                "epoch": int(epoch),
                "train_total_loss": float(loss.detach().item()),
                "train_cox_loss": float(cox_loss.detach().item()),
                "train_distillation_loss": float(distillation.detach().item()),
                "train_parameter_anchor_loss": float(parameter_anchor.detach().item()),
                "train_ranking_loss": float(ranking_loss.detach().item()),
                "validation_c_index": float(val_c_index),
                "validation_cox_loss": float(val_cox_loss),
            }
        )
        improved = val_c_index > best_val_c_index + 1e-12 or (
            abs(val_c_index - best_val_c_index) <= 1e-12 and val_cox_loss < best_val_cox_loss - 1e-8
        )
        if improved:
            best_state = _cpu_state_dict(model)
            best_epoch = int(epoch)
            best_val_c_index = float(val_c_index)
            best_val_cox_loss = float(val_cox_loss)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break

    model.load_state_dict(best_state)
    model.to(device)
    return RefinedPrediction(
        mode=mode,
        best_epoch=best_epoch,
        train_risk=_predict_refiner(model, train_tensors),
        val_risk=_predict_refiner(model, val_tensors),
        test_risk=_predict_refiner(model, test_tensors),
        state_dict=best_state,
        history=history,
    )


def select_validation_candidate(
    candidates: Sequence[ValidationCandidate],
    *,
    reference_name: str,
    min_validation_delta: float,
    max_validation_cox_loss_increase: float,
) -> str:
    by_name = {candidate.name: candidate for candidate in candidates}
    if reference_name not in by_name:
        raise ValueError(f"Missing reference candidate: {reference_name}")
    reference = by_name[reference_name]
    eligible = [reference]
    for candidate in candidates:
        if candidate.name == reference_name:
            continue
        c_index_ok = candidate.val_c_index - reference.val_c_index >= float(min_validation_delta)
        cox_ok = (
            candidate.calibrated_val_cox_loss - reference.calibrated_val_cox_loss
            <= float(max_validation_cox_loss_increase)
        )
        if c_index_ok and cox_ok:
            eligible.append(candidate)
    selected = max(
        eligible,
        key=lambda candidate: (candidate.val_c_index, -candidate.calibrated_val_cox_loss),
    )
    return selected.name


def _build_dataset(config: dict[str, Any], *, split_seed: int):
    graph_preprocess = config.get("graph_preprocess", {})
    tabular_preprocess = config.get("tabular_preprocess", {})
    return build_dataset_from_csv(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
        node_feature_columns=config["model"]["node_feature_columns"],
        clinical_columns=config["model"]["clinical_columns"],
        metabolite_columns=config["model"]["metabolite_columns"],
        seed=config["seed"],
        split_seed=split_seed,
        keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
        min_edge_weight=graph_preprocess.get("min_edge_weight"),
        standardize_tabular=bool(tabular_preprocess.get("standardize", False)),
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
    )


def _filter_checkpoints(checkpoints: Sequence[Path], checkpoint_names: Sequence[str]) -> list[Path]:
    if not checkpoint_names:
        return list(checkpoints)
    requested = {str(name) for name in checkpoint_names}
    selected = [path for path in checkpoints if path.parent.name in requested]
    missing = requested.difference(path.parent.name for path in selected)
    if missing:
        raise ValueError(f"Requested checkpoint names were not found: {sorted(missing)}")
    return selected


def _validate_modes(modes: Sequence[str]) -> list[str]:
    selected = list(dict.fromkeys(str(mode) for mode in modes))
    if not selected:
        raise ValueError("At least one refinement mode is required.")
    unsupported = set(selected).difference({"head_only", "fusion_head"})
    if unsupported:
        raise ValueError(f"Unsupported refinement modes: {sorted(unsupported)}")
    return selected


def _load_or_create_cache(
    *,
    cache_path: Path,
    checkpoint_path: Path,
    split_seed: int,
    split_name: str,
    model: nn.Module,
    split_items: Sequence[Any],
    batch_size: int,
    device: torch.device,
) -> CachedHeadSplit:
    checkpoint_stat = checkpoint_path.stat()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata_matches = (
                str(cached["source_checkpoint"].item()) == str(checkpoint_path.resolve())
                and int(cached["source_size"].item()) == int(checkpoint_stat.st_size)
                and int(cached["source_mtime_ns"].item()) == int(checkpoint_stat.st_mtime_ns)
                and int(cached["split_seed"].item()) == int(split_seed)
                and str(cached["split_name"].item()) == str(split_name)
            )
            if metadata_matches:
                return _cached_split_from_npz(cached)

    split = _cache_model_split(model, split_items, batch_size=batch_size, device=device)
    np.savez_compressed(
        cache_path,
        source_checkpoint=np.asarray(str(checkpoint_path.resolve())),
        source_size=np.asarray(int(checkpoint_stat.st_size), dtype=np.int64),
        source_mtime_ns=np.asarray(int(checkpoint_stat.st_mtime_ns), dtype=np.int64),
        split_seed=np.asarray(int(split_seed), dtype=np.int64),
        split_name=np.asarray(str(split_name)),
        sample_ids=np.asarray(split.sample_ids),
        time=split.time,
        event=split.event,
        graph_embedding=split.graph_embedding,
        clinical=split.clinical,
        metabolites=split.metabolites,
        latent=split.latent,
        base_risk=split.base_risk,
    )
    return split


def _cached_split_from_npz(cached: Any) -> CachedHeadSplit:
    split = CachedHeadSplit(
        sample_ids=[str(value) for value in cached["sample_ids"].tolist()],
        time=np.asarray(cached["time"], dtype=float),
        event=np.asarray(cached["event"], dtype=float),
        graph_embedding=np.asarray(cached["graph_embedding"], dtype=np.float32),
        clinical=np.asarray(cached["clinical"], dtype=np.float32),
        metabolites=np.asarray(cached["metabolites"], dtype=np.float32),
        latent=np.asarray(cached["latent"], dtype=np.float32),
        base_risk=np.asarray(cached["base_risk"], dtype=float),
    )
    _validate_cached_split(split)
    return split


def _cache_model_split(
    model: nn.Module,
    split_items: Sequence[Any],
    *,
    batch_size: int,
    device: torch.device,
) -> CachedHeadSplit:
    loader = DataLoader(split_items, batch_size=int(batch_size), shuffle=False)
    sample_ids: list[str] = []
    arrays: dict[str, list[np.ndarray]] = {
        "time": [],
        "event": [],
        "graph_embedding": [],
        "clinical": [],
        "metabolites": [],
        "latent": [],
        "base_risk": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, compute_contrastive=False)
            current_batch_size = len(batch.sample_id)
            sample_ids.extend(str(value) for value in batch.sample_id)
            arrays["time"].append(batch.time.detach().cpu().numpy().reshape(-1))
            arrays["event"].append(batch.event.detach().cpu().numpy().reshape(-1))
            arrays["graph_embedding"].append(output["graph_embedding"].detach().cpu().numpy())
            arrays["clinical"].append(
                batch.clinical.view(current_batch_size, -1).detach().cpu().numpy()
            )
            arrays["metabolites"].append(
                batch.metabolites.view(current_batch_size, -1).detach().cpu().numpy()
            )
            arrays["latent"].append(output["latent"].detach().cpu().numpy())
            arrays["base_risk"].append(output["risk"].detach().cpu().numpy().reshape(-1))
    split = CachedHeadSplit(
        sample_ids=sample_ids,
        time=np.concatenate(arrays["time"]).astype(float),
        event=np.concatenate(arrays["event"]).astype(float),
        graph_embedding=np.concatenate(arrays["graph_embedding"]).astype(np.float32),
        clinical=np.concatenate(arrays["clinical"]).astype(np.float32),
        metabolites=np.concatenate(arrays["metabolites"]).astype(np.float32),
        latent=np.concatenate(arrays["latent"]).astype(np.float32),
        base_risk=np.concatenate(arrays["base_risk"]).astype(float),
    )
    _validate_cached_split(split)
    return split


def _validate_cached_split(split: CachedHeadSplit) -> None:
    size = len(split.sample_ids)
    arrays = {
        "time": split.time,
        "event": split.event,
        "graph_embedding": split.graph_embedding,
        "clinical": split.clinical,
        "metabolites": split.metabolites,
        "latent": split.latent,
        "base_risk": split.base_risk,
    }
    for name, values in arrays.items():
        if values.shape[0] != size:
            raise ValueError(f"Cached split field {name} has {values.shape[0]} rows; expected {size}.")
        if not np.isfinite(values).all():
            raise ValueError(f"Cached split field {name} contains non-finite values.")


def _assert_aligned(reference: CachedHeadSplit, candidate: CachedHeadSplit, split_name: str) -> None:
    if reference.sample_ids != candidate.sample_ids:
        raise RuntimeError(f"Cached checkpoint sample IDs are not aligned for split {split_name}.")
    if not np.array_equal(reference.time, candidate.time) or not np.array_equal(reference.event, candidate.event):
        raise RuntimeError(f"Cached checkpoint outcomes are not aligned for split {split_name}.")


def _split_tensors(split: CachedHeadSplit, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "graph_embedding": torch.as_tensor(split.graph_embedding, dtype=torch.float32, device=device),
        "clinical": torch.as_tensor(split.clinical, dtype=torch.float32, device=device),
        "metabolites": torch.as_tensor(split.metabolites, dtype=torch.float32, device=device),
        "latent": torch.as_tensor(split.latent, dtype=torch.float32, device=device),
    }


def _predict_refiner(model: CachedCoxHeadRefiner, tensors: dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(**tensors).detach().cpu().numpy().astype(float)


def _parameter_anchor_loss(
    model: nn.Module,
    initial_parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = [
        torch.mean((parameter - initial_parameters[name]) ** 2)
        for name, parameter in model.named_parameters()
    ]
    if not terms:
        raise ValueError("Refiner has no trainable parameters.")
    return torch.stack(terms).mean()


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _softmax_ensemble_risks(
    *,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    test_matrix: np.ndarray,
    val_time: np.ndarray,
    val_event: np.ndarray,
    temperature: float,
) -> tuple[dict[str, np.ndarray], list[float], dict[str, list[float]]]:
    train_scaled, val_scaled, test_scaled, scaler = _standardize_members_by_validation(
        train_matrix,
        val_matrix,
        test_matrix,
    )
    member_val_c_indices = [
        concordance_index(val_time, val_event, val_matrix[index])
        for index in range(val_matrix.shape[0])
    ]
    weights = _validation_softmax_weights(member_val_c_indices, temperature=temperature)
    return (
        {
            "train": _apply_weights(train_scaled, weights),
            "val": _apply_weights(val_scaled, weights),
            "test": _apply_weights(test_scaled, weights),
        },
        weights,
        scaler,
    )


def _standardize_members_by_validation(
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    train = np.asarray(train_matrix, dtype=float)
    val = np.asarray(val_matrix, dtype=float)
    test = np.asarray(test_matrix, dtype=float)
    if train.ndim != 2 or val.ndim != 2 or test.ndim != 2:
        raise ValueError("Risk matrices must be two-dimensional.")
    if not (train.shape[0] == val.shape[0] == test.shape[0]):
        raise ValueError("Risk matrices must contain the same number of members.")
    means = val.mean(axis=1, keepdims=True)
    stds = np.maximum(val.std(axis=1, keepdims=True), 1e-6)
    return (
        (train - means) / stds,
        (val - means) / stds,
        (test - means) / stds,
        {
            "risk_means": means.squeeze(axis=1).astype(float).tolist(),
            "risk_stds": stds.squeeze(axis=1).astype(float).tolist(),
        },
    )


def _validation_softmax_weights(
    member_val_c_indices: Sequence[float],
    *,
    temperature: float,
) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scores = np.asarray(member_val_c_indices, dtype=float)
    if scores.size == 0:
        raise ValueError("At least one member validation c-index is required.")
    scaled = np.exp((scores - scores.max()) / float(temperature))
    return (scaled / scaled.sum()).astype(float).tolist()


def _test_summary(
    *,
    name: str,
    candidate: dict[str, Any],
    validation_rows: Sequence[dict[str, Any]],
    val_split: CachedHeadSplit,
    test_split: CachedHeadSplit,
) -> dict[str, Any]:
    validation_row = next(row for row in validation_rows if row["name"] == name)
    calibration = validation_row["validation_cox_scale_calibration"]
    test_risk = np.asarray(candidate["risks"]["test"], dtype=float)
    scale = float(calibration["scale"])
    return {
        "name": name,
        "validation_c_index": float(validation_row["validation_c_index"]),
        "validation_cox_scale_calibration": calibration,
        "test_c_index": concordance_index(test_split.time, test_split.event, test_risk),
        "test_cox_loss": _cohort_cox_loss(test_risk, test_split.time, test_split.event),
        "calibrated_test_cox_loss": _cohort_cox_loss(
            test_risk * scale,
            test_split.time,
            test_split.event,
        ),
        "weights": [float(value) for value in candidate["weights"]],
        "calibration_fit_split": "validation",
        "test_metrics_used_for_selection": False,
    }


def _parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--checkpoint-names", default="")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--modes", default="head_only,fusion_head")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--parameter-anchor-weight", type=float, default=0.01)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.0)
    parser.add_argument("--softmax-temperature", type=float, default=0.003)
    parser.add_argument("--min-validation-delta", type=float, default=0.0003)
    parser.add_argument("--max-validation-cox-loss-increase", type=float, default=0.0)
    parser.add_argument("--refiner-seed", type=int, default=42)
    parser.add_argument("--cache-batch-size", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/full_risk_head_refiner_v2/summary.json",
    )
    args = parser.parse_args()
    result = run_full_risk_head_refinement(
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        checkpoint_names=_parse_list(args.checkpoint_names),
        split_seed=args.split_seed,
        output_path=args.output,
        device_arg=args.device,
        modes=_parse_list(args.modes),
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        distillation_weight=args.distillation_weight,
        parameter_anchor_weight=args.parameter_anchor_weight,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        softmax_temperature=args.softmax_temperature,
        min_validation_delta=args.min_validation_delta,
        max_validation_cox_loss_increase=args.max_validation_cox_loss_increase,
        refiner_seed=args.refiner_seed,
        cache_batch_size=args.cache_batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
