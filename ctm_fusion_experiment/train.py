from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader as TensorDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GraphDataLoader

from ctm_fusion_experiment.evaluate import build_comparison_summary
from ctm_fusion_experiment.models.baseline_concat import BaselineConcatModel
from ctm_fusion_experiment.models.ctm_fusion import CTMFusionModel
from ctm_fusion_experiment.models.graph_encoder import GraphOnlyGATCoxEncoder
from ctm_fusion_experiment.plot_results import build_plots
from ctm_fusion_experiment.utils.data_loader import (
    FoldSplit,
    FusionArraySet,
    GraphDatasetBundle,
    load_graph_dataset,
    make_cv_splits,
    prepare_fusion_arrays,
    subset_graphs,
)
from ctm_fusion_experiment.utils.losses import (
    cox_partial_likelihood_loss,
    ctm_cox_loss,
    gather_stable_risk,
)
from ctm_fusion_experiment.utils.metrics import concordance_index
from ctm_fusion_experiment.utils.reporting import (
    count_parameters,
    save_embeddings,
    write_csv,
    write_json,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _copy_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _graph_loader(
    bundle: GraphDatasetBundle,
    sample_ids: tuple[str, ...],
    batch_size: int,
    shuffle: bool,
) -> GraphDataLoader:
    return GraphDataLoader(subset_graphs(bundle.graphs_by_id, sample_ids), batch_size=batch_size, shuffle=shuffle)


def _evaluate_graph_encoder(
    model: GraphOnlyGATCoxEncoder,
    loader: GraphDataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    all_time, all_event, all_risk = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            losses.append(float(cox_partial_likelihood_loss(output["risk"], batch.time, batch.event).item()))
            all_time.extend(batch.time.detach().cpu().tolist())
            all_event.extend(batch.event.detach().cpu().tolist())
            all_risk.extend(output["risk"].detach().cpu().tolist())
    return {
        "loss": _mean(losses),
        "c_index": concordance_index(all_time, all_event, all_risk),
    }


def _train_graph_encoder(
    bundle: GraphDatasetBundle,
    split: FoldSplit,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[GraphOnlyGATCoxEncoder, dict[str, Any]]:
    settings = config["graph_encoder"]
    model = GraphOnlyGATCoxEncoder(
        node_feature_dim=bundle.node_feature_dim,
        hidden_dim=int(settings["hidden_dim"]),
        heads=int(settings["heads"]),
        edge_hidden_dim=int(settings["edge_hidden_dim"]),
        embedding_dim=int(settings["embedding_dim"]),
        dropout=float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    train_loader = _graph_loader(bundle, split.train_ids, int(settings["batch_size"]), shuffle=True)
    val_loader = _graph_loader(bundle, split.val_ids, int(settings["batch_size"]), shuffle=False)
    best_c_index = float("-inf")
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = cox_partial_likelihood_loss(output["risk"], batch.time, batch.event)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = _evaluate_graph_encoder(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": _mean(losses), **val_metrics})
        if val_metrics["c_index"] > best_c_index + float(settings["min_delta"]):
            best_c_index = val_metrics["c_index"]
            best_state = _copy_state_dict(model)
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["early_stop_patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_c_index": best_c_index,
        "training_seconds": time.perf_counter() - started,
        "parameters": count_parameters(model),
        "history": history,
    }


def _export_graph_embeddings(
    model: GraphOnlyGATCoxEncoder,
    bundle: GraphDatasetBundle,
    sample_ids: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    loader = _graph_loader(bundle, sample_ids, batch_size, shuffle=False)
    embeddings = {}
    with torch.no_grad():
        for batch in loader:
            ids = [str(sample_id) for sample_id in batch.sample_id]
            output = model(batch.to(device))
            values = output["graph_embedding"].detach().cpu().numpy()
            embeddings.update({sample_id: value for sample_id, value in zip(ids, values)})
    return embeddings


def _tensor_loader(
    arrays: FusionArraySet,
    batch_size: int,
    shuffle: bool,
) -> TensorDataLoader:
    dataset = TensorDataset(
        torch.tensor(arrays.graph, dtype=torch.float32),
        torch.tensor(arrays.clinical, dtype=torch.float32),
        torch.tensor(arrays.metabolite, dtype=torch.float32),
        torch.tensor(arrays.time, dtype=torch.float32),
        torch.tensor(arrays.event, dtype=torch.float32),
    )
    return TensorDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_fusion(
    model: torch.nn.Module,
    arrays: FusionArraySet,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    loader = _tensor_loader(arrays, batch_size, shuffle=False)
    all_time, all_event, all_risk = [], [], []
    losses = []
    stable_ticks = []
    best_loss_ticks_by_batch = []
    stable_loss_ticks_by_batch = []
    attention = []
    with torch.no_grad():
        for graph, clinical, metabolite, times, events in loader:
            graph = graph.to(device)
            clinical = clinical.to(device)
            metabolite = metabolite.to(device)
            times = times.to(device)
            events = events.to(device)
            if model_name == "baseline":
                output = model(graph, clinical, metabolite)
                risk = output["risk"]
                loss = cox_partial_likelihood_loss(risk, times, events)
            else:
                output = model(graph, clinical, metabolite, track_attention=True)
                risk, batch_ticks = gather_stable_risk(output["risk_per_tick"])
                stable_ticks.extend(batch_ticks.detach().cpu().tolist())
                attention.append(output["attention_weights"].detach().cpu())
                loss_result = ctm_cox_loss(output["risk_per_tick"], times, events)
                best_loss_ticks_by_batch.append(loss_result.best_loss_tick)
                stable_loss_ticks_by_batch.append(loss_result.stable_tick)
                loss = loss_result.loss
            losses.append(float(loss.item()))
            all_time.extend(times.detach().cpu().tolist())
            all_event.extend(events.detach().cpu().tolist())
            all_risk.extend(risk.detach().cpu().tolist())

    details: dict[str, Any] = {"risk": all_risk}
    if stable_ticks:
        details["stable_ticks"] = stable_ticks
        details["stable_tick_histogram"] = dict(sorted(Counter(stable_ticks).items()))
        details["best_loss_tick_by_batch"] = best_loss_ticks_by_batch
        details["best_loss_tick_histogram_by_batch"] = dict(sorted(Counter(best_loss_ticks_by_batch).items()))
        details["stable_loss_tick_by_batch"] = stable_loss_ticks_by_batch
        stacked_attention = torch.cat(attention, dim=0)
        details["mean_attention_by_tick_and_modality"] = stacked_attention.mean(dim=(0, 2)).tolist()
    return {
        "loss": _mean(losses),
        "c_index": concordance_index(all_time, all_event, all_risk),
    }, details


def _build_fusion_model(
    model_name: str,
    arrays: FusionArraySet,
    config: dict[str, Any],
) -> torch.nn.Module:
    common = config["fusion"]
    if model_name == "baseline":
        return BaselineConcatModel(
            graph_dim=arrays.graph.shape[1],
            clinical_dim=arrays.clinical.shape[1],
            metabolite_dim=arrays.metabolite.shape[1],
            hidden_dim=int(common["baseline_hidden_dim"]),
            dropout=float(common["dropout"]),
        )
    ctm = config["ctm"]
    return CTMFusionModel(
        graph_dim=arrays.graph.shape[1],
        clinical_dim=arrays.clinical.shape[1],
        metabolite_dim=arrays.metabolite.shape[1],
        d_input=int(ctm["d_input"]),
        d_model=int(ctm["d_model"]),
        iterations=int(ctm["iterations"]),
        memory_length=int(ctm["memory_length"]),
        nlm_hidden_dim=int(ctm["nlm_hidden_dim"]),
        n_heads=int(ctm["n_heads"]),
        n_synch_action=int(ctm["n_synch_action"]),
        n_synch_out=int(ctm["n_synch_out"]),
        n_self_pairs=int(ctm["n_self_pairs"]),
        synapse_depth=int(ctm["synapse_depth"]),
        dropout=float(common["dropout"]),
    )


def _train_fusion(
    model_name: str,
    arrays,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    settings = config["fusion"]
    model = _build_fusion_model(model_name, arrays.train, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    train_loader = _tensor_loader(arrays.train, int(settings["batch_size"]), shuffle=True)
    best_c_index = float("-inf")
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        train_losses = []
        best_loss_ticks = []
        stable_loss_ticks = []
        for graph, clinical, metabolite, times, events in train_loader:
            graph = graph.to(device)
            clinical = clinical.to(device)
            metabolite = metabolite.to(device)
            times = times.to(device)
            events = events.to(device)
            optimizer.zero_grad(set_to_none=True)
            if model_name == "baseline":
                output = model(graph, clinical, metabolite)
                loss = cox_partial_likelihood_loss(output["risk"], times, events)
            else:
                output = model(graph, clinical, metabolite)
                loss_result = ctm_cox_loss(output["risk_per_tick"], times, events)
                best_loss_ticks.append(loss_result.best_loss_tick)
                stable_loss_ticks.append(loss_result.stable_tick)
                loss = loss_result.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics, _ = _evaluate_fusion(
            model,
            arrays.val,
            model_name,
            int(settings["batch_size"]),
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean(train_losses),
                "best_loss_tick_histogram": dict(sorted(Counter(best_loss_ticks).items())),
                "stable_loss_tick_histogram": dict(sorted(Counter(stable_loss_ticks).items())),
                **val_metrics,
            }
        )
        if val_metrics["c_index"] > best_c_index + float(settings["min_delta"]):
            best_c_index = val_metrics["c_index"]
            best_state = _copy_state_dict(model)
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["early_stop_patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, details = _evaluate_fusion(
        model,
        arrays.test,
        model_name,
        int(settings["batch_size"]),
        device,
    )
    return model, {
        "best_val_c_index": best_c_index,
        "training_seconds": time.perf_counter() - started,
        "parameters": count_parameters(model),
        "history": history,
        "test": test_metrics,
    }, details


def _prediction_rows(
    arrays: FusionArraySet,
    baseline_details: dict[str, Any],
    ctm_details: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "time": float(arrays.time[index]),
            "event": float(arrays.event[index]),
            "baseline_risk": baseline_details["risk"][index],
            "ctm_risk": ctm_details["risk"][index],
            "ctm_stable_tick": ctm_details["stable_ticks"][index],
        }
        for index, sample_id in enumerate(arrays.sample_ids)
    ]


def run_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = resolve_device(device_arg)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", config)
    bundle = load_graph_dataset(config, sample_limit=config.get("runtime", {}).get("sample_limit"))
    write_json(
        output_dir / "run_metadata.json",
        {
            "device": str(device),
            "sample_count": int(len(bundle.sample_table)),
            "dataset_version": "topology_v6",
            "dataset_is_synthetic_noisy_augmented": True,
        },
    )
    splits = make_cv_splits(
        bundle.sample_table,
        folds=int(config["cv"]["folds"]),
        seed=int(config["seed"]),
        val_ratio=float(config["cv"]["val_ratio"]),
    )
    max_folds = config.get("runtime", {}).get("max_folds")
    if max_folds is not None:
        splits = splits[: int(max_folds)]

    for split in splits:
        set_seed(int(config["seed"]) + split.fold)
        fold_dir = output_dir / f"fold_{split.fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / "split.json", split.__dict__)
        graph_model, graph_summary = _train_graph_encoder(bundle, split, config, device)
        torch.save(graph_model.state_dict(), fold_dir / "graph_encoder.pt")
        export_ids = tuple(sorted(set(split.train_ids + split.val_ids + split.test_ids)))
        embeddings = _export_graph_embeddings(
            graph_model,
            bundle,
            export_ids,
            int(config["graph_encoder"]["batch_size"]),
            device,
        )
        save_embeddings(fold_dir / "frozen_graph_embeddings.npz", embeddings)
        arrays = prepare_fusion_arrays(
            sample_table=bundle.sample_table,
            split=split,
            graph_embeddings=embeddings,
            clinical_columns=config["model"]["clinical_columns"],
            metabolite_columns=config["model"]["metabolite_columns"],
        )

        set_seed(int(config["seed"]) + split.fold)
        baseline_model, baseline_summary, baseline_details = _train_fusion("baseline", arrays, config, device)
        torch.save(baseline_model.state_dict(), fold_dir / "baseline_concat.pt")
        write_json(fold_dir / "baseline_history.json", baseline_summary["history"])

        set_seed(int(config["seed"]) + split.fold)
        ctm_model, ctm_summary, ctm_details = _train_fusion("ctm", arrays, config, device)
        torch.save(ctm_model.state_dict(), fold_dir / "ctm_fusion.pt")
        write_json(fold_dir / "ctm_history.json", ctm_summary["history"])
        write_json(fold_dir / "ctm_analysis.json", ctm_details)
        write_csv(
            fold_dir / "test_predictions.csv",
            _prediction_rows(arrays.test, baseline_details, ctm_details),
        )
        graph_summary.pop("history")
        baseline_summary.pop("history")
        ctm_summary.pop("history")
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoder": graph_summary,
                "baseline": baseline_summary,
                "ctm": ctm_summary,
            },
        )

    summary = build_comparison_summary(output_dir)
    summary["plots"] = build_plots(output_dir)
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/ctm_fusion.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
