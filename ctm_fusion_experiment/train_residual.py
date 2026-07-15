from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ctm_fusion_experiment.evaluate_residual import build_residual_comparison_summary
from ctm_fusion_experiment.models.residual_ctm_fusion import ResidualCTMFusionModel
from ctm_fusion_experiment.train import (
    _copy_state_dict,
    _export_graph_embeddings,
    _mean,
    _tensor_loader,
    _train_fusion,
    _train_graph_encoder,
    resolve_device,
    set_seed,
)
from ctm_fusion_experiment.utils.data_loader import (
    FusionArraySet,
    load_graph_dataset,
    make_cv_splits,
    prepare_fusion_arrays,
)
from ctm_fusion_experiment.utils.losses import gather_stable_risk, residual_ctm_cox_loss
from ctm_fusion_experiment.utils.metrics import concordance_index
from ctm_fusion_experiment.utils.reporting import count_parameters, save_embeddings, write_csv, write_json


def _baseline_risk(
    baseline_model: torch.nn.Module,
    graph: torch.Tensor,
    clinical: torch.Tensor,
    metabolite: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return baseline_model(graph, clinical, metabolite)["risk"].detach()


def _build_residual_model(arrays: FusionArraySet, config: dict[str, Any]) -> ResidualCTMFusionModel:
    ctm = config["ctm"]
    residual = config["residual"]
    return ResidualCTMFusionModel(
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
        dropout=float(config["fusion"]["dropout"]),
        max_residual_gate=float(residual["max_residual_gate"]),
        initial_gate_logit=float(residual["initial_gate_logit"]),
    )


def _evaluate_residual(
    model: ResidualCTMFusionModel,
    baseline_model: torch.nn.Module,
    arrays: FusionArraySet,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    baseline_model.eval()
    loader = _tensor_loader(arrays, int(config["fusion"]["batch_size"]), shuffle=False)
    all_time, all_event, all_risk = [], [], []
    losses = []
    stable_ticks = []
    best_loss_ticks = []
    attention = []
    residual_gates = []
    delta_std = []
    loss_config = config["loss"]
    with torch.no_grad():
        for graph, clinical, metabolite, times, events in loader:
            graph = graph.to(device)
            clinical = clinical.to(device)
            metabolite = metabolite.to(device)
            times = times.to(device)
            events = events.to(device)
            baseline = _baseline_risk(baseline_model, graph, clinical, metabolite)
            output = model(graph, clinical, metabolite, baseline_risk=baseline, track_attention=True)
            final_risk = output["risk_per_tick"][:, -1]
            loss_result = residual_ctm_cox_loss(
                risk_scores_per_tick=output["risk_per_tick"],
                baseline_risk=baseline,
                times=times,
                events=events,
                **loss_config,
            )
            _, sample_stable_ticks = gather_stable_risk(output["risk_per_tick"])
            stable_ticks.extend(sample_stable_ticks.detach().cpu().tolist())
            best_loss_ticks.append(loss_result.best_loss_tick)
            attention.append(output["attention_weights"].detach().cpu())
            residual_gates.append(float(output["residual_gate"].detach().cpu().item()))
            delta_std.append(float(output["delta_per_tick"][:, -1].std(unbiased=False).detach().cpu().item()))
            losses.append(float(loss_result.loss.item()))
            all_time.extend(times.detach().cpu().tolist())
            all_event.extend(events.detach().cpu().tolist())
            all_risk.extend(final_risk.detach().cpu().tolist())

    details = {
        "risk": all_risk,
        "stable_ticks": stable_ticks,
        "stable_tick_histogram": dict(sorted(Counter(stable_ticks).items())),
        "best_loss_tick_by_batch": best_loss_ticks,
        "best_loss_tick_histogram_by_batch": dict(sorted(Counter(best_loss_ticks).items())),
        "mean_attention_by_tick_and_modality": torch.cat(attention, dim=0).mean(dim=(0, 2)).tolist(),
        "mean_residual_gate": float(np.mean(residual_gates)),
        "mean_final_delta_std": float(np.mean(delta_std)),
    }
    metrics = {
        "loss": _mean(losses),
        "c_index": concordance_index(all_time, all_event, all_risk),
    }
    return metrics, details


def _train_residual(
    arrays,
    baseline_model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ResidualCTMFusionModel, dict[str, Any], dict[str, Any]]:
    settings = config["fusion"]
    loss_config = config["loss"]
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad_(False)
    model = _build_residual_model(arrays.train, config).to(device)
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
        component_totals: dict[str, list[float]] = {}
        best_loss_ticks = []
        for graph, clinical, metabolite, times, events in train_loader:
            graph = graph.to(device)
            clinical = clinical.to(device)
            metabolite = metabolite.to(device)
            times = times.to(device)
            events = events.to(device)
            baseline = _baseline_risk(baseline_model, graph, clinical, metabolite)
            optimizer.zero_grad(set_to_none=True)
            output = model(graph, clinical, metabolite, baseline_risk=baseline)
            loss_result = residual_ctm_cox_loss(
                risk_scores_per_tick=output["risk_per_tick"],
                baseline_risk=baseline,
                times=times,
                events=events,
                **loss_config,
            )
            loss_result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss_result.loss.item()))
            best_loss_ticks.append(loss_result.best_loss_tick)
            for name, value in loss_result.components.items():
                component_totals.setdefault(name, []).append(float(value.item()))

        val_metrics, val_details = _evaluate_residual(model, baseline_model, arrays.val, config, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean(train_losses),
                "best_loss_tick_histogram": dict(sorted(Counter(best_loss_ticks).items())),
                "mean_residual_gate": val_details["mean_residual_gate"],
                **{f"train_{name}": _mean(values) for name, values in component_totals.items()},
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
    test_metrics, details = _evaluate_residual(model, baseline_model, arrays.test, config, device)
    return model, {
        "best_val_c_index": best_c_index,
        "training_seconds": time.perf_counter() - started,
        "parameters": count_parameters(model),
        "mean_residual_gate": details["mean_residual_gate"],
        "mean_final_delta_std": details["mean_final_delta_std"],
        "history": history,
        "test": test_metrics,
    }, details


def _prediction_rows(arrays: FusionArraySet, baseline_details: dict[str, Any], residual_details: dict[str, Any]):
    return [
        {
            "sample_id": sample_id,
            "time": float(arrays.time[index]),
            "event": float(arrays.event[index]),
            "baseline_risk": baseline_details["risk"][index],
            "residual_ctm_risk": residual_details["risk"][index],
            "residual_ctm_stable_tick": residual_details["stable_ticks"][index],
        }
        for index, sample_id in enumerate(arrays.sample_ids)
    ]


def run_residual_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
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
            "flow": "residual_ctm",
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
        residual_model, residual_summary, residual_details = _train_residual(arrays, baseline_model, config, device)
        torch.save(residual_model.state_dict(), fold_dir / "residual_ctm.pt")
        write_json(fold_dir / "residual_ctm_history.json", residual_summary["history"])
        write_json(fold_dir / "residual_ctm_analysis.json", residual_details)
        write_csv(fold_dir / "test_predictions.csv", _prediction_rows(arrays.test, baseline_details, residual_details))
        graph_summary.pop("history")
        baseline_summary.pop("history")
        residual_summary.pop("history")
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoder": graph_summary,
                "baseline": baseline_summary,
                "residual_ctm": residual_summary,
            },
        )

    return build_residual_comparison_summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/residual_ctm.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_residual_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
