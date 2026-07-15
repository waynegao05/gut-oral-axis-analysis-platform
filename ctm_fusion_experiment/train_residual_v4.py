from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from ctm_fusion_experiment.evaluate_residual_v4 import build_residual_v4_comparison_summary
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
from ctm_fusion_experiment.train_residual import _baseline_risk, _build_residual_model, _evaluate_residual
from ctm_fusion_experiment.train_residual_v2 import _collect_baseline_and_delta
from ctm_fusion_experiment.utils.cindex_ensemble_selection import (
    apply_ensemble_weights,
    choose_cindex_ensemble,
)
from ctm_fusion_experiment.utils.data_loader import (
    FusionArraySet,
    load_graph_dataset,
    make_cv_splits,
    prepare_fusion_arrays,
)
from ctm_fusion_experiment.utils.losses import (
    baseline_discordant_pairwise_loss,
    cox_partial_likelihood_loss,
    pairwise_ranking_loss,
    residual_ctm_cox_loss,
)
from ctm_fusion_experiment.utils.metrics import concordance_index
from ctm_fusion_experiment.utils.pair_diagnostics import pairwise_cindex_diagnostics
from ctm_fusion_experiment.utils.reporting import count_parameters, save_embeddings, write_csv, write_json


def _residual_loss_kwargs(config: dict[str, Any]) -> dict[str, float]:
    allowed = {
        "mean_weight",
        "final_weight",
        "best_weight",
        "distillation_weight",
        "separation_weight",
        "min_risk_std",
    }
    return {key: float(value) for key, value in config["loss"].items() if key in allowed}


def _residual_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    eval_config = dict(config)
    eval_config["loss"] = _residual_loss_kwargs(config)
    return eval_config


def _train_residual_v4(
    arrays,
    baseline_model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ResidualCTMFusionModel, dict[str, Any], dict[str, Any]]:
    settings = config["fusion"]
    loss_config = config["loss"]
    residual_loss_kwargs = _residual_loss_kwargs(config)
    eval_config = _residual_eval_config(config)
    ranking_weight = float(loss_config.get("pairwise_ranking_weight", 0.0))
    ranking_margin = float(loss_config.get("pairwise_margin", 0.0))
    hard_pair_weight = float(loss_config.get("hard_pair_weight", 0.0))
    hard_pair_margin = float(loss_config.get("hard_pair_margin", ranking_margin))
    delta_l2_weight = float(loss_config.get("delta_l2_weight", 0.0))

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
                **residual_loss_kwargs,
            )
            ranking = pairwise_ranking_loss(
                output["risk_per_tick"][:, -1],
                times,
                events,
                margin=ranking_margin,
            )
            hard_pair = baseline_discordant_pairwise_loss(
                output["risk_per_tick"][:, -1],
                baseline,
                times,
                events,
                margin=hard_pair_margin,
            )
            delta_l2 = torch.mean((output["risk_per_tick"][:, -1] - baseline.detach()) ** 2)
            loss = (
                loss_result.loss
                + ranking_weight * ranking
                + hard_pair_weight * hard_pair
                + delta_l2_weight * delta_l2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.item()))
            best_loss_ticks.append(loss_result.best_loss_tick)
            component_totals.setdefault("pairwise_ranking", []).append(float(ranking.detach().cpu().item()))
            component_totals.setdefault("hard_pair_ranking", []).append(float(hard_pair.detach().cpu().item()))
            component_totals.setdefault("delta_l2", []).append(float(delta_l2.detach().cpu().item()))
            for name, value in loss_result.components.items():
                component_totals.setdefault(name, []).append(float(value.item()))

        val_metrics, val_details = _evaluate_residual(model, baseline_model, arrays.val, eval_config, device)
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
    test_metrics, details = _evaluate_residual(model, baseline_model, arrays.test, eval_config, device)
    return model, {
        "best_val_c_index": best_c_index,
        "training_seconds": time.perf_counter() - started,
        "parameters": count_parameters(model),
        "mean_residual_gate": details["mean_residual_gate"],
        "mean_final_delta_std": details["mean_final_delta_std"],
        "history": history,
        "test": test_metrics,
    }, details


def _evaluate_selected_ensemble(
    collected: dict[str, torch.Tensor],
    deltas: list[torch.Tensor],
    weights: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    risk = apply_ensemble_weights(collected["baseline"], deltas, weights)
    loss = cox_partial_likelihood_loss(risk, collected["time"], collected["event"])
    c_index = concordance_index(
        collected["time"].numpy(),
        collected["event"].numpy(),
        risk.detach().cpu().numpy(),
    )
    diagnostics = pairwise_cindex_diagnostics(
        collected["time"].numpy(),
        collected["event"].numpy(),
        collected["baseline"].numpy(),
        risk.detach().cpu().numpy(),
    )
    return {
        "loss": float(loss.item()),
        "c_index": float(c_index),
    }, {
        "risk": risk.detach().cpu().tolist(),
        "baseline_risk": collected["baseline"].tolist(),
        "pair_diagnostics": diagnostics,
    }


def _prediction_rows(
    arrays: FusionArraySet,
    selected_details: dict[str, Any],
    seed_collections: list[dict[str, torch.Tensor]],
    selection_weights: list[float],
    seed_values: list[int],
) -> list[dict[str, Any]]:
    rows = []
    seed_risks = [
        (collection["baseline"] + collection["delta"]).detach().cpu().tolist()
        for collection in seed_collections
    ]
    for index, sample_id in enumerate(arrays.sample_ids):
        row = {
            "sample_id": sample_id,
            "time": float(arrays.time[index]),
            "event": float(arrays.event[index]),
            "baseline_risk": selected_details["baseline_risk"][index],
            "selected_risk": selected_details["risk"][index],
        }
        for seed_position, seed_value in enumerate(seed_values):
            row[f"seed_{seed_value}_risk"] = seed_risks[seed_position][index]
            row[f"seed_{seed_value}_weight"] = selection_weights[seed_position]
        rows.append(row)
    return rows


def run_residual_v4_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = resolve_device(device_arg)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", config)
    bundle = load_graph_dataset(config, sample_limit=config.get("runtime", {}).get("sample_limit"))
    seed_values = [int(seed) for seed in config["ensemble"]["residual_seeds"]]
    write_json(
        output_dir / "run_metadata.json",
        {
            "device": str(device),
            "sample_count": int(len(bundle.sample_table)),
            "dataset_version": "topology_v6",
            "dataset_is_synthetic_noisy_augmented": True,
            "flow": "residual_ctm_v4_cindex_ensemble",
            "residual_seed_count": len(seed_values),
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
        baseline_model, baseline_summary, _ = _train_fusion("baseline", arrays, config, device)
        torch.save(baseline_model.state_dict(), fold_dir / "baseline_concat.pt")
        write_json(fold_dir / "baseline_history.json", baseline_summary["history"])

        residual_summaries = []
        val_collections = []
        test_collections = []
        for seed_position, seed_value in enumerate(seed_values):
            set_seed(int(config["seed"]) + split.fold * 1000 + seed_value)
            seed_dir = fold_dir / f"seed_{seed_value}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            residual_model, residual_summary, residual_details = _train_residual_v4(
                arrays,
                baseline_model,
                config,
                device,
            )
            torch.save(residual_model.state_dict(), seed_dir / "residual_ctm_v4.pt")
            write_json(seed_dir / "history.json", residual_summary["history"])
            write_json(seed_dir / "analysis.json", residual_details)
            val_collections.append(
                _collect_baseline_and_delta(
                    baseline_model,
                    residual_model,
                    arrays.val,
                    int(config["fusion"]["batch_size"]),
                    device,
                )
            )
            test_collections.append(
                _collect_baseline_and_delta(
                    baseline_model,
                    residual_model,
                    arrays.test,
                    int(config["fusion"]["batch_size"]),
                    device,
                )
            )
            residual_summary["seed"] = seed_value
            residual_summary["seed_position"] = seed_position
            residual_summaries.append(residual_summary)

        val_reference = val_collections[0]
        val_deltas = [collection["delta"] for collection in val_collections]
        selection = choose_cindex_ensemble(
            baseline_risk=val_reference["baseline"],
            deltas=val_deltas,
            times=val_reference["time"],
            events=val_reference["event"],
            alpha_grid=[float(alpha) for alpha in config["calibration"]["alpha_grid"]],
            min_c_index_delta=float(config["ensemble"]["min_validation_c_index_delta"]),
            softmax_temperature=float(config["ensemble"]["softmax_temperature"]),
        )
        test_reference = test_collections[0]
        test_deltas = [collection["delta"] for collection in test_collections]
        selected_metrics, selected_details = _evaluate_selected_ensemble(
            test_reference,
            test_deltas,
            selection.weights,
        )
        write_json(
            fold_dir / "residual_ctm_v4_selection.json",
            {
                "candidate_name": selection.candidate_name,
                "weights": selection.weights,
                "validation_c_index": selection.c_index,
                "validation_baseline_c_index": selection.baseline_c_index,
                "candidates": selection.candidates,
                "test": selected_metrics,
                "pair_diagnostics": selected_details["pair_diagnostics"],
            },
        )
        write_csv(
            fold_dir / "test_predictions.csv",
            _prediction_rows(arrays.test, selected_details, test_collections, selection.weights, seed_values),
        )

        graph_summary.pop("history")
        baseline_summary.pop("history")
        for residual_summary in residual_summaries:
            residual_summary.pop("history")
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoder": graph_summary,
                "baseline": baseline_summary,
                "residual_ctm_seeds": residual_summaries,
                "residual_ctm_v4_selected": {
                    "candidate_name": selection.candidate_name,
                    "weights": selection.weights,
                    "validation_c_index": selection.c_index,
                    "validation_baseline_c_index": selection.baseline_c_index,
                    "candidate_count": len(selection.candidates),
                    "test": selected_metrics,
                    "pair_diagnostics": selected_details["pair_diagnostics"],
                },
            },
        )

    return build_residual_v4_comparison_summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/residual_ctm_v4.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_residual_v4_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
