from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ctm_fusion_experiment.evaluate_baseline_v5 import build_baseline_v5_comparison_summary
from ctm_fusion_experiment.models.baseline_concat import BaselineConcatModel
from ctm_fusion_experiment.train import (
    _copy_state_dict,
    _evaluate_fusion,
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
from ctm_fusion_experiment.utils.bootstrap_risk_ensemble_selection import choose_bootstrap_risk_ensemble
from ctm_fusion_experiment.utils.losses import cox_partial_likelihood_loss, pairwise_ranking_loss
from ctm_fusion_experiment.utils.metrics import concordance_index
from ctm_fusion_experiment.utils.pair_diagnostics import pairwise_cindex_diagnostics
from ctm_fusion_experiment.utils.reporting import count_parameters, save_embeddings, write_csv, write_json
from ctm_fusion_experiment.utils.risk_ensemble_selection import (
    apply_risk_ensemble,
    choose_cindex_risk_ensemble,
)


def _train_pairwise_baseline(
    arrays,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    settings = config["fusion"]
    pairwise_weight = float(config["baseline_v5"].get("pairwise_ranking_weight", 0.0))
    pairwise_margin = float(config["baseline_v5"].get("pairwise_margin", 0.0))
    model = BaselineConcatModel(
        graph_dim=arrays.train.graph.shape[1],
        clinical_dim=arrays.train.clinical.shape[1],
        metabolite_dim=arrays.train.metabolite.shape[1],
        hidden_dim=int(settings["baseline_hidden_dim"]),
        dropout=float(settings["dropout"]),
    ).to(device)
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
        train_cox = []
        train_pairwise = []
        for graph, clinical, metabolite, times, events in train_loader:
            graph = graph.to(device)
            clinical = clinical.to(device)
            metabolite = metabolite.to(device)
            times = times.to(device)
            events = events.to(device)
            optimizer.zero_grad(set_to_none=True)
            risk = model(graph, clinical, metabolite)["risk"]
            cox = cox_partial_likelihood_loss(risk, times, events)
            ranking = pairwise_ranking_loss(risk, times, events, margin=pairwise_margin)
            loss = cox + pairwise_weight * ranking
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_cox.append(float(cox.detach().cpu().item()))
            train_pairwise.append(float(ranking.detach().cpu().item()))

        val_metrics, _ = _evaluate_fusion(
            model,
            arrays.val,
            "baseline",
            int(settings["batch_size"]),
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean(train_losses),
                "train_cox": _mean(train_cox),
                "train_pairwise": _mean(train_pairwise),
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
    val_metrics, val_details = _evaluate_fusion(
        model,
        arrays.val,
        "baseline",
        int(settings["batch_size"]),
        device,
    )
    test_metrics, test_details = _evaluate_fusion(
        model,
        arrays.test,
        "baseline",
        int(settings["batch_size"]),
        device,
    )
    return model, {
        "best_val_c_index": best_c_index,
        "training_seconds": time.perf_counter() - started,
        "parameters": count_parameters(model),
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
    }, val_details, test_details


def _candidate_record(
    *,
    name: str,
    graph_seed: int,
    baseline_seed: int,
    summary: dict[str, Any],
    val_details: dict[str, Any],
    test_details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "graph_seed": graph_seed,
        "baseline_seed": baseline_seed,
        "training_seconds": summary["training_seconds"],
        "parameters": summary["parameters"],
        "validation": summary["validation"],
        "test": summary["test"],
        "val_risk": val_details["risk"],
        "test_risk": test_details["risk"],
    }


def _selection_prediction_rows(
    arrays: FusionArraySet,
    candidate_records: list[dict[str, Any]],
    selected_risk: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for index, sample_id in enumerate(arrays.sample_ids):
        row = {
            "sample_id": sample_id,
            "time": float(arrays.time[index]),
            "event": float(arrays.event[index]),
            "reference_risk": candidate_records[0]["test_risk"][index],
            "selected_risk": selected_risk[index],
        }
        for candidate in candidate_records:
            row[f"{candidate['name']}_risk"] = candidate["test_risk"][index]
        rows.append(row)
    return rows


def _selected_metrics(
    arrays: FusionArraySet,
    reference_risk: torch.Tensor,
    selected_risk: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    loss = cox_partial_likelihood_loss(
        selected_risk,
        torch.tensor(arrays.time, dtype=torch.float32),
        torch.tensor(arrays.event, dtype=torch.float32),
    )
    c_index = concordance_index(arrays.time, arrays.event, selected_risk.detach().cpu().numpy())
    diagnostics = pairwise_cindex_diagnostics(
        arrays.time,
        arrays.event,
        reference_risk.detach().cpu().numpy(),
        selected_risk.detach().cpu().numpy(),
    )
    return {"loss": float(loss.item()), "c_index": float(c_index)}, diagnostics


def _graph_seed_value(base_seed: int, fold: int, graph_seed: int) -> int:
    if graph_seed == 0:
        return base_seed + fold
    return base_seed + fold * 10000 + graph_seed


def run_baseline_v5_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = resolve_device(device_arg)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", config)
    bundle = load_graph_dataset(config, sample_limit=config.get("runtime", {}).get("sample_limit"))
    graph_seeds = [int(seed) for seed in config["baseline_v5"]["graph_seeds"]]
    baseline_seeds = [int(seed) for seed in config["baseline_v5"]["baseline_seeds"]]
    write_json(
        output_dir / "run_metadata.json",
        {
            "device": str(device),
            "sample_count": int(len(bundle.sample_table)),
            "dataset_version": "topology_v6",
            "dataset_is_synthetic_noisy_augmented": True,
            "flow": "baseline_v5_supervised_ensemble",
            "graph_seed_count": len(graph_seeds),
            "baseline_seed_count": len(baseline_seeds),
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
        fold_dir = output_dir / f"fold_{split.fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / "split.json", split.__dict__)
        graph_summaries = []
        candidate_records = []
        val_arrays: FusionArraySet | None = None
        test_arrays: FusionArraySet | None = None

        for graph_seed in graph_seeds:
            set_seed(_graph_seed_value(int(config["seed"]), split.fold, graph_seed))
            graph_dir = fold_dir / f"graph_seed_{graph_seed}"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_model, graph_summary = _train_graph_encoder(bundle, split, config, device)
            torch.save(graph_model.state_dict(), graph_dir / "graph_encoder.pt")
            export_ids = tuple(sorted(set(split.train_ids + split.val_ids + split.test_ids)))
            embeddings = _export_graph_embeddings(
                graph_model,
                bundle,
                export_ids,
                int(config["graph_encoder"]["batch_size"]),
                device,
            )
            save_embeddings(graph_dir / "frozen_graph_embeddings.npz", embeddings)
            arrays = prepare_fusion_arrays(
                sample_table=bundle.sample_table,
                split=split,
                graph_embeddings=embeddings,
                clinical_columns=config["model"]["clinical_columns"],
                metabolite_columns=config["model"]["metabolite_columns"],
            )
            val_arrays = arrays.val
            test_arrays = arrays.test
            graph_summary.pop("history")
            graph_summary["graph_seed"] = graph_seed
            graph_summaries.append(graph_summary)

            if graph_seed == graph_seeds[0]:
                set_seed(int(config["seed"]) + split.fold)
                reference_model, reference_summary, reference_test_details = _train_fusion("baseline", arrays, config, device)
                reference_val_metrics, reference_val_details = _evaluate_fusion(
                    reference_model,
                    arrays.val,
                    "baseline",
                    int(config["fusion"]["batch_size"]),
                    device,
                )
                torch.save(reference_model.state_dict(), graph_dir / "reference_baseline.pt")
                write_json(graph_dir / "reference_baseline_history.json", reference_summary["history"])
                reference_summary["validation"] = reference_val_metrics
                reference_summary.pop("history")
                candidate_records.append(
                    _candidate_record(
                        name="reference",
                        graph_seed=graph_seed,
                        baseline_seed=0,
                        summary=reference_summary,
                        val_details=reference_val_details,
                        test_details=reference_test_details,
                    )
                )

            for baseline_seed in baseline_seeds:
                set_seed(int(config["seed"]) + split.fold * 10000 + graph_seed * 100 + baseline_seed)
                candidate_dir = graph_dir / f"baseline_seed_{baseline_seed}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                model, summary, val_details, test_details = _train_pairwise_baseline(arrays, config, device)
                torch.save(model.state_dict(), candidate_dir / "baseline_v5.pt")
                write_json(candidate_dir / "history.json", summary["history"])
                summary.pop("history")
                candidate_records.append(
                    _candidate_record(
                        name=f"g{graph_seed}_b{baseline_seed}",
                        graph_seed=graph_seed,
                        baseline_seed=baseline_seed,
                        summary=summary,
                        val_details=val_details,
                        test_details=test_details,
                    )
                )

        if val_arrays is None or test_arrays is None:
            raise RuntimeError("No test arrays were prepared.")

        val_matrix = torch.tensor([candidate["val_risk"] for candidate in candidate_records], dtype=torch.float32)
        test_matrix = torch.tensor([candidate["test_risk"] for candidate in candidate_records], dtype=torch.float32)
        test_reference = torch.tensor(candidate_records[0]["test_risk"], dtype=torch.float32)
        selection = _choose_v5_selection(
            config=config,
            val_matrix=val_matrix,
            val_arrays=val_arrays,
            candidate_records=candidate_records,
            split_fold=split.fold,
        )
        selected_test_risk = apply_risk_ensemble(
            test_matrix,
            selection.weights,
            selection.risk_means,
            selection.risk_stds,
        )
        selected_metrics, diagnostics = _selected_metrics(test_arrays, test_reference, selected_test_risk)
        write_json(
            fold_dir / "baseline_v5_selection.json",
            {
                "candidate_name": selection.candidate_name,
                "weights": selection.weights,
                "validation_c_index": selection.c_index,
                "validation_reference_c_index": selection.reference_c_index,
                "risk_means": selection.risk_means,
                "risk_stds": selection.risk_stds,
                "candidates": selection.candidates,
                "test": selected_metrics,
                "pair_diagnostics": diagnostics,
            },
        )
        write_csv(
            fold_dir / "test_predictions.csv",
            _selection_prediction_rows(test_arrays, candidate_records, selected_test_risk.detach().cpu().tolist()),
        )
        candidate_summaries = [
            {
                "name": candidate["name"],
                "graph_seed": candidate["graph_seed"],
                "baseline_seed": candidate["baseline_seed"],
                "training_seconds": candidate["training_seconds"],
                "parameters": candidate["parameters"],
                "validation": candidate["validation"],
                "test": candidate["test"],
            }
            for candidate in candidate_records
        ]
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoders": graph_summaries,
                "reference_baseline": candidate_summaries[0],
                "candidate_models": candidate_summaries,
                "baseline_v5_selected": {
                    "candidate_name": selection.candidate_name,
                    "weights": selection.weights,
                    "validation_c_index": selection.c_index,
                    "validation_reference_c_index": selection.reference_c_index,
                    "candidate_count": len(selection.candidates),
                    "test": selected_metrics,
                    "pair_diagnostics": diagnostics,
                },
            },
        )

    return build_baseline_v5_comparison_summary(output_dir)


def _choose_v5_selection(
    *,
    config: dict[str, Any],
    val_matrix: torch.Tensor,
    val_arrays: FusionArraySet,
    candidate_records: list[dict[str, Any]],
    split_fold: int,
):
    model_names = [candidate["name"] for candidate in candidate_records]
    times = torch.tensor(val_arrays.time, dtype=torch.float32)
    events = torch.tensor(val_arrays.event, dtype=torch.float32)
    strategy = config["baseline_v5"].get("selection_strategy", "cindex")
    if strategy == "bootstrap":
        bootstrap = config["baseline_v5"].get("bootstrap", {})
        return choose_bootstrap_risk_ensemble(
            risk_matrix=val_matrix,
            times=times,
            events=events,
            model_names=model_names,
            reference_index=0,
            min_validation_delta=float(config["baseline_v5"]["min_validation_c_index_delta"]),
            min_resample_quantile_delta=float(bootstrap.get("min_resample_quantile_delta", -0.0005)),
            resamples=int(bootstrap.get("resamples", 80)),
            subsample_fraction=float(bootstrap.get("subsample_fraction", 0.7)),
            lower_quantile=float(bootstrap.get("lower_quantile", 0.1)),
            stability_penalty=float(bootstrap.get("stability_penalty", 0.5)),
            softmax_temperature=float(config["baseline_v5"]["softmax_temperature"]),
            candidate_policy=config["baseline_v5"].get("selection_policy", "all"),
            max_top_k=int(config["baseline_v5"].get("max_top_k", 3)),
            seed=int(config["seed"]) + int(split_fold) * 1000,
        )
    if strategy == "cindex":
        return choose_cindex_risk_ensemble(
            risk_matrix=val_matrix,
            times=times,
            events=events,
            model_names=model_names,
            reference_index=0,
            min_c_index_delta=float(config["baseline_v5"]["min_validation_c_index_delta"]),
            softmax_temperature=float(config["baseline_v5"]["softmax_temperature"]),
            candidate_policy=config["baseline_v5"].get("selection_policy", "all"),
            max_top_k=int(config["baseline_v5"].get("max_top_k", 3)),
        )
    raise ValueError(f"Unknown baseline_v5.selection_strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/baseline_v5.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_baseline_v5_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
