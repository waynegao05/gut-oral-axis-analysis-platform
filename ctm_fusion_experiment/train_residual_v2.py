from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from ctm_fusion_experiment.evaluate_residual_v2 import build_residual_v2_comparison_summary
from ctm_fusion_experiment.train import (
    _export_graph_embeddings,
    _train_fusion,
    _train_graph_encoder,
    resolve_device,
    set_seed,
)
from ctm_fusion_experiment.train_residual import (
    _evaluate_residual,
    _prediction_rows,
    _train_residual,
)
from ctm_fusion_experiment.utils.calibration import apply_residual_alpha, choose_residual_alpha
from ctm_fusion_experiment.utils.data_loader import (
    FusionArraySet,
    load_graph_dataset,
    make_cv_splits,
    prepare_fusion_arrays,
)
from ctm_fusion_experiment.utils.losses import cox_partial_likelihood_loss
from ctm_fusion_experiment.utils.metrics import concordance_index
from ctm_fusion_experiment.utils.reporting import save_embeddings, write_csv, write_json


def _collect_baseline_and_delta(
    baseline_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    arrays: FusionArraySet,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from ctm_fusion_experiment.train import _tensor_loader
    from ctm_fusion_experiment.train_residual import _baseline_risk

    baseline_model.eval()
    residual_model.eval()
    baseline_risks = []
    deltas = []
    times = []
    events = []
    with torch.no_grad():
        for graph, clinical, metabolite, batch_times, batch_events in _tensor_loader(arrays, batch_size, shuffle=False):
            graph = graph.to(device)
            clinical = clinical.to(device)
            metabolite = metabolite.to(device)
            baseline = _baseline_risk(baseline_model, graph, clinical, metabolite)
            output = residual_model(graph, clinical, metabolite, baseline_risk=baseline)
            baseline_risks.append(baseline.detach().cpu())
            deltas.append((output["risk_per_tick"][:, -1] - baseline).detach().cpu())
            times.append(batch_times.detach().cpu())
            events.append(batch_events.detach().cpu())
    return {
        "baseline": torch.cat(baseline_risks),
        "delta": torch.cat(deltas),
        "time": torch.cat(times),
        "event": torch.cat(events),
    }


def _evaluate_calibrated(
    baseline_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    arrays: FusionArraySet,
    alpha: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    collected = _collect_baseline_and_delta(baseline_model, residual_model, arrays, batch_size, device)
    risk = apply_residual_alpha(collected["baseline"], collected["delta"], alpha)
    loss = cox_partial_likelihood_loss(risk, collected["time"], collected["event"])
    metrics = {
        "loss": float(loss.item()),
        "c_index": concordance_index(
            collected["time"].numpy(),
            collected["event"].numpy(),
            risk.numpy(),
        ),
    }
    details = {
        "risk": risk.tolist(),
        "baseline_risk": collected["baseline"].tolist(),
        "delta": collected["delta"].tolist(),
        "selected_alpha": float(alpha),
    }
    return metrics, details


def _calibrate_alpha(
    baseline_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    arrays,
    config: dict[str, Any],
    device: torch.device,
):
    collected = _collect_baseline_and_delta(
        baseline_model,
        residual_model,
        arrays.val,
        int(config["fusion"]["batch_size"]),
        device,
    )
    return choose_residual_alpha(
        baseline_risk=collected["baseline"],
        delta=collected["delta"],
        times=collected["time"],
        events=collected["event"],
        alpha_grid=[float(alpha) for alpha in config["calibration"]["alpha_grid"]],
    )


def _calibrated_prediction_rows(
    arrays: FusionArraySet,
    raw_details: dict[str, Any],
    calibrated_details: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _prediction_rows(arrays, {"risk": calibrated_details["baseline_risk"]}, raw_details)
    for index, row in enumerate(rows):
        row["residual_raw_risk"] = row.pop("residual_ctm_risk")
        row["residual_raw_stable_tick"] = row.pop("residual_ctm_stable_tick")
        row["residual_calibrated_risk"] = calibrated_details["risk"][index]
        row["residual_delta"] = calibrated_details["delta"][index]
        row["selected_alpha"] = calibrated_details["selected_alpha"]
    return rows


def run_residual_v2_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
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
            "flow": "residual_ctm_v2_calibrated",
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

        set_seed(int(config["seed"]) + split.fold)
        residual_model, residual_summary, raw_details = _train_residual(arrays, baseline_model, config, device)
        torch.save(residual_model.state_dict(), fold_dir / "residual_ctm_v2.pt")
        write_json(fold_dir / "residual_ctm_v2_history.json", residual_summary["history"])
        write_json(fold_dir / "residual_ctm_raw_analysis.json", raw_details)

        alpha_result = _calibrate_alpha(baseline_model, residual_model, arrays, config, device)
        calibrated_metrics, calibrated_details = _evaluate_calibrated(
            baseline_model,
            residual_model,
            arrays.test,
            alpha=alpha_result.alpha,
            batch_size=int(config["fusion"]["batch_size"]),
            device=device,
        )
        calibrated_details["validation_alpha_candidates"] = alpha_result.candidates
        write_json(fold_dir / "residual_ctm_calibration.json", calibrated_details)
        write_csv(fold_dir / "test_predictions.csv", _calibrated_prediction_rows(arrays.test, raw_details, calibrated_details))

        graph_summary.pop("history")
        baseline_summary.pop("history")
        residual_summary.pop("history")
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoder": graph_summary,
                "baseline": baseline_summary,
                "residual_ctm_raw": residual_summary,
                "residual_ctm_calibrated": {
                    "parameters": residual_summary["parameters"],
                    "selected_alpha": alpha_result.alpha,
                    "validation_c_index": alpha_result.c_index,
                    "alpha_candidates": alpha_result.candidates,
                    "test": calibrated_metrics,
                },
            },
        )

    return build_residual_v2_comparison_summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/residual_ctm_v2.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_residual_v2_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
