from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from ctm_fusion_experiment.evaluate_residual_v3 import build_residual_v3_comparison_summary
from ctm_fusion_experiment.train import (
    _export_graph_embeddings,
    _train_fusion,
    _train_graph_encoder,
    resolve_device,
    set_seed,
)
from ctm_fusion_experiment.train_residual import _train_residual
from ctm_fusion_experiment.train_residual_v2 import _collect_baseline_and_delta
from ctm_fusion_experiment.utils.aggressive_calibration import choose_aggressive_residual_alpha
from ctm_fusion_experiment.utils.calibration import apply_residual_alpha
from ctm_fusion_experiment.utils.data_loader import (
    FusionArraySet,
    load_graph_dataset,
    make_cv_splits,
    prepare_fusion_arrays,
)
from ctm_fusion_experiment.utils.losses import cox_partial_likelihood_loss
from ctm_fusion_experiment.utils.reporting import save_embeddings, write_csv, write_json
from ctm_fusion_experiment.utils.utility_metrics import risk_utility_metrics


def _evaluate_alpha_from_collected(
    collected: dict[str, torch.Tensor],
    *,
    alpha: float,
    top_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    risk = apply_residual_alpha(collected["baseline"], collected["delta"], alpha)
    loss = cox_partial_likelihood_loss(risk, collected["time"], collected["event"])
    utility = risk_utility_metrics(
        collected["time"].numpy(),
        collected["event"].numpy(),
        risk.numpy(),
        top_fraction=top_fraction,
    )
    metrics = {
        "loss": float(loss.item()),
        "c_index": float(utility["c_index"]),
    }
    details = {
        "risk": risk.tolist(),
        "baseline_risk": collected["baseline"].tolist(),
        "delta": collected["delta"].tolist(),
        "selected_alpha": float(alpha),
        "utility": utility,
    }
    return metrics, details


def _calibrate_aggressive_alpha(
    baseline_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    arrays,
    config: dict[str, Any],
    device: torch.device,
):
    selection = config["aggressive_selection"]
    collected = _collect_baseline_and_delta(
        baseline_model,
        residual_model,
        arrays.val,
        int(config["fusion"]["batch_size"]),
        device,
    )
    return choose_aggressive_residual_alpha(
        baseline_risk=collected["baseline"],
        delta=collected["delta"],
        times=collected["time"],
        events=collected["event"],
        alpha_grid=[float(alpha) for alpha in config["calibration"]["alpha_grid"]],
        top_fraction=float(selection["top_fraction"]),
        c_index_weight=float(selection["c_index_weight"]),
        top_event_lift_weight=float(selection["top_event_lift_weight"]),
        high_low_gap_weight=float(selection["high_low_gap_weight"]),
        risk_spread_weight=float(selection["risk_spread_weight"]),
        min_c_index_delta=float(selection["min_c_index_delta"]),
        min_objective_delta=float(selection["min_objective_delta"]),
    )


def _aggressive_prediction_rows(
    arrays: FusionArraySet,
    raw_details: dict[str, Any],
    aggressive_details: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for index, sample_id in enumerate(arrays.sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "time": float(arrays.time[index]),
                "event": float(arrays.event[index]),
                "baseline_risk": aggressive_details["baseline_risk"][index],
                "residual_raw_risk": raw_details["risk"][index],
                "residual_raw_stable_tick": raw_details["stable_ticks"][index],
                "residual_delta": aggressive_details["delta"][index],
                "residual_aggressive_risk": aggressive_details["risk"][index],
                "selected_alpha": aggressive_details["selected_alpha"],
            }
        )
    return rows


def run_residual_v3_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
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
            "flow": "residual_ctm_v3_aggressive",
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

    top_fraction = float(config["aggressive_selection"]["top_fraction"])
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
        torch.save(residual_model.state_dict(), fold_dir / "residual_ctm_v3.pt")
        write_json(fold_dir / "residual_ctm_v3_history.json", residual_summary["history"])
        write_json(fold_dir / "residual_ctm_raw_analysis.json", raw_details)

        alpha_result = _calibrate_aggressive_alpha(baseline_model, residual_model, arrays, config, device)
        test_collected = _collect_baseline_and_delta(
            baseline_model,
            residual_model,
            arrays.test,
            int(config["fusion"]["batch_size"]),
            device,
        )
        baseline_test_utility = risk_utility_metrics(
            test_collected["time"].numpy(),
            test_collected["event"].numpy(),
            test_collected["baseline"].numpy(),
            top_fraction=top_fraction,
        )
        raw_metrics, raw_alpha_details = _evaluate_alpha_from_collected(
            test_collected,
            alpha=1.0,
            top_fraction=top_fraction,
        )
        aggressive_metrics, aggressive_details = _evaluate_alpha_from_collected(
            test_collected,
            alpha=alpha_result.alpha,
            top_fraction=top_fraction,
        )
        aggressive_details["validation_alpha_candidates"] = alpha_result.candidates
        aggressive_details["validation_baseline_utility"] = alpha_result.baseline_utility
        aggressive_details["validation_selected_utility"] = alpha_result.utility
        aggressive_details["validation_objective"] = alpha_result.objective
        write_json(fold_dir / "residual_ctm_aggressive_calibration.json", aggressive_details)
        write_csv(fold_dir / "test_predictions.csv", _aggressive_prediction_rows(arrays.test, raw_details, aggressive_details))

        graph_summary.pop("history")
        baseline_summary.pop("history")
        residual_summary.pop("history")
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoder": graph_summary,
                "baseline": {
                    **baseline_summary,
                    "test_utility": baseline_test_utility,
                },
                "residual_ctm_raw": {
                    **residual_summary,
                    "test": {
                        **residual_summary["test"],
                        "c_index": raw_metrics["c_index"],
                    },
                    "test_utility": raw_alpha_details["utility"],
                },
                "residual_ctm_aggressive": {
                    "parameters": residual_summary["parameters"],
                    "selected_alpha": alpha_result.alpha,
                    "validation_objective": alpha_result.objective,
                    "validation_c_index": alpha_result.c_index,
                    "validation_baseline_utility": alpha_result.baseline_utility,
                    "validation_selected_utility": alpha_result.utility,
                    "alpha_candidates": alpha_result.candidates,
                    "test": aggressive_metrics,
                    "test_utility": aggressive_details["utility"],
                },
            },
        )

    return build_residual_v3_comparison_summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/residual_ctm_v3.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_residual_v3_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
