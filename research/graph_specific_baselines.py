from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List

import yaml

from research.baseline_compare import (
    build_tabular_dataframe,
    prepare_split_data,
    split_dataframe,
    train_tabular_cox,
)


def run_graph_specific_baselines(config_path: str, seeds: List[int]) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df, feature_groups = build_tabular_dataframe(config)

    baseline_specs = {
        "graph_summary_linear_cox": {
            "features": feature_groups["graph_summary"],
            "model_type": "linear",
        },
        "graph_summary_mlp_cox": {
            "features": feature_groups["graph_summary"],
            "model_type": "mlp",
        },
    }

    all_runs = {name: [] for name in baseline_specs}

    for seed in seeds:
        train_df, val_df, test_df = split_dataframe(
            df=df,
            seed=seed,
            val_ratio=config["train"]["val_ratio"],
            test_ratio=config["train"]["test_ratio"],
        )

        for baseline_name, spec in baseline_specs.items():
            split = prepare_split_data(train_df, val_df, test_df, spec["features"])
            _, val_metrics, test_metrics = train_tabular_cox(
                split=split,
                model_type=spec["model_type"],
                hidden_dim=max(16, int(config["train"].get("hidden_dim", 32))),
                dropout=float(config["train"].get("dropout", 0.3)),
                lr=float(config["train"].get("lr", 5e-4)),
                weight_decay=float(config["train"].get("weight_decay", 1e-3)),
                epochs=max(40, int(config["train"].get("epochs", 80))),
                patience=max(8, int(config["train"].get("early_stop_patience", 10))),
                min_delta=float(config["train"].get("min_delta", 5e-4)),
                seed=seed,
            )
            all_runs[baseline_name].append(
                {
                    "seed": seed,
                    "num_features": len(spec["features"]),
                    "model_type": spec["model_type"],
                    "best_val_c_index": val_metrics["best_val_c_index"],
                    "test_loss": test_metrics["loss"],
                    "test_c_index": test_metrics["c_index"],
                }
            )

    summary = {"config_path": config_path, "seeds": seeds, "baselines": {}}
    for baseline_name, runs in all_runs.items():
        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        summary["baselines"][baseline_name] = {
            "runs": runs,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_loss": statistics.mean(losses),
        }

    output_dir = Path("outputs/current_mainline")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "graph_specific_baselines_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    args = parser.parse_args()

    summary = run_graph_specific_baselines(config_path=args.config, seeds=args.seeds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()