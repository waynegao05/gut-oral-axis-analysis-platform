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
    train_discrete_hazard_logistic,
    train_tabular_cox,
)
from research.data import split_sample_table
from research.task import get_survival_task_definition


def run_graph_specific_baselines(
    config_path: str,
    seeds: List[int],
    output_root: str = "outputs/current_mainline_v2",
    split_seed: int | None = None,
    only_baselines: List[str] | None = None,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df, feature_groups, data_summary = build_tabular_dataframe(config)

    baseline_specs = {
        "graph_summary_linear_cox": {
            "features": feature_groups["graph_summary"],
            "baseline_family": "linear_cox",
        },
        "graph_summary_mlp_cox": {
            "features": feature_groups["graph_summary"],
            "baseline_family": "mlp_cox",
        },
        "graph_summary_discrete_hazard_logistic": {
            "features": feature_groups["graph_summary"],
            "baseline_family": "discrete_hazard_logistic",
        },
    }
    if only_baselines:
        missing = sorted(set(only_baselines).difference(baseline_specs))
        if missing:
            raise ValueError(
                f"Unknown graph baselines requested in --only: {missing}. "
                f"Available baselines: {sorted(baseline_specs)}"
            )
        baseline_specs = {name: baseline_specs[name] for name in only_baselines}

    all_runs = {name: [] for name in baseline_specs}

    for seed in seeds:
        effective_split_seed = seed if split_seed is None else split_seed
        train_df, val_df, test_df, split_summary = split_sample_table(
            sample_df=df[["sample_id", "time", "event"] + feature_groups["graph_summary"]].copy(),
            seed=effective_split_seed,
            val_ratio=config["train"]["val_ratio"],
            test_ratio=config["train"]["test_ratio"],
        )

        for baseline_name, spec in baseline_specs.items():
            split = prepare_split_data(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_columns=spec["features"],
                num_time_bins=int(config["train"].get("num_time_bins", 12)),
            )
            if spec["baseline_family"] == "linear_cox":
                _, val_metrics, test_metrics = train_tabular_cox(
                    split=split,
                    model_type="linear",
                    hidden_dim=max(16, int(config["train"].get("hidden_dim", 32))),
                    dropout=float(config["train"].get("dropout", 0.3)),
                    lr=float(config["train"].get("lr", 5e-4)),
                    weight_decay=float(config["train"].get("weight_decay", 1e-3)),
                    epochs=max(40, int(config["train"].get("epochs", 80))),
                    patience=max(8, int(config["train"].get("early_stop_patience", 10))),
                    min_delta=float(config["train"].get("min_delta", 5e-4)),
                    seed=seed,
                )
            elif spec["baseline_family"] == "mlp_cox":
                _, val_metrics, test_metrics = train_tabular_cox(
                    split=split,
                    model_type="mlp",
                    hidden_dim=max(16, int(config["train"].get("hidden_dim", 32))),
                    dropout=float(config["train"].get("dropout", 0.3)),
                    lr=float(config["train"].get("lr", 5e-4)),
                    weight_decay=float(config["train"].get("weight_decay", 1e-3)),
                    epochs=max(40, int(config["train"].get("epochs", 80))),
                    patience=max(8, int(config["train"].get("early_stop_patience", 10))),
                    min_delta=float(config["train"].get("min_delta", 5e-4)),
                    seed=seed,
                )
            elif spec["baseline_family"] == "discrete_hazard_logistic":
                _, val_metrics, test_metrics = train_discrete_hazard_logistic(split=split, seed=seed)
            else:
                raise ValueError(f"Unsupported baseline family: {spec['baseline_family']}")

            all_runs[baseline_name].append(
                {
                    "seed": seed,
                    "num_features": len(spec["features"]),
                    "baseline_family": spec["baseline_family"],
                    "split_seed": effective_split_seed,
                    "best_val_c_index": val_metrics["best_val_c_index"],
                    "test_loss": test_metrics["loss"],
                    "test_c_index": test_metrics["c_index"],
                    "split_summary": split_summary,
                }
            )

    summary = {
        "config_path": config_path,
        "output_root": output_root,
        "task_definition": get_survival_task_definition(),
        "data_summary": data_summary,
        "seeds": seeds,
        "split_seed": split_seed,
        "selected_baselines": list(baseline_specs.keys()),
        "baselines": {},
    }
    for baseline_name, runs in all_runs.items():
        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        summary["baselines"][baseline_name] = {
            "runs": runs,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": float(min(c_indices)),
            "max_test_c_index": float(max(c_indices)),
            "mean_test_loss": statistics.mean(losses),
        }

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "graph_specific_baselines_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2")
    args = parser.parse_args()

    summary = run_graph_specific_baselines(
        config_path=args.config,
        seeds=args.seeds,
        output_root=args.output_root,
        split_seed=args.split_seed,
        only_baselines=args.only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
