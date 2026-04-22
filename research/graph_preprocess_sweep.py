from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


GRAPH_PREPROCESS_PRESETS = {
    "baseline": {"keep_top_k_edges": None, "min_edge_weight": None},
    "topk8": {"keep_top_k_edges": 8, "min_edge_weight": None},
    "topk6": {"keep_top_k_edges": 6, "min_edge_weight": None},
    "minw03": {"keep_top_k_edges": None, "min_edge_weight": 0.3},
    "minw05": {"keep_top_k_edges": None, "min_edge_weight": 0.5},
    "topk6_minw03": {"keep_top_k_edges": 6, "min_edge_weight": 0.3},
}

DEFAULT_SWEEP_NAMES = ["baseline", "topk8", "topk6", "minw05"]


def run_graph_preprocess_sweep(
    base_config_path: str,
    sweep_names: list[str],
    seeds: list[int],
    device: str,
    split_seed: int = 42,
    output_root: str = "outputs/current_mainline_v2/graph_preprocess_sweep",
) -> dict:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_config_path": base_config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "output_root": output_root,
        "sweeps": {},
    }

    for sweep_name in sweep_names:
        if sweep_name not in GRAPH_PREPROCESS_PRESETS:
            raise ValueError(
                f"Unsupported graph preprocess preset: {sweep_name}. "
                f"Available presets: {sorted(GRAPH_PREPROCESS_PRESETS)}"
            )

        overrides = GRAPH_PREPROCESS_PRESETS[sweep_name]
        runs = []

        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            config.setdefault("train", {})
            config["train"]["split_seed"] = split_seed
            config.setdefault("graph_preprocess", {})
            config["graph_preprocess"]["keep_top_k_edges"] = overrides["keep_top_k_edges"]
            config["graph_preprocess"]["min_edge_weight"] = overrides["min_edge_weight"]
            config["paths"]["output_dir"] = str((output_root_path / f"{sweep_name}_seed{seed}").as_posix())

            temp_config_path = output_root_path / f"{sweep_name}_seed{seed}.yaml"
            temp_config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                "-m",
                "research.train_v2",
                "--config",
                str(temp_config_path),
                "--device",
                device,
                "--split-seed",
                str(split_seed),
            ]
            subprocess.run(cmd, check=True)

            metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "seed": seed,
                    "test_loss": metrics["loss"],
                    "test_c_index": metrics["c_index"],
                    "graph_aux_loss": metrics.get("graph_aux_loss", 0.0),
                    "node_aux_loss": metrics.get("node_aux_loss", 0.0),
                    "graph_preprocess": overrides,
                    "output_dir": config["paths"]["output_dir"],
                }
            )

        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        summary["sweeps"][sweep_name] = {
            "graph_preprocess": overrides,
            "runs": runs,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_loss": statistics.mean(losses),
        }

    ranking_table = []
    for sweep_name, result in summary["sweeps"].items():
        ranking_table.append(
            {
                "sweep_name": sweep_name,
                "graph_preprocess": result["graph_preprocess"],
                "mean_test_c_index": result["mean_test_c_index"],
                "std_test_c_index": result["std_test_c_index"],
                "mean_test_loss": result["mean_test_loss"],
            }
        )
    ranking_table.sort(key=lambda row: row["mean_test_c_index"], reverse=True)
    summary["ranking_table"] = ranking_table

    out_path = output_root_path / "graph_preprocess_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--sweeps", nargs="+", default=DEFAULT_SWEEP_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/graph_preprocess_sweep")
    args = parser.parse_args()

    run_graph_preprocess_sweep(
        base_config_path=args.config,
        sweep_names=args.sweeps,
        seeds=args.seeds,
        device=args.device,
        split_seed=args.split_seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
