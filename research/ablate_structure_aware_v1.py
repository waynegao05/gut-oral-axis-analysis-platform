from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


ABLATIONS = {
    "mainline": {},
    "no_graph_aux": {"graph_aux_weight": 0.0},
    "no_node_aux": {"node_aux_weight": 0.0},
    "no_struct_aux": {"graph_aux_weight": 0.0, "node_aux_weight": 0.0},
}


def run_ablation(
    base_config_path: str,
    seeds: list[int],
    device: str,
    split_seed: int | None = None,
    output_root: str = "outputs/current_mainline_v2/ablations",
) -> dict:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    temp_root = Path(output_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_config_path": base_config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "output_root": output_root,
        "ablations": {},
    }

    for ablation_name, overrides in ABLATIONS.items():
        runs = []
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            if split_seed is not None:
                config.setdefault("train", {})
                config["train"]["split_seed"] = split_seed
            for key, value in overrides.items():
                config["train"][key] = value
            config["paths"]["output_dir"] = str((temp_root / f"{ablation_name}_seed{seed}").as_posix())

            temp_config_path = temp_root / f"{ablation_name}_seed{seed}.yaml"
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
            ]
            if split_seed is not None:
                cmd.extend(["--split-seed", str(split_seed)])
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
                    "overrides": overrides,
                    "output_dir": config["paths"]["output_dir"],
                }
            )

        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        summary["ablations"][ablation_name] = {
            "overrides": overrides,
            "runs": runs,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_loss": statistics.mean(losses),
        }

    out_path = temp_root / "ablation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/ablations")
    args = parser.parse_args()

    result = run_ablation(
        args.config,
        args.seeds,
        args.device,
        split_seed=args.split_seed,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
