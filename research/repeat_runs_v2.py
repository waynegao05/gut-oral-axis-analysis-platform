from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


def run_repeated_training(
    base_config_path: str,
    seeds: list[int],
    device: str,
    split_seed: int | None = None,
    output_root: str = "outputs/current_mainline_v2",
) -> dict:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    summary = []

    output_root_path = Path(output_root)
    temp_dir = output_root_path / "repeat_runs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        if split_seed is not None:
            config.setdefault("train", {})
            config["train"]["split_seed"] = split_seed
        config["paths"]["output_dir"] = str((output_root_path / f"research_seed{seed}").as_posix())

        temp_config_path = temp_dir / f"temp_config_seed{seed}.yaml"
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

        summary.append(
            {
                "seed": seed,
                "test_loss": metrics["loss"],
                "test_c_index": metrics["c_index"],
                "graph_aux_loss": metrics.get("graph_aux_loss", 0.0),
                "node_aux_loss": metrics.get("node_aux_loss", 0.0),
                "output_dir": config["paths"]["output_dir"],
            }
        )

    c_indices = [item["test_c_index"] for item in summary]
    losses = [item["test_loss"] for item in summary]

    result = {
        "base_config_path": base_config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "output_root": output_root,
        "runs": summary,
        "mean_test_c_index": statistics.mean(c_indices),
        "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
        "min_test_c_index": min(c_indices),
        "max_test_c_index": max(c_indices),
        "mean_test_loss": statistics.mean(losses),
    }

    out_path = output_root_path / "research_repeat_runs_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2")
    args = parser.parse_args()

    run_repeated_training(
        args.config,
        args.seeds,
        args.device,
        split_seed=args.split_seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
