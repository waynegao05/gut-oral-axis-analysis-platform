from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
from pathlib import Path

import yaml


def run_repeated_training(base_config_path: str, seeds: list[int], device: str) -> dict:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    summary = []

    temp_dir = Path("outputs/repeat_runs_structure_aware")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        config["paths"]["output_dir"] = f"outputs/research_structure_aware_seed{seed}"

        temp_config_path = temp_dir / f"temp_config_seed{seed}.yaml"
        temp_config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        cmd = [
            "python",
            "-m",
            "research.train_structure_aware",
            "--config",
            str(temp_config_path),
            "--device",
            device,
        ]
        subprocess.run(cmd, check=True)

        metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary.append(
            {
                "seed": seed,
                "test_loss": metrics["loss"],
                "test_c_index": metrics["c_index"],
                "test_aux_loss": metrics.get("aux_loss", 0.0),
                "output_dir": config["paths"]["output_dir"],
            }
        )

    c_indices = [item["test_c_index"] for item in summary]
    result = {
        "base_config_path": base_config_path,
        "seeds": seeds,
        "runs": summary,
        "mean_test_c_index": statistics.mean(c_indices),
        "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
        "min_test_c_index": min(c_indices),
        "max_test_c_index": max(c_indices),
    }

    out_path = Path("outputs/research_structure_aware_summary.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_structure_aware.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    result = run_repeated_training(args.config, args.seeds, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
