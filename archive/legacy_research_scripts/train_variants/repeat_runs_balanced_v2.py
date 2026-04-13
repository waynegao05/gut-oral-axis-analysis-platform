from __future__ import annotations

import copy
import json
import statistics
import subprocess
from pathlib import Path

import yaml


def run_repeated_training(
    base_config_path: str = "research_config_balanced_eval_v2.yaml",
    seeds: list[int] | None = None,
) -> None:
    if seeds is None:
        seeds = [7, 21, 42, 123, 2026]

    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    summary = []

    temp_dir = Path("outputs/repeat_runs_balanced_v2")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        config["paths"]["output_dir"] = f"outputs/research_balanced_eval_seed{seed}"

        temp_config_path = temp_dir / f"temp_config_seed{seed}.yaml"
        temp_config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        cmd = [
            "python",
            "-m",
            "research.train_balanced_v2",
            "--config",
            str(temp_config_path),
        ]
        subprocess.run(cmd, check=True)

        metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
        split_summary_path = Path(config["paths"]["output_dir"]) / "split_summary.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

        summary.append(
            {
                "seed": seed,
                "test_loss": metrics["loss"],
                "test_c_index": metrics["c_index"],
                "split_summary": split_summary,
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

    out_path = Path("outputs/research_balanced_eval_summary.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_repeated_training()
