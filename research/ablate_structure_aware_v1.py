from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
from pathlib import Path

import yaml


ABLATIONS = {
    "full": {"aux_loss_weight": 0.1},
    "no_aux": {"aux_loss_weight": 0.0},
    "mid_aux": {"aux_loss_weight": 0.2},
    "strong_aux": {"aux_loss_weight": 0.3},
}


def run_ablation(base_config_path: str, seeds: list[int], device: str) -> dict:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    temp_root = Path("outputs/current_mainline/ablate_structure_aware_v1")
    temp_root.mkdir(parents=True, exist_ok=True)
    summary = {"base_config_path": base_config_path, "seeds": seeds, "ablations": {}}

    for ablation_name, overrides in ABLATIONS.items():
        runs = []
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            for k, v in overrides.items():
                config["train"][k] = v
            config["paths"]["output_dir"] = f"outputs/current_mainline/ablate_structure_aware_v1/{ablation_name}_seed{seed}"

            temp_config_path = temp_root / f"{ablation_name}_seed{seed}.yaml"
            temp_config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            cmd = [
                "python",
                "-m",
                "research.train",
                "--config",
                str(temp_config_path),
                "--device",
                device,
            ]
            subprocess.run(cmd, check=True)

            metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "seed": seed,
                    "test_loss": metrics["loss"],
                    "test_c_index": metrics["c_index"],
                    "test_aux_loss": metrics.get("aux_loss", 0.0),
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

    out_path = Path("outputs/current_mainline/ablate_structure_aware_v1_summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    result = run_ablation(args.config, args.seeds, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()