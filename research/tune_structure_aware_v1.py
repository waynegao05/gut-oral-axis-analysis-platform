from __future__ import annotations

import argparse
import copy
import itertools
import json
import statistics
import subprocess
from pathlib import Path

import yaml


def run_tuning(base_config_path: str, seeds: list[int], device: str) -> dict:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))

    grid = {
        "hidden_dim": [56, 64, 72],
        "dropout": [0.2, 0.25, 0.3],
        "aux_loss_weight": [0.1, 0.15, 0.2],
    }

    temp_root = Path("outputs/tune_structure_aware_v1")
    temp_root.mkdir(parents=True, exist_ok=True)
    results = []

    combos = list(itertools.product(grid["hidden_dim"], grid["dropout"], grid["aux_loss_weight"]))
    for combo_id, (hidden_dim, dropout, aux_weight) in enumerate(combos, start=1):
        run_metrics = []
        combo_name = f"hd{hidden_dim}_do{str(dropout).replace('.', '')}_aux{str(aux_weight).replace('.', '')}"
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            config["train"]["hidden_dim"] = hidden_dim
            config["train"]["dropout"] = dropout
            config["train"]["aux_loss_weight"] = aux_weight
            config["paths"]["output_dir"] = f"outputs/tune_structure_aware_v1/{combo_name}_seed{seed}"

            temp_config_path = temp_root / f"{combo_name}_seed{seed}.yaml"
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
            run_metrics.append(
                {
                    "seed": seed,
                    "test_loss": metrics["loss"],
                    "test_c_index": metrics["c_index"],
                    "test_aux_loss": metrics.get("aux_loss", 0.0),
                    "output_dir": config["paths"]["output_dir"],
                }
            )

        c_indices = [item["test_c_index"] for item in run_metrics]
        losses = [item["test_loss"] for item in run_metrics]
        result = {
            "combo_id": combo_id,
            "combo_name": combo_name,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "aux_loss_weight": aux_weight,
            "runs": run_metrics,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_loss": statistics.mean(losses),
        }
        results.append(result)

    results.sort(key=lambda x: (x["mean_test_c_index"], -x["std_test_c_index"]), reverse=True)
    summary = {
        "base_config_path": base_config_path,
        "seeds": seeds,
        "search_space": grid,
        "num_combinations": len(combos),
        "results": results,
        "best": results[0] if results else None,
    }

    out_path = Path("outputs/tune_structure_aware_v1_summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_structure_aware.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    summary = run_tuning(args.config, args.seeds, args.device)
    print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
