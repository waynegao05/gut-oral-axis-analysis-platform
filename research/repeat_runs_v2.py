from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run_repeated_training(
    base_config_path: str,
    seeds: list[int],
    device: str,
    split_seed: int | None = None,
    output_root: str = "outputs/current_mainline_v2",
    epochs_override: int | None = None,
    patience_override: int | None = None,
    batch_size_override: int | None = None,
    resume_existing: bool = False,
    validation_group: str | int | None = None,
    test_group: str | int | None = None,
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
        if validation_group is not None:
            config.setdefault("train", {})
            config["train"]["validation_group"] = validation_group
        if test_group is not None:
            config.setdefault("train", {})
            config["train"]["test_group"] = test_group
        if epochs_override is not None:
            config["train"]["epochs"] = int(epochs_override)
        if patience_override is not None:
            config["train"]["early_stop_patience"] = int(patience_override)
        if batch_size_override is not None:
            config["train"]["batch_size"] = int(batch_size_override)
        config["paths"]["output_dir"] = str((output_root_path / f"research_seed{seed}").as_posix())

        metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
        training_summary_path = Path(config["paths"]["output_dir"]) / "training_summary.json"
        resumed = bool(
            resume_existing
            and metrics_path.exists()
            and training_summary_path.exists()
            and (Path(config["paths"]["output_dir"]) / "best_model.pt").exists()
        )

        if not resumed:
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
            time.sleep(3.0)

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))

        summary.append(
            {
                "seed": seed,
                "resumed_existing": resumed,
                "test_loss": metrics.get("cohort_loss", metrics["loss"]),
                "legacy_batch_test_loss": metrics["loss"],
                "test_c_index": metrics["c_index"],
                "train_c_index": training_summary["train_c_index"],
                "validation_c_index": training_summary["validation_c_index"],
                "test_cox_loss": metrics.get("cohort_cox_loss", metrics["cox_loss"]),
                "epochs_run": training_summary["epochs_run"],
                "training_seconds": training_summary["training_seconds"],
                "parameter_count": training_summary["parameter_count"],
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
        "epochs_override": epochs_override,
        "patience_override": patience_override,
        "batch_size_override": batch_size_override,
        "resume_existing": resume_existing,
        "validation_group": validation_group,
        "test_group": test_group,
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
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--patience-override", type=int, default=None)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--validation-group", default=None)
    parser.add_argument("--test-group", default=None)
    args = parser.parse_args()

    run_repeated_training(
        args.config,
        args.seeds,
        args.device,
        split_seed=args.split_seed,
        output_root=args.output_root,
        epochs_override=args.epochs_override,
        patience_override=args.patience_override,
        batch_size_override=args.batch_size_override,
        resume_existing=args.resume_existing,
        validation_group=args.validation_group,
        test_group=args.test_group,
    )


if __name__ == "__main__":
    main()
