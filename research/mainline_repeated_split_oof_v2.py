from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


def run_mainline_repeated_split_oof(
    *,
    base_config_path: str = "research_config_v2.yaml",
    split_seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    output_root: str | Path = "outputs/current_mainline_v2/mainline_repeated_split_oof_v2",
    device: str = "cuda",
    epochs_override: int | None = None,
    patience_override: int | None = None,
    max_runs: int | None = None,
    skip_completed: bool = True,
) -> dict[str, Any]:
    base_config = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_count = 0
    for split_seed in split_seeds:
        for model_seed in model_seeds:
            if max_runs is not None and run_count >= int(max_runs):
                break
            config = _build_config(
                base_config,
                split_seed=int(split_seed),
                model_seed=int(model_seed),
                output_dir=output_path / f"split_seed_{int(split_seed)}" / f"model_seed_{int(model_seed)}",
                epochs_override=epochs_override,
                patience_override=patience_override,
            )
            config_path = output_path / "configs" / f"split{int(split_seed)}_seed{int(model_seed)}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
            if not (skip_completed and metrics_path.exists()):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "research.train_v2",
                        "--config",
                        str(config_path),
                        "--split-seed",
                        str(int(split_seed)),
                        "--device",
                        device,
                    ],
                    check=True,
                )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "test_c_index": float(metrics["c_index"]),
                    "test_loss": float(metrics["loss"]),
                    "output_dir": config["paths"]["output_dir"],
                    "config_path": str(config_path),
                }
            )
            run_count += 1
        if max_runs is not None and run_count >= int(max_runs):
            break
    result = _summarize_rows(
        rows,
        {
            "base_config_path": base_config_path,
            "split_seeds": [int(value) for value in split_seeds],
            "model_seeds": [int(value) for value in model_seeds],
            "epochs_override": epochs_override,
            "patience_override": patience_override,
            "max_runs": max_runs,
            "skip_completed": bool(skip_completed),
            "interpretation": (
                "Repeated split OOS runner for the research/ mainline. Full use should train several model seeds per "
                "split seed, then run risk_adapter/meta selection within each split. Smoke runs only validate plumbing."
            ),
        },
    )
    (output_path / "mainline_repeated_split_oof_v2_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _build_config(
    base_config: dict[str, Any],
    *,
    split_seed: int,
    model_seed: int,
    output_dir: Path,
    epochs_override: int | None,
    patience_override: int | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["seed"] = int(model_seed)
    config.setdefault("train", {})["split_seed"] = int(split_seed)
    if epochs_override is not None:
        config["train"]["epochs"] = int(epochs_override)
    if patience_override is not None:
        config["train"]["early_stop_patience"] = int(patience_override)
    config["paths"]["output_dir"] = str(output_dir.as_posix())
    return config


def _summarize_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No repeated split rows were produced.")
    c_indices = [float(row["test_c_index"]) for row in rows]
    by_split: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(int(row["split_seed"]), []).append(row)
    return {
        **metadata,
        "num_runs": len(rows),
        "mean_test_c_index": statistics.mean(c_indices),
        "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
        "min_test_c_index": min(c_indices),
        "max_test_c_index": max(c_indices),
        "runs": rows,
        "split_summaries": {
            str(split_seed): {
                "num_model_seeds": len(split_rows),
                "mean_test_c_index": statistics.mean([float(row["test_c_index"]) for row in split_rows]),
            }
            for split_seed, split_rows in sorted(by_split.items())
        },
    }


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--split-seeds", default="42,43,44,45,46")
    parser.add_argument("--model-seeds", default="7,21,42,123,2026")
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/mainline_repeated_split_oof_v2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--patience-override", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    run_mainline_repeated_split_oof(
        base_config_path=args.config,
        split_seeds=_parse_ints(args.split_seeds),
        model_seeds=_parse_ints(args.model_seeds),
        output_root=args.output_root,
        device=args.device,
        epochs_override=args.epochs_override,
        patience_override=args.patience_override,
        max_runs=args.max_runs,
        skip_completed=not args.rerun_completed,
    )


if __name__ == "__main__":
    main()
