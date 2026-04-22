from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_mainline_row(summary: dict) -> dict:
    return {
        "model_name": "conservative_gnn_mainline",
        "model_family": "gnn_survival",
        "source_summary": summary["output_root"],
        "mean_test_c_index": summary["mean_test_c_index"],
        "std_test_c_index": summary["std_test_c_index"],
        "mean_test_loss": summary["mean_test_loss"],
    }


def _extract_baseline_row(summary: dict, baseline_name: str, family: str) -> dict:
    baseline_summary = summary["baselines"][baseline_name]
    return {
        "model_name": baseline_name,
        "model_family": family,
        "source_summary": summary["output_root"],
        "mean_test_c_index": baseline_summary["mean_test_c_index"],
        "std_test_c_index": baseline_summary["std_test_c_index"],
        "mean_test_loss": baseline_summary["mean_test_loss"],
    }


def run_fixed_split_benchmark(
    config_path: str,
    seeds: list[int],
    split_seed: int,
    device: str,
    output_root: str = "outputs/current_mainline_v2/fixed_split_benchmark",
    run_ensemble: bool = False,
) -> dict:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    mainline_root = output_root_path / "mainline"
    tabular_root = output_root_path / "tabular_baseline"
    graph_root = output_root_path / "graph_baseline"
    ensemble_output = output_root_path / "ensemble_test_summary.json"

    _run(
        [
            sys.executable,
            "-m",
            "research.repeat_runs_v2",
            "--config",
            config_path,
            "--seeds",
            *[str(seed) for seed in seeds],
            "--split-seed",
            str(split_seed),
            "--output-root",
            str(mainline_root),
            "--device",
            device,
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "research.baseline_compare",
            "--config",
            config_path,
            "--seeds",
            *[str(seed) for seed in seeds],
            "--split-seed",
            str(split_seed),
            "--only",
            "all_tabular_mlp_cox",
            "--output-root",
            str(tabular_root),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "research.graph_specific_baselines",
            "--config",
            config_path,
            "--seeds",
            *[str(seed) for seed in seeds],
            "--split-seed",
            str(split_seed),
            "--only",
            "graph_summary_mlp_cox",
            "--output-root",
            str(graph_root),
        ]
    )

    if run_ensemble:
        _run(
            [
                sys.executable,
                "-m",
                "research.ensemble_v2",
                "--config",
                config_path,
                "--checkpoint-glob",
                str((mainline_root / "research_seed*" / "best_model.pt").as_posix()),
                "--split",
                "test",
                "--device",
                device,
                "--split-seed",
                str(split_seed),
                "--output",
                str(ensemble_output),
            ]
        )

    mainline_summary = _load_json(mainline_root / "research_repeat_runs_summary.json")
    tabular_summary = _load_json(tabular_root / "baseline_compare_summary.json")
    graph_summary = _load_json(graph_root / "graph_specific_baselines_summary.json")

    ranking_table = [
        _extract_mainline_row(mainline_summary),
        _extract_baseline_row(tabular_summary, "all_tabular_mlp_cox", "tabular_survival_baseline"),
        _extract_baseline_row(graph_summary, "graph_summary_mlp_cox", "graph_summary_survival_baseline"),
    ]

    ensemble_summary = None
    if run_ensemble and ensemble_output.exists():
        ensemble_summary = _load_json(ensemble_output)
        ranking_table.append(
            {
                "model_name": "mainline_checkpoint_ensemble",
                "model_family": "gnn_ensemble",
                "source_summary": str(ensemble_output.as_posix()),
                "mean_test_c_index": ensemble_summary["ensemble_c_index"],
                "std_test_c_index": 0.0,
                "mean_test_loss": None,
            }
        )

    ranking_table.sort(key=lambda row: row["mean_test_c_index"], reverse=True)

    result = {
        "config_path": config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "device": device,
        "output_root": output_root,
        "components": {
            "mainline_summary": str((mainline_root / "research_repeat_runs_summary.json").as_posix()),
            "tabular_baseline_summary": str((tabular_root / "baseline_compare_summary.json").as_posix()),
            "graph_baseline_summary": str((graph_root / "graph_specific_baselines_summary.json").as_posix()),
            "ensemble_summary": str(ensemble_output.as_posix()) if ensemble_summary is not None else None,
        },
        "ranking_table": ranking_table,
        "mainline": mainline_summary,
        "tabular_baseline": tabular_summary,
        "graph_baseline": graph_summary,
        "ensemble": ensemble_summary,
    }

    out_path = output_root_path / "fixed_split_benchmark_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/fixed_split_benchmark")
    parser.add_argument("--run-ensemble", action="store_true")
    args = parser.parse_args()

    run_fixed_split_benchmark(
        config_path=args.config,
        seeds=args.seeds,
        split_seed=args.split_seed,
        device=args.device,
        output_root=args.output_root,
        run_ensemble=args.run_ensemble,
    )


if __name__ == "__main__":
    main()
