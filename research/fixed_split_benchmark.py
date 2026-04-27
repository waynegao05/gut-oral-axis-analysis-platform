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


def _extract_mainline_row(summary: dict, model_name: str) -> dict:
    return {
        "model_name": model_name,
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


def _extract_ensemble_row(summary: dict, source_path: Path) -> dict:
    return {
        "model_name": "gnn_5seed_ensemble",
        "model_family": "gnn_survival_ensemble",
        "source_summary": str(source_path.as_posix()),
        "mean_test_c_index": summary["ensemble_c_index"],
        "std_test_c_index": None,
        "mean_test_loss": None,
        "num_models": summary.get("num_models"),
        "mean_member_c_index": summary.get("mean_member_c_index"),
    }


def _print_console_summary(result: dict, out_path: Path) -> None:
    console_summary = {
        "summary_path": str(out_path.as_posix()),
        "ranking_table": result["ranking_table"],
        "components": result["components"],
    }
    print(json.dumps(console_summary, indent=2))


def summarize_component_paths(
    config_path: str,
    seeds: list[int],
    split_seed: int,
    device: str,
    output_root: str,
    mainline_summary_path: Path,
    tabular_summary_path: Path,
    graph_summary_path: Path,
    ensemble_summary_path: Path | None = None,
    mainline_model_name: str = "conservative_gnn_mainline",
) -> dict:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    mainline_summary = _load_json(mainline_summary_path)
    tabular_summary = _load_json(tabular_summary_path)
    graph_summary = _load_json(graph_summary_path)

    ranking_table = [
        _extract_mainline_row(mainline_summary, mainline_model_name),
        _extract_baseline_row(tabular_summary, "all_tabular_mlp_cox", "tabular_survival_baseline"),
        _extract_baseline_row(graph_summary, "graph_summary_mlp_cox", "graph_summary_survival_baseline"),
    ]

    ensemble_summary = None
    if ensemble_summary_path is not None and ensemble_summary_path.exists():
        ensemble_summary = _load_json(ensemble_summary_path)
        ranking_table.append(_extract_ensemble_row(ensemble_summary, ensemble_summary_path))

    ranking_table.sort(key=lambda row: row["mean_test_c_index"], reverse=True)

    result = {
        "config_path": config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "device": device,
        "output_root": output_root,
        "components": {
            "mainline_summary": str(mainline_summary_path.as_posix()),
            "tabular_baseline_summary": str(tabular_summary_path.as_posix()),
            "graph_baseline_summary": str(graph_summary_path.as_posix()),
            "ensemble_summary": str(ensemble_summary_path.as_posix()) if ensemble_summary is not None else None,
        },
        "ranking_table": ranking_table,
        "mainline": mainline_summary,
        "tabular_baseline": tabular_summary,
        "graph_baseline": graph_summary,
        "ensemble": ensemble_summary,
    }

    out_path = output_root_path / "fixed_split_benchmark_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_console_summary(result, out_path)
    return result


def summarize_fixed_split_outputs(
    config_path: str,
    seeds: list[int],
    split_seed: int,
    device: str,
    output_root: str,
    mainline_summary_path: str | None = None,
    tabular_summary_path: str | None = None,
    graph_summary_path: str | None = None,
    ensemble_summary_path: str | None = None,
    mainline_model_name: str = "conservative_gnn_mainline",
) -> dict:
    output_root_path = Path(output_root)
    mainline_root = output_root_path / "mainline"
    tabular_root = output_root_path / "tabular_baseline"
    graph_root = output_root_path / "graph_baseline"
    ensemble_output = output_root_path / "ensemble_test_summary.json"
    resolved_mainline_summary_path = Path(mainline_summary_path) if mainline_summary_path is not None else mainline_root / "research_repeat_runs_summary.json"
    resolved_tabular_summary_path = Path(tabular_summary_path) if tabular_summary_path is not None else tabular_root / "baseline_compare_summary.json"
    resolved_graph_summary_path = Path(graph_summary_path) if graph_summary_path is not None else graph_root / "graph_specific_baselines_summary.json"
    resolved_ensemble_summary_path = Path(ensemble_summary_path) if ensemble_summary_path is not None else ensemble_output

    return summarize_component_paths(
        config_path=config_path,
        seeds=seeds,
        split_seed=split_seed,
        device=device,
        output_root=output_root,
        mainline_summary_path=resolved_mainline_summary_path,
        tabular_summary_path=resolved_tabular_summary_path,
        graph_summary_path=resolved_graph_summary_path,
        ensemble_summary_path=resolved_ensemble_summary_path,
        mainline_model_name=mainline_model_name,
    )


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
            "--device",
            device,
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
            "--device",
            device,
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

    return summarize_fixed_split_outputs(
        config_path=config_path,
        seeds=seeds,
        split_seed=split_seed,
        device=device,
        output_root=output_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/fixed_split_benchmark")
    parser.add_argument("--run-ensemble", action="store_true")
    parser.add_argument("--mainline-summary-path", default=None)
    parser.add_argument("--tabular-summary-path", default=None)
    parser.add_argument("--graph-summary-path", default=None)
    parser.add_argument("--ensemble-summary-path", default=None)
    parser.add_argument("--mainline-model-name", default="conservative_gnn_mainline")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Only rebuild fixed_split_benchmark_summary.json from existing component summaries.",
    )
    args = parser.parse_args()

    if args.summarize_existing:
        summarize_fixed_split_outputs(
            config_path=args.config,
            seeds=args.seeds,
            split_seed=args.split_seed,
            device=args.device,
            output_root=args.output_root,
            mainline_summary_path=args.mainline_summary_path,
            tabular_summary_path=args.tabular_summary_path,
            graph_summary_path=args.graph_summary_path,
            ensemble_summary_path=args.ensemble_summary_path,
            mainline_model_name=args.mainline_model_name,
        )
        return

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
