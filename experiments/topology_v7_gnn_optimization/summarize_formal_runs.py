from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_ROOT = Path(
    "outputs/current_mainline_v3/topology_v7_generator_v2/gnn_identity_fullrisk_v1"
)
SPLIT_DIRS = ("split42_five_seed", "split43_five_seed")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output_root: Path) -> dict:
    split_results = []
    all_runs = []

    for split_dir in SPLIT_DIRS:
        split_root = output_root / split_dir
        repeated = read_json(split_root / "research_repeat_runs_summary.json")
        ensemble = read_json(split_root / "ensemble_summary.json")
        runs = repeated["runs"]
        all_runs.extend(runs)
        split_results.append(
            {
                "split_seed": repeated["split_seed"],
                "num_seeds": len(runs),
                "seeds": repeated["seeds"],
                "mean_test_c_index": repeated["mean_test_c_index"],
                "std_test_c_index": repeated["std_test_c_index"],
                "min_test_c_index": repeated["min_test_c_index"],
                "max_test_c_index": repeated["max_test_c_index"],
                "mean_test_cohort_loss": repeated["mean_test_loss"],
                "ensemble_c_index": ensemble["ensemble_c_index"],
                "ensemble_gain_over_member_mean": (
                    ensemble["ensemble_c_index"] - repeated["mean_test_c_index"]
                ),
            }
        )

    c_indices = [run["test_c_index"] for run in all_runs]
    losses = [run["test_loss"] for run in all_runs]
    training_seconds = [run["training_seconds"] for run in all_runs]
    seed_means = {}
    for seed in sorted({run["seed"] for run in all_runs}):
        values = [run["test_c_index"] for run in all_runs if run["seed"] == seed]
        seed_means[str(seed)] = statistics.mean(values)

    first_summary = read_json(
        output_root
        / SPLIT_DIRS[0]
        / f"research_seed{all_runs[0]['seed']}"
        / "training_summary.json"
    )
    return {
        "status": "complete",
        "protocol": {
            "dataset": "topology_v7_generator_v2",
            "split_seeds": [item["split_seed"] for item in split_results],
            "training_seeds": sorted({run["seed"] for run in all_runs}),
            "num_runs": len(all_runs),
            "selection_rule": "Hyperparameters selected on validation metrics only.",
            "ensemble_rule": "Equal-weight mean of five independently seeded risk scores.",
        },
        "model": {
            "parameter_count": first_summary["parameter_count"],
            "cox_ties_method": first_summary["cox_ties_method"],
            "node_identity_dim": first_summary["node_identity_dim"],
            "identity_readout_dim": first_summary["identity_readout_dim"],
            "pool_every_layer": first_summary["pool_every_layer"],
            "graph_projection_dim": first_summary["graph_projection_dim"],
            "tabular_projection_dim": first_summary["tabular_projection_dim"],
        },
        "split_results": split_results,
        "aggregate_independent_runs": {
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices),
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_cohort_loss": statistics.mean(losses),
            "mean_training_seconds": statistics.mean(training_seconds),
            "total_training_seconds": sum(training_seconds),
            "mean_test_c_index_by_training_seed_across_splits": seed_means,
        },
    }


def render_markdown(summary: dict) -> str:
    aggregate = summary["aggregate_independent_runs"]
    lines = [
        "# V7 GNN Formal Retraining Summary",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{summary['protocol']['dataset']}`",
        f"- Independent runs: {summary['protocol']['num_runs']}",
        f"- Split seeds: {summary['protocol']['split_seeds']}",
        f"- Training seeds: {summary['protocol']['training_seeds']}",
        "- Hyperparameters were selected using validation results only.",
        "",
        "## Results",
        "",
        "| Split seed | Mean test C-index | SD | Mean cohort loss | Ensemble C-index |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["split_results"]:
        lines.append(
            f"| {item['split_seed']} | {item['mean_test_c_index']:.6f} | "
            f"{item['std_test_c_index']:.6f} | {item['mean_test_cohort_loss']:.6f} | "
            f"{item['ensemble_c_index']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Ten-run mean test C-index: {aggregate['mean_test_c_index']:.6f}",
            f"- Ten-run SD: {aggregate['std_test_c_index']:.6f}",
            f"- Ten-run range: {aggregate['min_test_c_index']:.6f} to "
            f"{aggregate['max_test_c_index']:.6f}",
            f"- Mean cohort loss: {aggregate['mean_test_cohort_loss']:.6f}",
            "",
            "The two split results should be reported separately because they use different "
            "held-out groups; the ten-run mean is a compact secondary summary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    summary = summarize(args.output_root)
    json_path = args.output_root / "formal_retraining_summary.json"
    markdown_path = args.output_root / "formal_retraining_report.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
