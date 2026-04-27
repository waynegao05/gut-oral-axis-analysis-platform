from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def perturb_graph_dataframe(graph_df: pd.DataFrame, variant: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_groups = []

    for _, sample_graph in graph_df.groupby("sample_id"):
        sg = sample_graph.copy().reset_index(drop=True)
        node_names = sg["node_name"].drop_duplicates().tolist()
        num_edges = len(sg)

        if variant == "original":
            out_groups.append(sg)
            continue

        if variant == "shuffle_weights":
            shuffled = sg.copy()
            shuffled["edge_weight"] = rng.permutation(shuffled["edge_weight"].to_numpy())
            out_groups.append(shuffled)
            continue

        if variant in {"shuffle_edges", "shuffle_edges_and_weights"}:
            new_edges = []
            tries = 0
            max_tries = max(100, num_edges * 20)
            while len(new_edges) < num_edges and tries < max_tries:
                src = rng.choice(node_names)
                dst = rng.choice(node_names)
                tries += 1
                if src == dst:
                    continue
                new_edges.append((src, dst))
            while len(new_edges) < num_edges:
                src = node_names[len(new_edges) % len(node_names)]
                dst = node_names[(len(new_edges) + 1) % len(node_names)]
                if src != dst:
                    new_edges.append((src, dst))

            shuffled = sg.copy()
            shuffled["src"] = [e[0] for e in new_edges]
            shuffled["dst"] = [e[1] for e in new_edges]
            if variant == "shuffle_edges_and_weights":
                shuffled["edge_weight"] = rng.permutation(shuffled["edge_weight"].to_numpy())
            out_groups.append(shuffled)
            continue

        raise ValueError(f"Unsupported variant: {variant}")

    return pd.concat(out_groups, ignore_index=True)


def _aggregate_variant_runs(summary: Dict[str, List[dict]]) -> dict:
    aggregate = {"variants": {}}
    for variant, runs in summary.items():
        if not runs:
            continue
        runs = sorted(runs, key=lambda item: item["seed"])
        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        aggregate["variants"][variant] = {
            "runs": runs,
            "mean_test_c_index": float(np.mean(c_indices)),
            "std_test_c_index": float(np.std(c_indices, ddof=1)) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": float(np.min(c_indices)),
            "max_test_c_index": float(np.max(c_indices)),
            "mean_test_loss": float(np.mean(losses)),
        }
    return aggregate


def summarize_existing_graph_structure_tests(
    config_path: str,
    seeds: List[int],
    variants: List[str],
    split_seed: int | None,
    output_root: str,
) -> dict:
    output_root_path = Path(output_root)
    summary: Dict[str, List[dict]] = {variant: [] for variant in variants}

    for variant in variants:
        for metrics_path in sorted(output_root_path.glob(f"{variant}_seed*/test_metrics.json")):
            seed = int(metrics_path.parent.name.rsplit("seed", 1)[1])
            if seed not in seeds:
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary[variant].append(
                {
                    "seed": seed,
                    "test_loss": metrics["loss"],
                    "test_c_index": metrics["c_index"],
                    "graph_aux_loss": metrics.get("graph_aux_loss", 0.0),
                    "node_aux_loss": metrics.get("node_aux_loss", 0.0),
                    "output_dir": str(metrics_path.parent.as_posix()),
                }
            )

    aggregate = {
        "config_path": config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "output_root": output_root,
        **_aggregate_variant_runs(summary),
    }

    out_path = output_root_path / "graph_structure_tests_summary.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return aggregate


def run_graph_structure_tests(
    config_path: str,
    seeds: List[int],
    variants: List[str],
    device: str,
    split_seed: int | None = None,
    output_root: str = "outputs/current_mainline_v2/graph_structure_tests",
) -> dict:
    base_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base_graph_df = pd.read_csv(base_config["paths"]["graph_csv"])

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, List[dict]] = {variant: [] for variant in variants}

    for variant in variants:
        for seed in seeds:
            graph_variant_path = output_root_path / f"graph_{variant}_seed{seed}.csv"
            perturbed_df = perturb_graph_dataframe(base_graph_df, variant=variant, seed=seed)
            perturbed_df.to_csv(graph_variant_path, index=False)

            config = copy.deepcopy(base_config)
            config["seed"] = seed
            if split_seed is not None:
                config.setdefault("train", {})
                config["train"]["split_seed"] = split_seed
            config["paths"]["graph_csv"] = str(graph_variant_path).replace("\\", "/")
            config["paths"]["output_dir"] = str((output_root_path / f"{variant}_seed{seed}").as_posix())

            temp_config_path = output_root_path / f"config_{variant}_seed{seed}.yaml"
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
            summary[variant].append(
                {
                    "seed": seed,
                    "test_loss": metrics["loss"],
                    "test_c_index": metrics["c_index"],
                    "graph_aux_loss": metrics.get("graph_aux_loss", 0.0),
                    "node_aux_loss": metrics.get("node_aux_loss", 0.0),
                    "output_dir": config["paths"]["output_dir"],
                }
            )

    variant_summary = _aggregate_variant_runs(summary)
    aggregate = {
        "config_path": config_path,
        "seeds": seeds,
        "split_seed": split_seed,
        "output_root": output_root,
        **variant_summary,
    }

    out_path = output_root_path / "graph_structure_tests_summary.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original", "shuffle_weights", "shuffle_edges", "shuffle_edges_and_weights"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/graph_structure_tests")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Only rebuild graph_structure_tests_summary.json from existing variant seed outputs.",
    )
    args = parser.parse_args()

    if args.summarize_existing:
        summarize_existing_graph_structure_tests(
            args.config,
            args.seeds,
            args.variants,
            args.split_seed,
            args.output_root,
        )
        return

    run_graph_structure_tests(
        args.config,
        args.seeds,
        args.variants,
        args.device,
        split_seed=args.split_seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
