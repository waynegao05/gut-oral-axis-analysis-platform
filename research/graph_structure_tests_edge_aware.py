from __future__ import annotations

import argparse
import copy
import json
import subprocess
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


def run_graph_structure_tests(config_path: str, seeds: List[int], variants: List[str], device: str) -> dict:
    base_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base_graph_df = pd.read_csv(base_config["paths"]["graph_csv"])

    temp_root = Path("outputs/graph_structure_tests_edge_aware")
    temp_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, List[dict]] = {variant: [] for variant in variants}

    for variant in variants:
        for seed in seeds:
            graph_variant_path = temp_root / f"graph_{variant}_seed{seed}.csv"
            perturbed_df = perturb_graph_dataframe(base_graph_df, variant=variant, seed=seed)
            perturbed_df.to_csv(graph_variant_path, index=False)

            config = copy.deepcopy(base_config)
            config["seed"] = seed
            config["paths"]["graph_csv"] = str(graph_variant_path).replace('\\', '/')
            config["paths"]["output_dir"] = f"outputs/graph_structure_tests_edge_aware/{variant}_seed{seed}"

            temp_config_path = temp_root / f"config_{variant}_seed{seed}.yaml"
            temp_config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            cmd = [
                "python",
                "-m",
                "research.train_edge_aware",
                "--config",
                str(temp_config_path),
                "--device",
                device,
            ]
            subprocess.run(cmd, check=True)

            metrics_path = Path(config["paths"]["output_dir"]) / "test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary[variant].append(
                {
                    "seed": seed,
                    "test_loss": metrics["loss"],
                    "test_c_index": metrics["c_index"],
                    "output_dir": config["paths"]["output_dir"],
                }
            )

    aggregate = {"config_path": config_path, "seeds": seeds, "variants": {}}
    for variant, runs in summary.items():
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

    out_path = temp_root / "graph_structure_tests_edge_aware_summary.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_edge_aware.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original", "shuffle_weights", "shuffle_edges", "shuffle_edges_and_weights"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    result = run_graph_structure_tests(args.config, args.seeds, args.variants, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
