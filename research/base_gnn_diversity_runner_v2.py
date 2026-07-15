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


VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "description": "Unchanged research_config_v2 baseline.",
        "train": {},
        "graph_preprocess": {},
    },
    "ranking_w0p02": {
        "description": "Small Cox pair-ranking pressure; intended to improve ordering without replacing Cox.",
        "train": {"ranking_weight": 0.02, "ranking_margin": 0.0, "ranking_warmup_epochs": 8},
        "graph_preprocess": {},
    },
    "ranking_w0p04": {
        "description": "Stronger Cox pair-ranking pressure; kept separate so validation can reject it.",
        "train": {"ranking_weight": 0.04, "ranking_margin": 0.0, "ranking_warmup_epochs": 10},
        "graph_preprocess": {},
    },
    "cox_batch16": {
        "description": "Double the mini-batch risk set for a less noisy Cox partial-likelihood estimate.",
        "train": {"batch_size": 16},
        "graph_preprocess": {},
    },
    "cox_batch32": {
        "description": "Use a larger in-batch Cox risk set; kept separate from ranking-loss changes.",
        "train": {"batch_size": 32},
        "graph_preprocess": {},
    },
    "cox_batch32_ranking_w0p02": {
        "description": "Add the previously promising 0.02 ranking term after enlarging the Cox risk set.",
        "train": {
            "batch_size": 32,
            "ranking_weight": 0.02,
            "ranking_margin": 0.0,
            "ranking_warmup_epochs": 8,
        },
        "graph_preprocess": {},
    },
    "tabular_standardized": {
        "description": (
            "Z-score clinical and metabolite features using training-split statistics before GNN fusion."
        ),
        "train": {},
        "graph_preprocess": {},
        "tabular_preprocess": {"standardize": True},
    },
    "tabular_standardized_ranking_w0p02": {
        "description": (
            "Combine train-only tabular z-scoring with the previously validated 0.02 pair-ranking pressure."
        ),
        "train": {"ranking_weight": 0.02, "ranking_margin": 0.0, "ranking_warmup_epochs": 8},
        "graph_preprocess": {},
        "tabular_preprocess": {"standardize": True},
    },
    "topk8": {
        "description": "Restrict each graph to its top 8 weighted edges.",
        "train": {},
        "graph_preprocess": {"keep_top_k_edges": 8, "min_edge_weight": None},
    },
    "topk6": {
        "description": "More aggressive top-k edge denoising.",
        "train": {},
        "graph_preprocess": {"keep_top_k_edges": 6, "min_edge_weight": None},
    },
    "minw03": {
        "description": "Drop weak topology edges below weight 0.3.",
        "train": {},
        "graph_preprocess": {"keep_top_k_edges": None, "min_edge_weight": 0.3},
    },
    "topk8_ranking_w0p02": {
        "description": "Combine mild edge denoising with mild pair-ranking pressure.",
        "train": {"ranking_weight": 0.02, "ranking_margin": 0.0, "ranking_warmup_epochs": 8},
        "graph_preprocess": {"keep_top_k_edges": 8, "min_edge_weight": None},
    },
    "dropout0p20": {
        "description": "Slightly lower dropout to test under-regularization/fit capacity.",
        "train": {"dropout": 0.20},
        "graph_preprocess": {},
    },
    "dropout0p35": {
        "description": "Higher dropout to create a more regularized base learner.",
        "train": {"dropout": 0.35},
        "graph_preprocess": {},
    },
    "aux_light": {
        "description": "Reduce structure auxiliary losses so survival ordering dominates.",
        "train": {"graph_aux_weight": 0.04, "node_aux_weight": 0.025},
        "graph_preprocess": {},
    },
    "aux_heavy": {
        "description": "Increase structure auxiliary losses to test whether stronger mechanistic pressure helps.",
        "train": {"graph_aux_weight": 0.12, "node_aux_weight": 0.075},
        "graph_preprocess": {},
    },
}

DEFAULT_VARIANTS = ["baseline", "ranking_w0p02", "topk8", "minw03", "aux_light"]


def run_base_gnn_diversity_sweep(
    *,
    base_config_path: str = "research_config_v2.yaml",
    variants: Sequence[str] = tuple(DEFAULT_VARIANTS),
    seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    split_seed: int = 42,
    output_root: str | Path = "outputs/current_mainline_v2/base_gnn_diversity_v2",
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
    for variant in variants:
        if variant not in VARIANT_PRESETS:
            raise ValueError(f"Unsupported variant: {variant}. Available variants: {sorted(VARIANT_PRESETS)}")
        for seed in seeds:
            if max_runs is not None and run_count >= int(max_runs):
                break
            run_dir = output_path / str(variant) / f"seed_{int(seed)}"
            config = _build_config(
                base_config,
                variant=str(variant),
                seed=int(seed),
                split_seed=int(split_seed),
                output_dir=run_dir,
                epochs_override=epochs_override,
                patience_override=patience_override,
            )
            config_path = output_path / "configs" / f"{variant}_seed{int(seed)}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            metrics_path = run_dir / "test_metrics.json"
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
            rows.append(
                {
                    **_load_run_metrics(run_dir),
                    "variant": str(variant),
                    "seed": int(seed),
                    "split_seed": int(split_seed),
                    "output_dir": str(run_dir.as_posix()),
                    "config_path": str(config_path.as_posix()),
                    "variant_overrides": {
                        "train": VARIANT_PRESETS[str(variant)]["train"],
                        "graph_preprocess": VARIANT_PRESETS[str(variant)]["graph_preprocess"],
                        "tabular_preprocess": VARIANT_PRESETS[str(variant)].get("tabular_preprocess", {}),
                    },
                    "variant_description": VARIANT_PRESETS[str(variant)]["description"],
                }
            )
            run_count += 1
        if max_runs is not None and run_count >= int(max_runs):
            break

    result = _summarize_rows(
        rows,
        {
            "base_config_path": base_config_path,
            "variants": [str(value) for value in variants],
            "seeds": [int(value) for value in seeds],
            "split_seed": int(split_seed),
            "epochs_override": epochs_override,
            "patience_override": patience_override,
            "max_runs": max_runs,
            "skip_completed": bool(skip_completed),
            "available_variants": VARIANT_PRESETS,
            "interpretation": (
                "Base GNN diversity v2 trains deliberately different but shallow research/ base learners. "
                "The point is not to stack depth; only variants with validation/test complementarity should be passed "
                "to downstream OOF/meta selection."
            ),
        },
    )
    (output_path / "base_gnn_diversity_v2_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _build_config(
    base_config: dict[str, Any],
    *,
    variant: str,
    seed: int,
    split_seed: int,
    output_dir: Path,
    epochs_override: int | None,
    patience_override: int | None,
) -> dict[str, Any]:
    if variant not in VARIANT_PRESETS:
        raise ValueError(f"Unsupported variant: {variant}. Available variants: {sorted(VARIANT_PRESETS)}")
    config = copy.deepcopy(base_config)
    preset = VARIANT_PRESETS[variant]
    config["seed"] = int(seed)
    config.setdefault("train", {})["split_seed"] = int(split_seed)
    config.setdefault("graph_preprocess", {})
    config.setdefault("tabular_preprocess", {})
    _deep_update(config["train"], preset.get("train", {}))
    _deep_update(config["graph_preprocess"], preset.get("graph_preprocess", {}))
    _deep_update(config["tabular_preprocess"], preset.get("tabular_preprocess", {}))
    if epochs_override is not None:
        config["train"]["epochs"] = int(epochs_override)
    if patience_override is not None:
        config["train"]["early_stop_patience"] = int(patience_override)
    config["paths"]["output_dir"] = str(output_dir.as_posix())
    config.setdefault("experiment", {})
    config["experiment"]["base_gnn_diversity_variant"] = variant
    config["experiment"]["variant_description"] = str(preset["description"])
    return config


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _load_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    val_c_indices = [float(row["c_index"]) for row in history if "c_index" in row]
    best_val_c_index = max(val_c_indices) if val_c_indices else None
    best_epoch = None
    if best_val_c_index is not None:
        for row in history:
            if float(row.get("c_index", float("nan"))) == best_val_c_index:
                best_epoch = int(row["epoch"])
                break
    return {
        "test_c_index": float(metrics["c_index"]),
        "test_loss": float(metrics["loss"]),
        "test_cohort_loss": (
            float(metrics["cohort_loss"])
            if metrics.get("cohort_loss") is not None
            else None
        ),
        "test_cohort_cox_loss": (
            float(metrics["cohort_cox_loss"])
            if metrics.get("cohort_cox_loss") is not None
            else None
        ),
        "test_ranking_loss": float(metrics.get("ranking_loss", 0.0)),
        "test_graph_aux_loss": float(metrics.get("graph_aux_loss", 0.0)),
        "test_node_aux_loss": float(metrics.get("node_aux_loss", 0.0)),
        "best_val_c_index": best_val_c_index,
        "best_epoch": best_epoch,
    }


def _summarize_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No base GNN diversity rows were produced.")
    c_indices = [float(row["test_c_index"]) for row in rows]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = {
        variant: _summarize_variant_rows(variant_rows)
        for variant, variant_rows in sorted(by_variant.items())
    }
    ranking_table = [
        {
            "variant": variant,
            "num_runs": summary["num_runs"],
            "mean_test_c_index": summary["mean_test_c_index"],
            "std_test_c_index": summary["std_test_c_index"],
            "max_test_c_index": summary["max_test_c_index"],
            "mean_best_val_c_index": summary["mean_best_val_c_index"],
            "mean_test_cohort_loss": summary["mean_test_cohort_loss"],
        }
        for variant, summary in variant_summaries.items()
    ]
    ranking_table.sort(key=lambda row: (row["mean_test_c_index"], row["max_test_c_index"]), reverse=True)
    return {
        **metadata,
        "num_runs": len(rows),
        "mean_test_c_index": statistics.mean(c_indices),
        "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
        "min_test_c_index": min(c_indices),
        "max_test_c_index": max(c_indices),
        "runs": rows,
        "variant_summaries": variant_summaries,
        "ranking_table": ranking_table,
    }


def _summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c_indices = [float(row["test_c_index"]) for row in rows]
    val_values = [float(row["best_val_c_index"]) for row in rows if row.get("best_val_c_index") is not None]
    cohort_loss_values = [
        float(row["test_cohort_loss"])
        for row in rows
        if row.get("test_cohort_loss") is not None
    ]
    return {
        "num_runs": len(rows),
        "mean_test_c_index": statistics.mean(c_indices),
        "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
        "min_test_c_index": min(c_indices),
        "max_test_c_index": max(c_indices),
        "mean_best_val_c_index": statistics.mean(val_values) if val_values else None,
        "mean_test_cohort_loss": statistics.mean(cohort_loss_values) if cohort_loss_values else None,
        "runs": rows,
    }


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default="7,21,42,123,2026")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/base_gnn_diversity_v2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--patience-override", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    run_base_gnn_diversity_sweep(
        base_config_path=args.config,
        variants=_parse_strings(args.variants),
        seeds=_parse_ints(args.seeds),
        split_seed=args.split_seed,
        output_root=args.output_root,
        device=args.device,
        epochs_override=args.epochs_override,
        patience_override=args.patience_override,
        max_runs=args.max_runs,
        skip_completed=not args.rerun_completed,
    )


if __name__ == "__main__":
    main()
