from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .topology_aft_fusion import AFT_PRESETS, FEATURE_SETS, run_experiment


DEFAULT_SEEDS = (7, 21, 42, 123, 2026)


def run_seed_sweep(
    *,
    config_path: str,
    mainline_predictions_path: str,
    output_root: str | Path,
    split_seed: int,
    seeds: Sequence[int],
    preset_names: Sequence[str],
    num_boost_round: int,
    early_stopping_rounds: int,
    nthread: int,
    minimum_c_index_delta: float,
    maximum_alpha: float,
    skip_completed: bool = True,
    feature_set: str = "full",
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = root / f"split{int(split_seed)}_seed{int(seed)}"
        summary_path = run_dir / "summary.json"
        if skip_completed and summary_path.exists():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            result = run_experiment(
                config_path=config_path,
                mainline_predictions_path=mainline_predictions_path,
                output_dir=run_dir,
                split_seed=int(split_seed),
                seed=int(seed),
                preset_names=preset_names,
                num_boost_round=int(num_boost_round),
                early_stopping_rounds=int(early_stopping_rounds),
                nthread=int(nthread),
                minimum_c_index_delta=float(minimum_c_index_delta),
                maximum_alpha=float(maximum_alpha),
                emit_json=False,
                feature_set=feature_set,
            )
        row = {
            "split_seed": int(split_seed),
            "model_seed": int(seed),
            "selected_expert": result["selected_expert"]["name"],
            "selected_alpha": float(result["blend_selection"]["selected"]["alpha"]),
            "validation_c_index": float(result["validation"]["selected_c_index"]),
            "reference_test_c_index": float(result["test"]["reference_c_index"]),
            "selected_test_c_index": float(result["test"]["selected_c_index"]),
            "test_c_index_delta": float(result["test"]["selected_c_index_delta"]),
            "test_cox_loss_delta": float(result["test"]["calibrated_cox_loss_delta"]),
            "run_dir": str(run_dir.as_posix()),
        }
        rows.append(row)
        print(json.dumps(row))

    result = {
        "protocol": "temporal_independent_v3_five_seed_sweep",
        "config_path": config_path,
        "mainline_predictions_path": mainline_predictions_path,
        "split_seed": int(split_seed),
        "seeds": [int(seed) for seed in seeds],
        "preset_names": list(preset_names),
        "feature_set": feature_set,
        "runs": rows,
    }
    manifest_path = root / f"split{int(split_seed)}_seed_sweep_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--mainline-predictions", required=True)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/temporal_independent_v3")
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--presets", nargs="+", default=list(AFT_PRESETS))
    parser.add_argument("--num-boost-round", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--nthread", type=int, default=6)
    parser.add_argument("--minimum-c-index-delta", type=float, default=0.0003)
    parser.add_argument("--maximum-alpha", type=float, default=1.0)
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full")
    args = parser.parse_args()
    run_seed_sweep(
        config_path=args.config,
        mainline_predictions_path=args.mainline_predictions,
        output_root=args.output_root,
        split_seed=args.split_seed,
        seeds=args.seeds,
        preset_names=args.presets,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        nthread=args.nthread,
        minimum_c_index_delta=args.minimum_c_index_delta,
        maximum_alpha=args.maximum_alpha,
        skip_completed=not args.no_skip_completed,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()
