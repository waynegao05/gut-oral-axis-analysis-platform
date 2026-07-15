from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from research.main_model_ctm_residual_v2 import export_main_model_structured_features
from research.main_model_meta_oof_v2 import SELECTION_POLICIES, run_main_model_meta_oof_experiment
from research.mainline_repeated_split_oof_v2 import run_mainline_repeated_split_oof
from research.structured_feature_export_v2 import export_structured_gnn_features


def run_mainline_split_meta_validation(
    *,
    base_config_path: str = "research_config_v2.yaml",
    split_seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    output_root: str | Path = "outputs/current_mainline_v2/mainline_split_meta_validation_v2",
    device: str = "cuda",
    adapter_device: str = "cpu",
    gnn_epochs_override: int | None = None,
    gnn_patience_override: int | None = None,
    meta_seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    meta_model_types: Sequence[str] = ("linear", "mlp"),
    meta_max_deltas: Sequence[float] = (0.03, 0.05, 0.10),
    meta_epochs: int = 80,
    meta_patience: int = 10,
    min_oof_delta: float = 0.00003,
    min_val_delta: float = 0.0,
    min_high_disagreement_val_delta: float = 0.0005,
    selection_policy: str = "hybrid_validation_oof",
    skip_completed: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        split_dir = output_path / f"split_seed_{int(split_seed)}"
        split_dir.mkdir(parents=True, exist_ok=True)
        gnn_root = split_dir / "gnn"
        gnn_summary = run_mainline_repeated_split_oof(
            base_config_path=base_config_path,
            split_seeds=(int(split_seed),),
            model_seeds=[int(value) for value in model_seeds],
            output_root=gnn_root,
            device=device,
            epochs_override=gnn_epochs_override,
            patience_override=gnn_patience_override,
            skip_completed=skip_completed,
        )
        checkpoint_glob = _checkpoint_glob(gnn_root, int(split_seed))
        structured_dir = split_dir / "structured_features"
        structured_npz = structured_dir / "structured_gnn_features_v2_all_splits.npz"
        structured_json = structured_dir / "structured_gnn_features_v2_summary.json"
        if not (skip_completed and structured_npz.exists() and structured_json.exists()):
            export_structured_gnn_features(
                config_path=base_config_path,
                checkpoint_glob=checkpoint_glob,
                split_seed=int(split_seed),
                device_arg=device,
                top_k=3,
                output_npz_path=structured_npz,
                output_json_path=structured_json,
            )
        main_feature_dir = split_dir / "main_model_features"
        main_feature_npz = main_feature_dir / "main_model_structured_features.npz"
        main_feature_json = main_feature_dir / "main_model_structured_features_summary.json"
        if not (skip_completed and main_feature_npz.exists() and main_feature_json.exists()):
            export_main_model_structured_features(
                config_path=base_config_path,
                checkpoint_glob=checkpoint_glob,
                structured_feature_npz=str(structured_npz),
                split_seed=int(split_seed),
                device_arg=device,
                adapter_device_arg=adapter_device,
                output_npz_path=main_feature_npz,
                output_json_path=main_feature_json,
            )
        meta_dir = split_dir / f"meta_oof_strict_{selection_policy}"
        meta_summary_path = meta_dir / "main_model_meta_oof_v2_summary.json"
        if skip_completed and meta_summary_path.exists():
            meta_summary = json.loads(meta_summary_path.read_text(encoding="utf-8"))
        else:
            meta_summary = run_main_model_meta_oof_experiment(
                feature_npz=main_feature_npz,
                output_dir=meta_dir,
                seeds=[int(value) for value in meta_seeds],
                model_types=[str(value) for value in meta_model_types],
                max_deltas=[float(value) for value in meta_max_deltas],
                inner_folds=5,
                epochs=int(meta_epochs),
                patience=int(meta_patience),
                min_oof_delta=float(min_oof_delta),
                min_val_delta=float(min_val_delta),
                min_high_disagreement_val_delta=float(min_high_disagreement_val_delta),
                selection_policy=selection_policy,
                device_arg=device,
                ensure_features=False,
            )
        selected = meta_summary["selected"]
        references = meta_summary["references"]
        rows.append(
            {
                "split_seed": int(split_seed),
                "gnn_mean_test_c_index": float(gnn_summary["mean_test_c_index"]),
                "gnn_max_test_c_index": float(gnn_summary["max_test_c_index"]),
                "main_model_test_c_index": float(references["test_main_c_index"]),
                "selected_candidate": selected["candidate_name"],
                "selected_alpha": float(selected["alpha"]),
                "selected_oof_delta": float(selected["oof_delta"]),
                "selected_val_delta": float(selected["val_delta"]),
                "selected_test_c_index": float(selected["test_c_index"]),
                "selected_test_delta_vs_main": float(selected["test_delta_vs_main"]),
                "net_corrected_pairs": int(selected["pair_change"]["net_corrected_pairs"]),
                "checkpoint_glob": checkpoint_glob,
                "structured_feature_npz": str(structured_npz.as_posix()),
                "main_feature_npz": str(main_feature_npz.as_posix()),
                "meta_summary_path": str(meta_summary_path.as_posix()),
            }
        )
    result = _summarize_rows(
        rows,
        {
            "base_config_path": base_config_path,
            "split_seeds": [int(value) for value in split_seeds],
            "model_seeds": [int(value) for value in model_seeds],
            "meta_seeds": [int(value) for value in meta_seeds],
            "meta_model_types": [str(value) for value in meta_model_types],
            "meta_max_deltas": [float(value) for value in meta_max_deltas],
            "meta_selection_policy": selection_policy,
            "gnn_epochs_override": gnn_epochs_override,
            "gnn_patience_override": gnn_patience_override,
            "meta_epochs": int(meta_epochs),
            "meta_patience": int(meta_patience),
            "min_oof_delta": float(min_oof_delta),
            "min_val_delta": float(min_val_delta),
            "min_high_disagreement_val_delta": float(min_high_disagreement_val_delta),
            "skip_completed": bool(skip_completed),
            "interpretation": (
                "Split-level validation retrains the fixed-split GNN seed ensemble per split seed, rebuilds the "
                "current main-model structured features, and reruns strict OOF meta selection. This is the stability "
                "check for whether the fixed split champion generalizes beyond split_seed=42."
            ),
        },
    )
    (output_path / "mainline_split_meta_validation_v2_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def _checkpoint_glob(gnn_root: Path, split_seed: int) -> str:
    return str((gnn_root / f"split_seed_{int(split_seed)}" / "model_seed_*" / "best_model.pt").as_posix())


def _summarize_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No split validation rows were produced.")
    deltas = [float(row["selected_test_delta_vs_main"]) for row in rows]
    return {
        **metadata,
        "num_splits": len(rows),
        "mean_selected_delta_vs_main": sum(deltas) / len(deltas),
        "num_positive_selected_deltas": sum(1 for value in deltas if value > 0.0),
        "rows": rows,
    }


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--split-seeds", default="42,43,44,45,46")
    parser.add_argument("--model-seeds", default="7,21,42,123,2026")
    parser.add_argument("--output-root", default="outputs/current_mainline_v2/mainline_split_meta_validation_v2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--adapter-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--gnn-epochs-override", type=int)
    parser.add_argument("--gnn-patience-override", type=int)
    parser.add_argument("--meta-seeds", default="7,21,42,123,2026")
    parser.add_argument("--meta-model-types", default="linear,mlp")
    parser.add_argument("--meta-max-deltas", default="0.03,0.05,0.1")
    parser.add_argument("--meta-epochs", type=int, default=80)
    parser.add_argument("--meta-patience", type=int, default=10)
    parser.add_argument("--min-oof-delta", type=float, default=0.00003)
    parser.add_argument("--min-val-delta", type=float, default=0.0)
    parser.add_argument("--min-high-disagreement-val-delta", type=float, default=0.0005)
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="hybrid_validation_oof",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    run_mainline_split_meta_validation(
        base_config_path=args.config,
        split_seeds=_parse_ints(args.split_seeds),
        model_seeds=_parse_ints(args.model_seeds),
        output_root=args.output_root,
        device=args.device,
        adapter_device=args.adapter_device,
        gnn_epochs_override=args.gnn_epochs_override,
        gnn_patience_override=args.gnn_patience_override,
        meta_seeds=_parse_ints(args.meta_seeds),
        meta_model_types=_parse_strings(args.meta_model_types),
        meta_max_deltas=_parse_floats(args.meta_max_deltas),
        meta_epochs=args.meta_epochs,
        meta_patience=args.meta_patience,
        min_oof_delta=args.min_oof_delta,
        min_val_delta=args.min_val_delta,
        min_high_disagreement_val_delta=args.min_high_disagreement_val_delta,
        selection_policy=args.selection_policy,
        skip_completed=not args.rerun_completed,
    )


if __name__ == "__main__":
    main()
