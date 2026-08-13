from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from experiments.temporal_independent_v3.seed_ensemble import build_seed_ensemble
from experiments.temporal_independent_v3.seed_sweep import run_seed_sweep
from experiments.temporal_independent_v3.topology_aft_fusion import AFT_PRESETS
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "experiments/topology_v7_generator_v3/fusion_lock.json"
CONFIG_PATH = ROOT / "research_config_v7_v3_gnn_locked.yaml"
GNN_ROOT = ROOT / "outputs/topology_v7_generator_v3_formal/gnn_locked_logo"
OUTPUT_ROOT = ROOT / "outputs/topology_v7_generator_v3_formal/aft_fusion_logo"
SEEDS = [7, 21, 42, 123, 2026]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_lock() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_formal_aft_fusion":
        raise RuntimeError("AFT fusion protocol is not locked.")
    manifest = ROOT / "data/research/topology_v7_generator_v3/topology_v7_manifest.json"
    gnn_summary = GNN_ROOT / "formal_logo_gnn_summary.json"
    if _sha256(manifest) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("Formal dataset manifest changed after the fusion protocol was locked.")
    if _sha256(gnn_summary) != lock["gnn_summary_sha256"]:
        raise RuntimeError("Formal GNN summary changed after the fusion protocol was locked.")
    expected_presets = list(AFT_PRESETS)
    if lock["protocol"]["aft_presets"] != expected_presets:
        raise RuntimeError("AFT preset order differs from the fusion protocol lock.")
    return lock


def _ensemble_json_to_arrays(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["predictions"]
    return {
        "sample_ids": np.asarray([str(row["sample_id"]) for row in rows]),
        "time": np.asarray([float(row["time"]) for row in rows], dtype=float),
        "event": np.asarray([float(row["event"]) for row in rows], dtype=float),
        "selected_risk": np.asarray(
            [float(row["ensemble_risk"]) for row in rows], dtype=float
        ),
    }


def _export_mainline_predictions(fold_root: Path) -> Path:
    splits = {
        name: _ensemble_json_to_arrays(fold_root / f"{name}_ensemble_summary.json")
        for name in ("train", "val", "test")
    }
    ids = [set(values["sample_ids"].tolist()) for values in splits.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("GNN train, validation, and test predictions overlap.")
    output = fold_root / "gnn_ensemble_predictions.npz"
    np.savez_compressed(
        output,
        **{
            f"{split_name}_{key}": value
            for split_name, values in splits.items()
            for key, value in values.items()
        },
    )
    return output


def _validation_standardized_oof(
    prediction_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(prediction_path, allow_pickle=False) as payload:
        val_selected = np.asarray(payload["val_selected_risk"], dtype=float)
        test_selected = np.asarray(payload["test_selected_risk"], dtype=float)
        val_reference = np.asarray(payload["val_mainline_risk"], dtype=float)
        test_reference = np.asarray(payload["test_mainline_risk"], dtype=float)
        test_time = np.asarray(payload["test_time"], dtype=float)
        test_event = np.asarray(payload["test_event"], dtype=float)

    def standardize(validation: np.ndarray, test: np.ndarray) -> np.ndarray:
        scale = float(np.std(validation))
        if scale < 1e-8:
            raise RuntimeError("Validation risk has zero variance.")
        return (test - float(np.mean(validation))) / scale

    return (
        test_time,
        test_event,
        standardize(val_reference, test_reference),
        standardize(val_selected, test_selected),
    )


def run_formal_aft_fusion(
    *,
    gnn_root: Path = GNN_ROOT,
    output_root: Path = OUTPUT_ROOT,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    lock = _load_lock()
    protocol = lock["protocol"]
    folds: list[dict[str, Any]] = []
    pooled_time: list[float] = []
    pooled_event: list[float] = []
    pooled_reference: list[float] = []
    pooled_selected: list[float] = []

    for test_group in protocol["outer_test_groups"]:
        validation_group = (int(test_group) + 1) % 5
        fold_name = f"outer_group{test_group}_val{validation_group}_five_seed"
        fold_root = gnn_root / fold_name
        mainline_path = _export_mainline_predictions(fold_root)
        sweep_root = output_root / fold_name / "aft_seed_sweep"
        run_seed_sweep(
            config_path=str(config_path),
            mainline_predictions_path=str(mainline_path),
            output_root=sweep_root,
            split_seed=int(test_group),
            seeds=protocol["aft_seeds"],
            preset_names=protocol["aft_presets"],
            num_boost_round=int(protocol["num_boost_round"]),
            early_stopping_rounds=int(protocol["early_stopping_rounds"]),
            nthread=int(protocol["nthread"]),
            minimum_c_index_delta=float(protocol["minimum_validation_c_index_delta"]),
            maximum_alpha=float(protocol["maximum_alpha"]),
            skip_completed=True,
            feature_set="full",
        )
        run_dirs = [
            sweep_root / f"split{int(test_group)}_seed{int(seed)}"
            for seed in protocol["aft_seeds"]
        ]
        ensemble_dir = output_root / fold_name / "aft_seed_ensemble"
        result = build_seed_ensemble(
            run_dirs=run_dirs,
            output_dir=ensemble_dir,
            minimum_c_index_delta=float(protocol["minimum_validation_c_index_delta"]),
            maximum_alpha=float(protocol["maximum_alpha"]),
            emit_json=False,
        )
        prediction_path = Path(result["artifacts"]["predictions"])
        time, event, reference, selected = _validation_standardized_oof(prediction_path)
        pooled_time.extend(time.tolist())
        pooled_event.extend(event.tolist())
        pooled_reference.extend(reference.tolist())
        pooled_selected.extend(selected.tolist())
        folds.append(
            {
                "outer_test_group": int(test_group),
                "inner_validation_group": int(validation_group),
                "selected_alpha": float(result["blend_selection"]["selected"]["alpha"]),
                "validation_reference_c_index": float(result["validation"]["reference_c_index"]),
                "validation_expert_c_index": float(
                    result["validation"]["expert_ensemble_c_index"]
                ),
                "validation_selected_c_index": float(result["validation"]["selected_c_index"]),
                "test_reference_c_index": float(result["test"]["reference_c_index"]),
                "test_expert_c_index": float(result["test"]["expert_ensemble_c_index"]),
                "test_selected_c_index": float(result["test"]["selected_c_index"]),
                "test_c_index_delta": float(result["test"]["selected_c_index_delta"]),
                "test_reference_calibrated_cox_loss": float(
                    result["test"]["reference_calibrated_cox_loss"]
                ),
                "test_selected_calibrated_cox_loss": float(
                    result["test"]["selected_calibrated_cox_loss"]
                ),
                "test_calibrated_cox_loss_delta": float(
                    result["test"]["calibrated_cox_loss_delta"]
                ),
                "test_main_expert_correlation": result["test"]["main_expert_correlation"],
                "test_pair_corrections": result["test"]["selected_pair_corrections"],
                "artifact": str(prediction_path.as_posix()),
            }
        )

    reference_scores = [row["test_reference_c_index"] for row in folds]
    selected_scores = [row["test_selected_c_index"] for row in folds]
    deltas = [row["test_c_index_delta"] for row in folds]
    loss_deltas = [row["test_calibrated_cox_loss_delta"] for row in folds]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            **protocol,
            "lock_path": str(LOCK_PATH.relative_to(ROOT).as_posix()),
            "lock_sha256": _sha256(LOCK_PATH),
            "selection_uses_test_labels": False,
        },
        "folds": folds,
        "aggregate": {
            "macro_mean_reference_test_c_index": float(statistics.mean(reference_scores)),
            "macro_mean_selected_test_c_index": float(statistics.mean(selected_scores)),
            "macro_mean_test_c_index_delta": float(statistics.mean(deltas)),
            "macro_std_selected_test_c_index": float(statistics.stdev(selected_scores)),
            "minimum_selected_test_c_index": float(min(selected_scores)),
            "maximum_selected_test_c_index": float(max(selected_scores)),
            "num_c_index_improved_folds": int(sum(delta > 0.0 for delta in deltas)),
            "num_c_index_non_decreased_folds": int(sum(delta >= 0.0 for delta in deltas)),
            "macro_mean_calibrated_cox_loss_delta": float(statistics.mean(loss_deltas)),
            "num_cox_loss_improved_folds": int(sum(delta < 0.0 for delta in loss_deltas)),
            "validation_standardized_pooled_oof_reference_c_index": float(
                concordance_index(pooled_time, pooled_event, pooled_reference)
            ),
            "validation_standardized_pooled_oof_selected_c_index": float(
                concordance_index(pooled_time, pooled_event, pooled_selected)
            ),
        },
        "legacy_reference": {
            "c_index": 0.7570563484482322,
            "protocol_note": "Legacy topology_v6 result used two split seeds, so it is contextual rather than a paired five-group comparator.",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "formal_aft_fusion_logo_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked AFT fusion for formal V7 v3 LOGO folds.")
    parser.add_argument("--gnn-root", type=Path, default=GNN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            run_formal_aft_fusion(
                gnn_root=args.gnn_root,
                output_root=args.output_root,
                config_path=args.config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
