from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.ensemble_stack_v2 import _cohort_cox_loss, _fit_cox_risk_scale
from research.metrics import concordance_index

from .topology_aft_fusion import (
    _pair_correction_diagnostics,
    _risk_correlation,
    _standardize_from_train,
    select_blend_alpha,
)


def _load_run(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(run_dir)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with np.load(path / "predictions.npz", allow_pickle=False) as payload:
        predictions = {key: np.asarray(payload[key]) for key in payload.files}
    return summary, predictions


def _assert_aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
    for split_name in ("train", "val", "test"):
        for suffix in ("sample_ids", "time", "event", "mainline_risk"):
            key = f"{split_name}_{suffix}"
            if reference[key].shape != candidate[key].shape or not np.array_equal(reference[key], candidate[key]):
                raise ValueError(f"Seed prediction artifacts are not aligned for {key}.")


def build_seed_ensemble(
    *,
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    minimum_c_index_delta: float = 0.0003,
    maximum_alpha: float = 1.0,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("At least two completed seed runs are required for a seed ensemble.")
    loaded = [_load_run(path) for path in run_dirs]
    reference_predictions = loaded[0][1]
    for _, predictions in loaded[1:]:
        _assert_aligned(reference_predictions, predictions)

    labels = {
        split_name: (
            np.asarray(reference_predictions[f"{split_name}_time"], dtype=float),
            np.asarray(reference_predictions[f"{split_name}_event"], dtype=float),
        )
        for split_name in ("train", "val", "test")
    }
    main_raw = {
        split_name: np.asarray(reference_predictions[f"{split_name}_mainline_risk"], dtype=float)
        for split_name in ("train", "val", "test")
    }
    main_arrays, main_scaler = _standardize_from_train(main_raw["train"], main_raw["val"], main_raw["test"])
    main_z = dict(zip(("train", "val", "test"), main_arrays))

    expert_members: dict[str, list[np.ndarray]] = {split_name: [] for split_name in ("train", "val", "test")}
    member_rows = []
    for summary, predictions in loaded:
        standardized, scaler = _standardize_from_train(
            predictions["train_expert_risk"],
            predictions["val_expert_risk"],
            predictions["test_expert_risk"],
        )
        member = dict(zip(("train", "val", "test"), standardized))
        for split_name in member:
            expert_members[split_name].append(member[split_name])
        member_rows.append(
            {
                "model_seed": int(summary["model_seed"]),
                "selected_expert": summary["selected_expert"]["name"],
                "risk_scaler": scaler,
                "validation_c_index": float(
                    concordance_index(labels["val"][0], labels["val"][1], member["val"])
                ),
                "test_c_index": float(
                    concordance_index(labels["test"][0], labels["test"][1], member["test"])
                ),
            }
        )

    expert_mean_raw = {
        split_name: np.mean(np.stack(expert_members[split_name], axis=0), axis=0)
        for split_name in ("train", "val", "test")
    }
    expert_arrays, ensemble_scaler = _standardize_from_train(
        expert_mean_raw["train"], expert_mean_raw["val"], expert_mean_raw["test"]
    )
    expert_z = dict(zip(("train", "val", "test"), expert_arrays))

    selection = select_blend_alpha(
        main_z["val"],
        expert_z["val"],
        labels["val"][0],
        labels["val"][1],
        minimum_c_index_delta=float(minimum_c_index_delta),
        maximum_alpha=float(maximum_alpha),
    )
    alpha = float(selection["selected"]["alpha"])
    selected = {
        split_name: (1.0 - alpha) * main_z[split_name] + alpha * expert_z[split_name]
        for split_name in ("train", "val", "test")
    }

    reference_calibration = _fit_cox_risk_scale(main_z["val"], labels["val"][0], labels["val"][1])
    selected_calibration = _fit_cox_risk_scale(selected["val"], labels["val"][0], labels["val"][1])
    reference_test_c = float(concordance_index(labels["test"][0], labels["test"][1], main_z["test"]))
    selected_test_c = float(concordance_index(labels["test"][0], labels["test"][1], selected["test"]))
    reference_loss = _cohort_cox_loss(
        main_z["test"] * float(reference_calibration["scale"]), labels["test"][0], labels["test"][1]
    )
    selected_loss = _cohort_cox_loss(
        selected["test"] * float(selected_calibration["scale"]), labels["test"][0], labels["test"][1]
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / "seed_ensemble_predictions.npz"
    np.savez_compressed(
        prediction_path,
        selected_alpha=np.asarray(alpha, dtype=float),
        **{
            f"{split_name}_{key}": value
            for split_name in ("train", "val", "test")
            for key, value in {
                "sample_ids": reference_predictions[f"{split_name}_sample_ids"],
                "time": labels[split_name][0],
                "event": labels[split_name][1],
                "mainline_risk": main_raw[split_name],
                "expert_ensemble_risk": expert_z[split_name],
                "selected_risk": selected[split_name],
            }.items()
        },
    )
    result = {
        "protocol": "fixed_five_seed_topology_aft_mean_then_validation_safe_fusion",
        "split_seed": int(loaded[0][0]["split_seed"]),
        "run_dirs": [str(Path(path).as_posix()) for path in run_dirs],
        "member_rows": member_rows,
        "mainline_scaler": main_scaler,
        "expert_ensemble_scaler": ensemble_scaler,
        "blend_selection": selection,
        "validation": {
            "reference_c_index": float(
                concordance_index(labels["val"][0], labels["val"][1], main_z["val"])
            ),
            "expert_ensemble_c_index": float(
                concordance_index(labels["val"][0], labels["val"][1], expert_z["val"])
            ),
            "selected_c_index": float(
                concordance_index(labels["val"][0], labels["val"][1], selected["val"])
            ),
        },
        "test": {
            "reference_c_index": reference_test_c,
            "expert_ensemble_c_index": float(
                concordance_index(labels["test"][0], labels["test"][1], expert_z["test"])
            ),
            "selected_c_index": selected_test_c,
            "selected_c_index_delta": selected_test_c - reference_test_c,
            "reference_calibrated_cox_loss": reference_loss,
            "selected_calibrated_cox_loss": selected_loss,
            "calibrated_cox_loss_delta": selected_loss - reference_loss,
            "main_expert_correlation": _risk_correlation(main_z["test"], expert_z["test"]),
            "selected_pair_corrections": _pair_correction_diagnostics(
                labels["test"][0], labels["test"][1], main_z["test"], selected["test"]
            ),
        },
        "artifacts": {"predictions": str(prediction_path.as_posix())},
        "decision": "accept_seed_ensemble" if selection["selected_expert"] else "keep_mainline",
    }
    summary_path = output_path / "seed_ensemble_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-c-index-delta", type=float, default=0.0003)
    parser.add_argument("--maximum-alpha", type=float, default=1.0)
    args = parser.parse_args()
    build_seed_ensemble(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        minimum_c_index_delta=args.minimum_c_index_delta,
        maximum_alpha=args.maximum_alpha,
    )


if __name__ == "__main__":
    main()
