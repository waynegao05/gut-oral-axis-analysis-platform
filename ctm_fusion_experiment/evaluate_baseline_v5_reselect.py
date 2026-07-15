from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

from ctm_fusion_experiment.utils.losses import cox_partial_likelihood_loss
from ctm_fusion_experiment.utils.metrics import concordance_index, summarize_paired_folds
from ctm_fusion_experiment.utils.pair_diagnostics import pairwise_cindex_diagnostics
from ctm_fusion_experiment.utils.reporting import write_csv, write_json
from ctm_fusion_experiment.utils.risk_ensemble_selection import apply_risk_ensemble


def build_baseline_v5_reselected_summary(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    min_validation_delta: float = 0.001,
    policy: str = "ensemble_only_or_reference",
) -> dict[str, object]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = source_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    rows = []

    for fold_dir in sorted(source_path.glob("fold_*")):
        fold_summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        selection = json.loads((fold_dir / "baseline_v5_selection.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(fold_dir / "test_predictions.csv")
        reference = float(fold_summary["reference_baseline"]["test"]["c_index"])
        selected_candidate = _select_candidate(
            selection["candidates"],
            reference_validation_c_index=float(selection["validation_reference_c_index"]),
            min_validation_delta=float(min_validation_delta),
            policy=policy,
        )
        test_risk = _candidate_test_risk(fold_summary, selection, predictions, selected_candidate)
        selected_c_index = concordance_index(
            predictions["time"].astype(float).to_numpy(),
            predictions["event"].astype(float).to_numpy(),
            test_risk.detach().cpu().numpy(),
        )
        selected_loss = cox_partial_likelihood_loss(
            test_risk,
            torch.tensor(predictions["time"].astype(float).to_numpy(), dtype=torch.float32),
            torch.tensor(predictions["event"].astype(float).to_numpy(), dtype=torch.float32),
        )
        diagnostics = pairwise_cindex_diagnostics(
            predictions["time"].astype(float).to_numpy(),
            predictions["event"].astype(float).to_numpy(),
            predictions["reference_risk"].astype(float).to_numpy(),
            test_risk.detach().cpu().numpy(),
        )
        row = {
            "fold": int(fold_summary["fold"]),
            "reference_c_index": reference,
            "selected_c_index": float(selected_c_index),
            "selected_delta": float(selected_c_index - reference),
            "selected_candidate": selected_candidate["candidate_name"],
            "selected_weights": json.dumps(selected_candidate["weights"]),
            "validation_reference_c_index": float(selection["validation_reference_c_index"]),
            "validation_selected_c_index": float(selected_candidate["c_index"]),
            "validation_selected_delta": float(selected_candidate["c_index"] - selection["validation_reference_c_index"]),
            "selected_loss": float(selected_loss.item()),
            "improved_pairs": diagnostics["improved_pairs"],
            "regressed_pairs": diagnostics["regressed_pairs"],
            "net_improved_pairs": diagnostics["net_improved_pairs"],
            "pair_credit_delta": diagnostics["pair_credit_delta"],
        }
        rows.append(row)
        write_json(output_path / f"fold_{int(fold_summary['fold']):02d}_selection.json", {**row, "policy": policy})

    result = {
        "source_dir": source_path.as_posix(),
        "policy": policy,
        "min_validation_delta": float(min_validation_delta),
        "num_folds": len(rows),
        "selected_paired_comparison": summarize_paired_folds(
            baseline=[row["reference_c_index"] for row in rows],
            ctm=[row["selected_c_index"] for row in rows],
        ),
        "folds": rows,
        "interpretation": (
            "This is a post-training v5 re-selection pass. It does not retrain models and does "
            "not use test labels for selecting candidates; candidates are filtered by validation "
            "c-index delta and policy, then evaluated on the held-out test fold."
        ),
        **metadata,
    }
    write_json(output_path / "baseline_v5_reselected_summary.json", result)
    write_csv(output_path / "baseline_v5_reselected_fold_comparison.csv", rows)
    return result


def _select_candidate(
    candidates: list[dict[str, Any]],
    *,
    reference_validation_c_index: float,
    min_validation_delta: float,
    policy: str,
) -> dict[str, Any]:
    filtered = []
    for candidate in candidates:
        name = str(candidate["candidate_name"])
        if name == "single:reference":
            continue
        if not _allowed_by_policy(name, policy):
            continue
        if name != "reference" and float(candidate["c_index"]) - reference_validation_c_index < min_validation_delta:
            continue
        filtered.append(candidate)

    if not filtered:
        filtered = [candidate for candidate in candidates if candidate["candidate_name"] == "reference"]
    if not filtered:
        raise ValueError("No reference candidate found.")
    return max(filtered, key=_candidate_sort_key)


def _allowed_by_policy(name: str, policy: str) -> bool:
    if name == "reference":
        return True
    if policy == "ensemble_only_or_reference":
        return not name.startswith("single:")
    if policy == "all":
        return True
    if policy == "single_only_or_reference":
        return name.startswith("single:")
    raise ValueError(f"Unknown policy: {policy}")


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    weights = [float(weight) for weight in candidate["weights"]]
    nonzero = sum(1 for weight in weights if abs(weight) > 1e-9)
    max_weight = max(abs(weight) for weight in weights)
    return (
        float(candidate["c_index"]),
        -float(nonzero),
        float(max_weight),
    )


def _candidate_test_risk(
    fold_summary: dict[str, Any],
    selection: dict[str, Any],
    predictions: pd.DataFrame,
    candidate: dict[str, Any],
) -> torch.Tensor:
    risk_rows = []
    for candidate_model in fold_summary["candidate_models"]:
        name = candidate_model["name"]
        column = "reference_risk" if name == "reference" else f"{name}_risk"
        risk_rows.append(predictions[column].astype(float).to_numpy())
    risk_matrix = torch.tensor(np.stack(risk_rows), dtype=torch.float32)
    return apply_risk_ensemble(
        risk_matrix,
        candidate["weights"],
        selection["risk_means"],
        selection["risk_stds"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="outputs/ctm_fusion_experiment/baseline_v5_formal")
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/baseline_v5_reselected_ensemble")
    parser.add_argument("--min-validation-delta", type=float, default=0.001)
    parser.add_argument(
        "--policy",
        choices=["ensemble_only_or_reference", "all", "single_only_or_reference"],
        default="ensemble_only_or_reference",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_baseline_v5_reselected_summary(
                args.source_dir,
                args.output_dir,
                min_validation_delta=args.min_validation_delta,
                policy=args.policy,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
