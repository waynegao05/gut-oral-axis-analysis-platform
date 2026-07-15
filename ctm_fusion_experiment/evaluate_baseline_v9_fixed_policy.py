from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ctm_fusion_experiment.train_baseline_v5 import _selected_metrics
from ctm_fusion_experiment.utils.data_loader import FusionArraySet
from ctm_fusion_experiment.utils.metrics import concordance_index, summarize_paired_folds
from ctm_fusion_experiment.utils.reporting import write_csv, write_json
from ctm_fusion_experiment.utils.risk_ensemble_selection import apply_risk_ensemble


def build_baseline_v9_fixed_policy_summary(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    policy_name: str = "mean_all",
    shrinkage_alpha: float = 1.0,
) -> dict[str, object]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    alpha = float(shrinkage_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("shrinkage_alpha must be between 0 and 1.")
    metadata_path = source_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    rows: list[dict[str, Any]] = []

    for fold_dir in sorted(source_path.glob("fold_*")):
        fold_summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        selection = json.loads((fold_dir / "baseline_v9_oof_selection.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(fold_dir / "test_predictions.csv")
        candidate_names = [str(candidate["name"]) for candidate in fold_summary["candidate_models"]]
        candidate = _find_candidate(selection["candidates"], policy_name)
        risk_matrix = torch.tensor(
            predictions[[f"{name}_risk" for name in candidate_names]].to_numpy().T,
            dtype=torch.float32,
        )
        candidate_risk = apply_risk_ensemble(
            risk_matrix,
            candidate["weights"],
            selection["risk_means"],
            selection["risk_stds"],
        )
        reference_weights = [0.0 for _ in candidate_names]
        reference_weights[0] = 1.0
        reference_risk = apply_risk_ensemble(
            risk_matrix,
            reference_weights,
            selection["risk_means"],
            selection["risk_stds"],
        )
        selected_risk = (1.0 - alpha) * reference_risk + alpha * candidate_risk
        arrays = FusionArraySet(
            sample_ids=tuple(str(value) for value in predictions["sample_id"].tolist()),
            graph=torch.empty((len(predictions), 0)).numpy(),
            clinical=torch.empty((len(predictions), 0)).numpy(),
            metabolite=torch.empty((len(predictions), 0)).numpy(),
            time=predictions["time"].to_numpy(dtype="float32"),
            event=predictions["event"].to_numpy(dtype="float32"),
        )
        selected_metrics, diagnostics = _selected_metrics(arrays, reference_risk, selected_risk)
        reference_c_index = concordance_index(arrays.time, arrays.event, reference_risk.numpy())
        rows.append(
            {
                "fold": int(fold_summary["fold"]),
                "reference_c_index": reference_c_index,
                "selected_c_index": selected_metrics["c_index"],
                "selected_delta": selected_metrics["c_index"] - reference_c_index,
                "policy_name": policy_name,
                "shrinkage_alpha": alpha,
                "weights": json.dumps(candidate["weights"]),
                "oof_c_index": candidate["c_index"],
                "oof_reference_c_index": selection["oof_reference_c_index"],
                "oof_delta": candidate["c_index_delta"],
                "selected_loss": selected_metrics["loss"],
                "improved_pairs": diagnostics["improved_pairs"],
                "regressed_pairs": diagnostics["regressed_pairs"],
                "net_improved_pairs": diagnostics["net_improved_pairs"],
                "pair_credit_delta": diagnostics["pair_credit_delta"],
            }
        )

    if not rows:
        raise ValueError(f"No v9 OOF fold outputs found under {source_path.as_posix()}.")

    reference_scores = [row["reference_c_index"] for row in rows]
    selected_scores = [row["selected_c_index"] for row in rows]
    result = {
        "source_dir": source_path.as_posix(),
        "policy_name": policy_name,
        "shrinkage_alpha": alpha,
        "num_folds": len(rows),
        "selected_paired_comparison": summarize_paired_folds(
            baseline=reference_scores,
            ctm=selected_scores,
        ),
        "folds": rows,
        "interpretation": (
            "This is a fixed-policy audit over already trained baseline v9 OOF models. It applies the same "
            "OOF-calibrated ensemble policy on every outer test fold, optionally shrunk back toward the "
            "reference risk; it does not choose a policy per fold."
        ),
        **metadata,
    }
    output_stem = _output_stem(policy_name, alpha)
    write_json(output_path / f"{output_stem}_summary.json", result)
    write_csv(output_path / f"{output_stem}_fold_comparison.csv", rows)
    return result


def _find_candidate(candidates: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    for candidate in candidates:
        if candidate["candidate_name"] == policy_name:
            return candidate
    raise ValueError(f"Policy candidate {policy_name!r} not found.")


def _output_stem(policy_name: str, shrinkage_alpha: float) -> str:
    if shrinkage_alpha == 1.0:
        return f"baseline_v9_fixed_{policy_name}"
    alpha_label = str(shrinkage_alpha).replace(".", "p")
    return f"baseline_v9_fixed_{policy_name}_alpha_{alpha_label}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="outputs/ctm_fusion_experiment/baseline_v9_oof_formal")
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/baseline_v9_fixed_mean_all")
    parser.add_argument("--policy-name", default="mean_all")
    parser.add_argument("--shrinkage-alpha", type=float, default=1.0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_baseline_v9_fixed_policy_summary(
                args.source_dir,
                args.output_dir,
                policy_name=args.policy_name,
                shrinkage_alpha=args.shrinkage_alpha,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
