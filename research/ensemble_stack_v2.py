from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from research.ensemble_v2 import build_loader, build_model, load_checkpoints
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.train_v2 import resolve_device


@dataclass(frozen=True)
class PredictionMatrix:
    sample_ids: list[str]
    time: np.ndarray
    event: np.ndarray
    risk_matrix: np.ndarray


@dataclass(frozen=True)
class EnsembleCandidate:
    name: str
    weights: list[float]
    val_c_index: float
    test_c_index: float


def evaluate_stacked_ensemble(
    config_path: str,
    checkpoint_glob: str,
    *,
    split_seed: int | None = None,
    device_arg: str = "cuda",
    min_validation_delta: float = 0.0,
    candidate_policy: str = "all",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    if split_seed is None:
        split_seed = config["train"].get("split_seed")
    if split_seed is None:
        split_seed = int(config["seed"])
    device = resolve_device(device_arg)

    val_predictions = _predict_split(config, checkpoints, split_seed=split_seed, split="val", device=device)
    test_predictions = _predict_split(config, checkpoints, split_seed=split_seed, split="test", device=device)
    val_standardized, test_standardized, scaler = _standardize_by_validation(
        val_predictions.risk_matrix,
        test_predictions.risk_matrix,
    )

    reference_weights = [1.0 / len(checkpoints) for _ in checkpoints]
    reference_val_risk = _apply_weights(val_predictions.risk_matrix, reference_weights)
    reference_test_risk = _apply_weights(test_predictions.risk_matrix, reference_weights)
    reference_val_c_index = concordance_index(val_predictions.time, val_predictions.event, reference_val_risk)
    reference_test_c_index = concordance_index(test_predictions.time, test_predictions.event, reference_test_risk)
    member_val_c_indices = [
        concordance_index(val_predictions.time, val_predictions.event, val_predictions.risk_matrix[index])
        for index in range(len(checkpoints))
    ]
    member_test_c_indices = [
        concordance_index(test_predictions.time, test_predictions.event, test_predictions.risk_matrix[index])
        for index in range(len(checkpoints))
    ]

    candidates = _build_candidates(
        val_standardized=val_standardized,
        test_standardized=test_standardized,
        val_time=val_predictions.time,
        val_event=val_predictions.event,
        test_time=test_predictions.time,
        test_event=test_predictions.event,
        member_val_c_indices=member_val_c_indices,
    )
    allowed_candidates = [
        candidate
        for candidate in candidates
        if _candidate_allowed(candidate.name, candidate_policy)
    ]
    eligible = [
        candidate
        for candidate in allowed_candidates
        if candidate.val_c_index - reference_val_c_index >= float(min_validation_delta)
    ]
    selected = max(eligible, key=lambda candidate: candidate.val_c_index) if eligible else None
    if selected is None:
        selected_name = "raw_mean_all"
        selected_weights = reference_weights
        selected_val_risk = reference_val_risk
        selected_val_c_index = reference_val_c_index
        selected_test_c_index = reference_test_c_index
        selected_test_risk = reference_test_risk
    else:
        selected_name = selected.name
        selected_weights = selected.weights
        selected_val_risk = _apply_weights(val_standardized, selected.weights)
        selected_val_c_index = selected.val_c_index
        selected_test_c_index = selected.test_c_index
        selected_test_risk = _apply_weights(test_standardized, selected.weights)

    reference_calibration = _fit_cox_risk_scale(
        reference_val_risk,
        val_predictions.time,
        val_predictions.event,
    )
    selected_calibration = _fit_cox_risk_scale(
        selected_val_risk,
        val_predictions.time,
        val_predictions.event,
    )
    reference_calibrated_test_risk = reference_test_risk * float(reference_calibration["scale"])
    selected_calibrated_test_risk = selected_test_risk * float(selected_calibration["scale"])

    result = {
        "config_path": config_path,
        "checkpoint_glob": checkpoint_glob,
        "split_seed": split_seed,
        "device": str(device),
        "num_models": len(checkpoints),
        "checkpoints": [str(path) for path in checkpoints],
        "reference": {
            "candidate_name": "raw_mean_all",
            "weights": reference_weights,
            "validation_c_index": reference_val_c_index,
            "test_c_index": reference_test_c_index,
            "validation_cohort_cox_loss": _cohort_cox_loss(
                reference_val_risk,
                val_predictions.time,
                val_predictions.event,
            ),
            "test_cohort_cox_loss": _cohort_cox_loss(
                reference_test_risk,
                test_predictions.time,
                test_predictions.event,
            ),
            "cox_scale_calibration": reference_calibration,
            "calibrated_test_cohort_cox_loss": _cohort_cox_loss(
                reference_calibrated_test_risk,
                test_predictions.time,
                test_predictions.event,
            ),
        },
        "selected": {
            "candidate_name": selected_name,
            "weights": selected_weights,
            "validation_c_index": selected_val_c_index,
            "validation_delta": selected_val_c_index - reference_val_c_index,
            "test_c_index": selected_test_c_index,
            "test_delta": selected_test_c_index - reference_test_c_index,
            "candidate_policy": candidate_policy,
            "validation_cohort_cox_loss": _cohort_cox_loss(
                selected_val_risk,
                val_predictions.time,
                val_predictions.event,
            ),
            "test_cohort_cox_loss": _cohort_cox_loss(
                selected_test_risk,
                test_predictions.time,
                test_predictions.event,
            ),
            "cox_scale_calibration": selected_calibration,
            "calibrated_test_cohort_cox_loss": _cohort_cox_loss(
                selected_calibrated_test_risk,
                test_predictions.time,
                test_predictions.event,
            ),
        },
        "member_c_indices": [
            {
                "checkpoint": str(checkpoint),
                "validation_c_index": float(val_c_index),
                "test_c_index": float(test_c_index),
            }
            for checkpoint, val_c_index, test_c_index in zip(checkpoints, member_val_c_indices, member_test_c_indices)
        ],
        "risk_standardization": scaler,
        "candidates": [
            {
                "candidate_name": candidate.name,
                "weights": candidate.weights,
                "validation_c_index": candidate.val_c_index,
                "validation_delta": candidate.val_c_index - reference_val_c_index,
                "test_c_index": candidate.test_c_index,
                "test_delta": candidate.test_c_index - reference_test_c_index,
            }
            for candidate in candidates
        ],
        "test_predictions": [
            {
                "sample_id": sample_id,
                "time": float(time),
                "event": float(event),
                "reference_risk": float(reference_risk),
                "reference_calibrated_risk": float(reference_calibrated_risk),
                "selected_risk": float(selected_risk),
                "selected_calibrated_risk": float(selected_calibrated_risk),
            }
            for sample_id, time, event, reference_risk, reference_calibrated_risk, selected_risk, selected_calibrated_risk in zip(
                test_predictions.sample_ids,
                test_predictions.time,
                test_predictions.event,
                reference_test_risk,
                reference_calibrated_test_risk,
                selected_test_risk,
                selected_calibrated_test_risk,
            )
        ],
        "interpretation": (
            "Validation-selected checkpoint stacking over the locked Cox mainline. Candidate weights are chosen "
            "only on the validation split, then evaluated once on the fixed test split."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _predict_split(
    config: dict[str, Any],
    checkpoints: Sequence[Path],
    *,
    split_seed: int,
    split: str,
    device: torch.device,
) -> PredictionMatrix:
    loader, dataset = build_loader(config, split_seed=split_seed, split=split)
    all_sample_ids: list[str] = []
    all_time: list[float] = []
    all_event: list[float] = []
    checkpoint_risks: list[list[float]] = []

    for checkpoint_path in checkpoints:
        model = build_model(config, dataset, device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        risks: list[float] = []
        sample_ids: list[str] = []
        time_values: list[float] = []
        event_values: list[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch, compute_contrastive=False)
                risks.extend(output["risk"].detach().cpu().numpy().tolist())
                sample_ids.extend(list(batch.sample_id))
                time_values.extend(batch.time.detach().cpu().numpy().tolist())
                event_values.extend(batch.event.detach().cpu().numpy().tolist())
        if not all_sample_ids:
            all_sample_ids = sample_ids
            all_time = time_values
            all_event = event_values
        elif sample_ids != all_sample_ids:
            raise RuntimeError("Checkpoint predictions are not aligned. Use a fixed split seed.")
        checkpoint_risks.append(risks)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return PredictionMatrix(
        sample_ids=all_sample_ids,
        time=np.asarray(all_time, dtype=float),
        event=np.asarray(all_event, dtype=float),
        risk_matrix=np.asarray(checkpoint_risks, dtype=float),
    )


def _standardize_by_validation(
    val_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    means = val_matrix.mean(axis=1, keepdims=True)
    stds = np.maximum(val_matrix.std(axis=1, keepdims=True), 1e-6)
    return (
        (val_matrix - means) / stds,
        (test_matrix - means) / stds,
        {
            "risk_means": means.squeeze(axis=1).astype(float).tolist(),
            "risk_stds": stds.squeeze(axis=1).astype(float).tolist(),
        },
    )


def _build_candidates(
    *,
    val_standardized: np.ndarray,
    test_standardized: np.ndarray,
    val_time: np.ndarray,
    val_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    member_val_c_indices: Sequence[float],
) -> list[EnsembleCandidate]:
    num_models = int(val_standardized.shape[0])
    candidate_weights: list[tuple[str, list[float]]] = []
    candidate_weights.append(("standardized_mean_all", [1.0 / num_models for _ in range(num_models)]))
    for index in range(num_models):
        weights = [0.0 for _ in range(num_models)]
        weights[index] = 1.0
        candidate_weights.append((f"single:{index}", weights))

    order = list(np.argsort(np.asarray(member_val_c_indices, dtype=float)))
    for count in range(2, num_models + 1):
        weights = [0.0 for _ in range(num_models)]
        for index in order[-count:]:
            weights[int(index)] = 1.0 / count
        candidate_weights.append((f"top{count}_val_mean", weights))

    for temperature in (0.003, 0.005, 0.01, 0.02, 0.05):
        scores = np.asarray(member_val_c_indices, dtype=float)
        scaled = np.exp((scores - scores.max()) / float(temperature))
        weights = (scaled / scaled.sum()).astype(float).tolist()
        candidate_weights.append((f"softmax_val_t{temperature}", weights))

    return [
        EnsembleCandidate(
            name=name,
            weights=[float(value) for value in weights],
            val_c_index=concordance_index(val_time, val_event, _apply_weights(val_standardized, weights)),
            test_c_index=concordance_index(test_time, test_event, _apply_weights(test_standardized, weights)),
        )
        for name, weights in candidate_weights
    ]


def _candidate_allowed(candidate_name: str, candidate_policy: str) -> bool:
    if candidate_policy == "all":
        return True
    if candidate_policy == "topk_mean_or_reference":
        return candidate_name.startswith("top") and candidate_name.endswith("_val_mean")
    if candidate_policy == "softmax_or_reference":
        return candidate_name.startswith("softmax_val_t")
    if candidate_policy == "single_or_reference":
        return candidate_name.startswith("single:")
    raise ValueError(f"Unknown candidate_policy: {candidate_policy}")


def _apply_weights(risk_matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float)
    if risk_matrix.shape[0] != weights_array.shape[0]:
        raise ValueError("weights must match the number of risk matrix rows.")
    return np.sum(risk_matrix * weights_array[:, None], axis=0)


def _cohort_cox_loss(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    with torch.no_grad():
        return float(
            cox_ph_loss(
                torch.as_tensor(np.asarray(risk, dtype=float), dtype=torch.float64),
                torch.as_tensor(np.asarray(time, dtype=float), dtype=torch.float64),
                torch.as_tensor(np.asarray(event, dtype=float), dtype=torch.float64),
            ).item()
        )


def _fit_cox_risk_scale(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> dict[str, Any]:
    risk_values = np.asarray(risk, dtype=float)
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    if not (risk_values.shape == time_values.shape == event_values.shape):
        raise ValueError("risk, time, and event must have identical shapes for Cox calibration.")

    uncalibrated_loss = _cohort_cox_loss(risk_values, time_values, event_values)
    best_scale = 1.0
    best_loss = uncalibrated_loss
    lower_log_scale = -4.0
    upper_log_scale = 4.0

    for _ in range(3):
        log_scales = np.linspace(lower_log_scale, upper_log_scale, num=81)
        losses = np.asarray(
            [
                _cohort_cox_loss(risk_values * float(np.exp(log_scale)), time_values, event_values)
                for log_scale in log_scales
            ],
            dtype=float,
        )
        best_index = int(np.argmin(losses))
        if float(losses[best_index]) < best_loss:
            best_loss = float(losses[best_index])
            best_scale = float(np.exp(log_scales[best_index]))
        lower_index = max(0, best_index - 1)
        upper_index = min(len(log_scales) - 1, best_index + 1)
        lower_log_scale = float(log_scales[lower_index])
        upper_log_scale = float(log_scales[upper_index])

    return {
        "fit_split": "validation",
        "method": "positive_scalar_grid_search",
        "scale": best_scale,
        "uncalibrated_validation_cox_loss": uncalibrated_loss,
        "calibrated_validation_cox_loss": best_loss,
        "preserves_ranking": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument(
        "--checkpoint-glob",
        default="outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--min-validation-delta", type=float, default=0.0)
    parser.add_argument(
        "--candidate-policy",
        choices=["all", "topk_mean_or_reference", "softmax_or_reference", "single_or_reference"],
        default="all",
    )
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/cox_fixed_split_stacked_ensemble/stacked_ensemble_summary.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_stacked_ensemble(
                config_path=args.config,
                checkpoint_glob=args.checkpoint_glob,
                split_seed=args.split_seed,
                device_arg=args.device,
                min_validation_delta=args.min_validation_delta,
                candidate_policy=args.candidate_policy,
                output_path=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
