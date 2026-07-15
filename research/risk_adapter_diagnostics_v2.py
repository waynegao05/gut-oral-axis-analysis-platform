from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from research.ensemble_stack_v2 import _apply_weights, _predict_split
from research.ensemble_v2 import load_checkpoints
from research.expert_stack_v2 import _load_feature_splits, _standardize_feature_splits
from research.metrics import concordance_index
from research.risk_adapter_v2 import _disagreement_features
from research.train_v2 import resolve_device


@dataclass(frozen=True)
class PairChangeSummary:
    permissible_pairs: int
    baseline_c_index: float
    selected_c_index: float
    c_index_delta: float
    corrected_pairs: int
    harmed_pairs: int
    tied_to_correct_pairs: int
    correct_to_tied_pairs: int
    unchanged_pairs: int
    net_corrected_pairs: int


def run_risk_adapter_diagnostics(
    summary_path: str | Path,
    *,
    config_path: str | None = None,
    checkpoint_glob: str | None = None,
    split_seed: int | None = None,
    device_arg: str = "cuda",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = config_path or str(summary["config_path"])
    checkpoint_glob = checkpoint_glob or str(summary["checkpoint_glob"])
    split_seed = int(split_seed if split_seed is not None else summary["split_seed"])

    predictions = _load_summary_predictions(summary)
    sample_ids = predictions["sample_ids"]
    time = predictions["time"]
    event = predictions["event"]
    baseline_risk = predictions["selection_reference_risk"]
    selected_risk = predictions["selected_risk"]
    raw_mean_risk = predictions["raw_mean_risk"]
    residual_delta = selected_risk - baseline_risk

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    checkpoints = load_checkpoints(checkpoint_glob)
    device = resolve_device(device_arg)
    test_predictions = _predict_split(config, checkpoints, split_seed=split_seed, split="test", device=device)
    test_order = _index_by_sample_id(test_predictions.sample_ids)
    prediction_order = [test_order[sample_id] for sample_id in sample_ids]
    risk_matrix = test_predictions.risk_matrix[:, prediction_order]
    reference_weights = [1.0 / len(checkpoints) for _ in checkpoints]
    recomputed_raw_mean = _apply_weights(risk_matrix, reference_weights)

    top3_indices = summary["references"]["gnn_top3"]["indices"]
    disagreement_raw = _disagreement_features(risk_matrix, recomputed_raw_mean, top3_indices)
    disagreement_names = [
        "risk_std_all",
        "risk_range_all",
        "risk_std_top3",
        "risk_range_top3",
        "abs_top3_minus_raw_mean",
        "max_abs_member_minus_raw_mean",
    ]

    feature_names, train_features, _, test_features = _load_feature_splits(config, split_seed)
    train_scaled, _, test_scaled, _ = _standardize_feature_splits(
        train_features.features,
        train_features.features,
        test_features.features,
    )
    _ = train_scaled
    feature_order = _index_by_sample_id(test_features.sample_ids)
    test_scaled = test_scaled[[feature_order[sample_id] for sample_id in sample_ids]]
    modality_features = _modality_conflict_features(feature_names, test_scaled)

    sample_features: dict[str, np.ndarray] = {
        name: disagreement_raw[:, index] for index, name in enumerate(disagreement_names)
    }
    sample_features.update(modality_features)
    sample_features["abs_adapter_delta"] = np.abs(residual_delta)
    sample_features["signed_adapter_delta"] = residual_delta
    sample_features["abs_standardized_top3_minus_raw_mean"] = np.abs(baseline_risk - raw_mean_risk)

    pair_summary, sample_pair_rows = pairwise_change_diagnostics(
        sample_ids=sample_ids,
        time=time,
        event=event,
        baseline_risk=baseline_risk,
        selected_risk=selected_risk,
    )
    subgroup_rows = _build_subgroup_rows(
        sample_features=sample_features,
        time=time,
        event=event,
        baseline_risk=baseline_risk,
        selected_risk=selected_risk,
        residual_delta=residual_delta,
    )
    calibration = {
        "baseline": calibration_proxy(time=time, event=event, risk=baseline_risk),
        "selected": calibration_proxy(time=time, event=event, risk=selected_risk),
    }
    calibration["selected_delta_vs_baseline"] = {
        "top_event_lift_delta": (
            calibration["selected"]["top_event_lift"] - calibration["baseline"]["top_event_lift"]
        ),
        "high_low_event_gap_delta": (
            calibration["selected"]["high_low_event_gap"] - calibration["baseline"]["high_low_event_gap"]
        ),
        "risk_event_monotonic_spearman_delta": (
            calibration["selected"]["risk_event_monotonic_spearman"]
            - calibration["baseline"]["risk_event_monotonic_spearman"]
        ),
        "risk_time_monotonic_spearman_delta": (
            calibration["selected"]["risk_time_monotonic_spearman"]
            - calibration["baseline"]["risk_time_monotonic_spearman"]
        ),
    }
    reclassification = continuous_nri_proxy(event=event, baseline_risk=baseline_risk, selected_risk=selected_risk)
    correlations = _feature_correlations(sample_features, residual_delta)
    top_samples = _top_sample_pair_rows(
        sample_pair_rows=sample_pair_rows,
        sample_ids=sample_ids,
        time=time,
        event=event,
        baseline_risk=baseline_risk,
        selected_risk=selected_risk,
        residual_delta=residual_delta,
        sample_features=sample_features,
        feature_names=[
            "risk_std_all",
            "abs_top3_minus_raw_mean",
            "modality_conflict_range",
            "abs_adapter_delta",
        ],
    )

    result = {
        "summary_path": str(summary_path),
        "config_path": config_path,
        "checkpoint_glob": checkpoint_glob,
        "split_seed": split_seed,
        "device": str(device),
        "selected": summary["selected"],
        "references": summary["references"],
        "baseline_risk_field": predictions["baseline_risk_field"],
        "sample_count": len(sample_ids),
        "risk_consistency_checks": {
            "max_abs_raw_mean_summary_vs_recomputed": float(np.max(np.abs(raw_mean_risk - recomputed_raw_mean))),
        },
        "global_pair_change": pair_summary.__dict__,
        "calibration_proxy": calibration,
        "continuous_net_reclassification_proxy": reclassification,
        "subgroup_diagnostics": subgroup_rows,
        "feature_correlations_with_adapter_delta": correlations,
        "top_samples_by_pair_net_gain": top_samples["gainers"],
        "top_samples_by_pair_net_harm": top_samples["harmed"],
        "interpretation": (
            "Diagnostics compare the selected risk adapter against the saved selection reference baseline. "
            "Pair-level corrected/harmed counts show whether the adapter changes event ordering in a "
            "localized, interpretable way; subgroup rows test whether gains concentrate in disagreement "
            "or modality-conflict strata."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    return _jsonable(result)


def pairwise_change_diagnostics(
    *,
    sample_ids: Sequence[str],
    time: np.ndarray,
    event: np.ndarray,
    baseline_risk: np.ndarray,
    selected_risk: np.ndarray,
) -> tuple[PairChangeSummary, list[dict[str, Any]]]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    baseline = np.asarray(baseline_risk, dtype=float)
    selected = np.asarray(selected_risk, dtype=float)
    sample_rows = [
        {
            "sample_id": str(sample_id),
            "corrected_pairs": 0,
            "harmed_pairs": 0,
            "net_pair_score_delta": 0.0,
            "permissible_pairs": 0,
        }
        for sample_id in sample_ids
    ]
    corrected = 0
    harmed = 0
    tied_to_correct = 0
    correct_to_tied = 0
    unchanged = 0
    baseline_score_sum = 0.0
    selected_score_sum = 0.0
    permissible = 0

    for i, j, direction in _iter_permissible_pairs(time, event):
        baseline_score = _pair_score(baseline[i], baseline[j], direction)
        selected_score = _pair_score(selected[i], selected[j], direction)
        delta = selected_score - baseline_score
        baseline_score_sum += baseline_score
        selected_score_sum += selected_score
        permissible += 1
        sample_rows[i]["permissible_pairs"] += 1
        sample_rows[j]["permissible_pairs"] += 1
        sample_rows[i]["net_pair_score_delta"] += delta
        sample_rows[j]["net_pair_score_delta"] += delta
        if selected_score > baseline_score:
            corrected += 1
            sample_rows[i]["corrected_pairs"] += 1
            sample_rows[j]["corrected_pairs"] += 1
            if baseline_score == 0.5 and selected_score == 1.0:
                tied_to_correct += 1
        elif selected_score < baseline_score:
            harmed += 1
            sample_rows[i]["harmed_pairs"] += 1
            sample_rows[j]["harmed_pairs"] += 1
            if baseline_score == 1.0 and selected_score == 0.5:
                correct_to_tied += 1
        else:
            unchanged += 1

    baseline_c = baseline_score_sum / permissible if permissible else 0.0
    selected_c = selected_score_sum / permissible if permissible else 0.0
    summary = PairChangeSummary(
        permissible_pairs=int(permissible),
        baseline_c_index=float(baseline_c),
        selected_c_index=float(selected_c),
        c_index_delta=float(selected_c - baseline_c),
        corrected_pairs=int(corrected),
        harmed_pairs=int(harmed),
        tied_to_correct_pairs=int(tied_to_correct),
        correct_to_tied_pairs=int(correct_to_tied),
        unchanged_pairs=int(unchanged),
        net_corrected_pairs=int(corrected - harmed),
    )
    return summary, sample_rows


def calibration_proxy(
    *,
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    num_bins: int = 5,
) -> dict[str, Any]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    bins = _risk_quantile_bins(risk, num_bins=num_bins)
    rows = []
    overall_event_rate = float(np.mean(event)) if len(event) else 0.0
    for bin_index in range(num_bins):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": int(bin_index + 1),
                "n": int(mask.sum()),
                "risk_min": float(np.min(risk[mask])),
                "risk_max": float(np.max(risk[mask])),
                "mean_risk": float(np.mean(risk[mask])),
                "event_rate": float(np.mean(event[mask])),
                "mean_time": float(np.mean(time[mask])),
            }
        )
    if not rows:
        return {
            "num_bins": int(num_bins),
            "bins": [],
            "top_event_lift": 0.0,
            "high_low_event_gap": 0.0,
            "risk_event_monotonic_spearman": 0.0,
            "risk_time_monotonic_spearman": 0.0,
        }
    mean_risk = np.asarray([row["mean_risk"] for row in rows], dtype=float)
    event_rate = np.asarray([row["event_rate"] for row in rows], dtype=float)
    mean_time = np.asarray([row["mean_time"] for row in rows], dtype=float)
    low = rows[0]
    high = rows[-1]
    return {
        "num_bins": int(num_bins),
        "overall_event_rate": overall_event_rate,
        "bins": rows,
        "top_event_lift": float(high["event_rate"] / max(overall_event_rate, 1e-12)),
        "high_low_event_gap": float(high["event_rate"] - low["event_rate"]),
        "risk_event_monotonic_spearman": _spearman(mean_risk, event_rate),
        "risk_time_monotonic_spearman": _spearman(mean_risk, -mean_time),
    }


def continuous_nri_proxy(
    *,
    event: np.ndarray,
    baseline_risk: np.ndarray,
    selected_risk: np.ndarray,
) -> dict[str, float]:
    event = np.asarray(event, dtype=float)
    baseline = np.asarray(baseline_risk, dtype=float)
    selected = np.asarray(selected_risk, dtype=float)
    delta = selected - baseline
    event_mask = event == 1.0
    nonevent_mask = event == 0.0

    def rate(mask: np.ndarray, condition: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom == 0:
            return 0.0
        return float(np.mean(condition[mask]))

    event_up = rate(event_mask, delta > 0.0)
    event_down = rate(event_mask, delta < 0.0)
    nonevent_down = rate(nonevent_mask, delta < 0.0)
    nonevent_up = rate(nonevent_mask, delta > 0.0)
    return {
        "event_up_rate": event_up,
        "event_down_rate": event_down,
        "nonevent_down_rate": nonevent_down,
        "nonevent_up_rate": nonevent_up,
        "event_component": event_up - event_down,
        "nonevent_component": nonevent_down - nonevent_up,
        "continuous_nri": (event_up - event_down) + (nonevent_down - nonevent_up),
    }


def _build_subgroup_rows(
    *,
    sample_features: dict[str, np.ndarray],
    time: np.ndarray,
    event: np.ndarray,
    baseline_risk: np.ndarray,
    selected_risk: np.ndarray,
    residual_delta: np.ndarray,
    min_samples: int = 30,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name, values in sample_features.items():
        values = np.asarray(values, dtype=float)
        if len(np.unique(values)) < 3:
            continue
        quantiles = np.quantile(values, [0.25, 0.50, 0.75])
        buckets = [
            ("low_q25", values <= quantiles[0]),
            ("mid_q25_q75", (values > quantiles[0]) & (values < quantiles[2])),
            ("high_q75", values >= quantiles[2]),
        ]
        for bucket_name, mask in buckets:
            if int(mask.sum()) < min_samples:
                continue
            baseline_c = concordance_index(time[mask], event[mask], baseline_risk[mask])
            selected_c = concordance_index(time[mask], event[mask], selected_risk[mask])
            pair_summary, _ = pairwise_change_diagnostics(
                sample_ids=[str(index) for index in range(int(mask.sum()))],
                time=time[mask],
                event=event[mask],
                baseline_risk=baseline_risk[mask],
                selected_risk=selected_risk[mask],
            )
            rows.append(
                {
                    "feature": feature_name,
                    "bucket": bucket_name,
                    "n": int(mask.sum()),
                    "event_rate": float(np.mean(event[mask])),
                    "feature_mean": float(np.mean(values[mask])),
                    "feature_min": float(np.min(values[mask])),
                    "feature_max": float(np.max(values[mask])),
                    "mean_abs_adapter_delta": float(np.mean(np.abs(residual_delta[mask]))),
                    "baseline_c_index": baseline_c,
                    "selected_c_index": selected_c,
                    "c_index_delta": selected_c - baseline_c,
                    "pair_change": pair_summary.__dict__,
                }
            )
    rows.sort(key=lambda row: (row["c_index_delta"], row["selected_c_index"]), reverse=True)
    return rows


def _risk_quantile_bins(risk: np.ndarray, num_bins: int) -> np.ndarray:
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="mergesort")
    bins = np.empty(len(risk), dtype=int)
    for position, index in enumerate(order):
        bins[index] = min(num_bins - 1, int(position * num_bins / max(len(risk), 1)))
    return bins


def _feature_correlations(sample_features: dict[str, np.ndarray], residual_delta: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    residual_delta = np.asarray(residual_delta, dtype=float)
    abs_delta = np.abs(residual_delta)
    for feature_name, values in sample_features.items():
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "feature": feature_name,
                "spearman_with_abs_adapter_delta": _spearman(values, abs_delta),
                "spearman_with_signed_adapter_delta": _spearman(values, residual_delta),
            }
        )
    rows.sort(key=lambda row: abs(row["spearman_with_abs_adapter_delta"]), reverse=True)
    return rows


def _top_sample_pair_rows(
    *,
    sample_pair_rows: list[dict[str, Any]],
    sample_ids: Sequence[str],
    time: np.ndarray,
    event: np.ndarray,
    baseline_risk: np.ndarray,
    selected_risk: np.ndarray,
    residual_delta: np.ndarray,
    sample_features: dict[str, np.ndarray],
    feature_names: Sequence[str],
    top_n: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, pair_row in enumerate(sample_pair_rows):
        row = {
            **pair_row,
            "time": float(time[index]),
            "event": float(event[index]),
            "baseline_risk": float(baseline_risk[index]),
            "selected_risk": float(selected_risk[index]),
            "adapter_delta": float(residual_delta[index]),
        }
        for feature_name in feature_names:
            row[feature_name] = float(sample_features[feature_name][index])
        rows.append(row)
    gainers = sorted(rows, key=lambda row: row["net_pair_score_delta"], reverse=True)[:top_n]
    harmed = sorted(rows, key=lambda row: row["net_pair_score_delta"])[:top_n]
    return {"gainers": gainers, "harmed": harmed}


def _modality_conflict_features(feature_names: Sequence[str], features: np.ndarray) -> dict[str, np.ndarray]:
    modality_indices = {
        "clinical": [index for index, name in enumerate(feature_names) if name.startswith("clinical:")],
        "metabolite": [index for index, name in enumerate(feature_names) if name.startswith("metabolite:")],
        "graph": [index for index, name in enumerate(feature_names) if name.startswith("graph:")],
    }
    modality_scores = {
        name: np.mean(features[:, indices], axis=1)
        for name, indices in modality_indices.items()
        if indices
    }
    score_matrix = np.vstack([modality_scores[name] for name in sorted(modality_scores)]).T
    result = {
        f"modality_score_{name}": values for name, values in modality_scores.items()
    }
    result["modality_conflict_std"] = np.std(score_matrix, axis=1)
    result["modality_conflict_range"] = np.max(score_matrix, axis=1) - np.min(score_matrix, axis=1)
    if {"clinical", "metabolite"}.issubset(modality_scores):
        result["abs_clinical_metabolite_gap"] = np.abs(modality_scores["clinical"] - modality_scores["metabolite"])
    if {"clinical", "graph"}.issubset(modality_scores):
        result["abs_clinical_graph_gap"] = np.abs(modality_scores["clinical"] - modality_scores["graph"])
    if {"metabolite", "graph"}.issubset(modality_scores):
        result["abs_metabolite_graph_gap"] = np.abs(modality_scores["metabolite"] - modality_scores["graph"])
    return result


def _load_summary_predictions(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["test_predictions"]
    has_selection_reference = bool(rows) and "selection_reference_risk" in rows[0]
    return {
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "time": np.asarray([row["time"] for row in rows], dtype=float),
        "event": np.asarray([row["event"] for row in rows], dtype=float),
        "raw_mean_risk": np.asarray([row["raw_mean_risk"] for row in rows], dtype=float),
        "gnn_top3_risk": np.asarray([row["gnn_top3_risk"] for row in rows], dtype=float),
        "selection_reference_risk": np.asarray(
            [
                row["selection_reference_risk"]
                if has_selection_reference
                else row["gnn_top3_risk"]
                for row in rows
            ],
            dtype=float,
        ),
        "baseline_risk_field": (
            "selection_reference_risk"
            if has_selection_reference
            else "gnn_top3_risk"
        ),
        "selected_risk": np.asarray([row["selected_risk"] for row in rows], dtype=float),
    }


def _iter_permissible_pairs(time: np.ndarray, event: np.ndarray):
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if event[i] == 0 and event[j] == 0:
                continue
            if time[i] < time[j] and event[i] == 1:
                yield i, j, 1
            elif time[j] < time[i] and event[j] == 1:
                yield i, j, -1


def _pair_score(risk_i: float, risk_j: float, direction: int) -> float:
    if risk_i == risk_j:
        return 0.5
    if direction == 1:
        return 1.0 if risk_i > risk_j else 0.0
    return 1.0 if risk_j > risk_i else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return 0.0
    x_rank = _rankdata(np.asarray(x, dtype=float))
    y_rank = _rankdata(np.asarray(y, dtype=float))
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _index_by_sample_id(sample_ids: Sequence[str]) -> dict[str, int]:
    return {str(sample_id): index for index, sample_id in enumerate(sample_ids)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/current_mainline_v2/risk_adapter_v2_dual_baseline/risk_adapter_v2_dual_baseline_summary.json",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-glob", default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/risk_adapter_diagnostics_v2/risk_adapter_diagnostics_v2_summary.json",
    )
    args = parser.parse_args()
    result = run_risk_adapter_diagnostics(
        summary_path=args.summary,
        config_path=args.config,
        checkpoint_glob=args.checkpoint_glob,
        split_seed=args.split_seed,
        device_arg=args.device,
        output_path=args.output,
    )
    print(json.dumps(result["global_pair_change"], indent=2))


if __name__ == "__main__":
    main()
