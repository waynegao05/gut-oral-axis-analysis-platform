from __future__ import annotations

import statistics
from typing import Sequence

import numpy as np
from scipy.stats import ttest_rel


def concordance_index(time, event, risk) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    permissible = 0.0

    for left in range(len(time)):
        for right in range(left + 1, len(time)):
            if time[left] == time[right]:
                continue
            if time[left] < time[right] and event[left] == 1:
                permissible += 1
                concordant += _pair_credit(risk[left], risk[right])
            elif time[right] < time[left] and event[right] == 1:
                permissible += 1
                concordant += _pair_credit(risk[right], risk[left])

    return float(concordant / permissible) if permissible else 0.0


def _pair_credit(earlier_risk: float, later_risk: float) -> float:
    if earlier_risk > later_risk:
        return 1.0
    if earlier_risk == later_risk:
        return 0.5
    return 0.0


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def summarize_paired_folds(
    baseline: Sequence[float],
    ctm: Sequence[float],
) -> dict[str, object]:
    if len(baseline) != len(ctm):
        raise ValueError("baseline and ctm fold scores must have the same length.")
    if not baseline:
        raise ValueError("At least one fold score is required.")

    deltas = [float(ctm_score - baseline_score) for baseline_score, ctm_score in zip(baseline, ctm)]
    if len(deltas) > 1:
        delta_std = float(np.std(deltas))
        if np.isclose(delta_std, 0.0):
            mean_delta = float(statistics.mean(deltas))
            statistic = 0.0 if np.isclose(mean_delta, 0.0) else float(np.sign(mean_delta) * np.finfo(float).max)
            p_value = 1.0 if np.isclose(mean_delta, 0.0) else 0.0
        else:
            result = ttest_rel(ctm, baseline)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
    else:
        statistic = 0.0
        p_value = 1.0

    return {
        "baseline": _summary(baseline),
        "ctm": _summary(ctm),
        "fold_deltas": deltas,
        "mean_delta": float(statistics.mean(deltas)),
        "paired_t_test": {
            "num_pairs": len(deltas),
            "statistic": statistic,
            "p_value": p_value,
            "zero_variance_delta": bool(len(deltas) > 1 and np.isclose(float(np.std(deltas)), 0.0)),
        },
    }
