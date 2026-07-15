from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_cindex_diagnostics(
    time: Any,
    event: Any,
    baseline_risk: Any,
    candidate_risk: Any,
) -> dict[str, float]:
    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=float)
    baseline_array = np.asarray(baseline_risk, dtype=float)
    candidate_array = np.asarray(candidate_risk, dtype=float)
    if not (
        len(time_array)
        == len(event_array)
        == len(baseline_array)
        == len(candidate_array)
    ):
        raise ValueError("time, event, baseline_risk, and candidate_risk must have the same length.")

    permissible = 0
    baseline_credit = 0.0
    candidate_credit = 0.0
    improved_pairs = 0
    regressed_pairs = 0
    unchanged_pairs = 0
    for left in range(len(time_array)):
        for right in range(left + 1, len(time_array)):
            if time_array[left] == time_array[right]:
                continue
            if time_array[left] < time_array[right] and event_array[left] == 1:
                earlier = left
                later = right
            elif time_array[right] < time_array[left] and event_array[right] == 1:
                earlier = right
                later = left
            else:
                continue

            permissible += 1
            base = _credit(baseline_array[earlier], baseline_array[later])
            cand = _credit(candidate_array[earlier], candidate_array[later])
            baseline_credit += base
            candidate_credit += cand
            if cand > base:
                improved_pairs += 1
            elif cand < base:
                regressed_pairs += 1
            else:
                unchanged_pairs += 1

    return {
        "permissible_pairs": float(permissible),
        "baseline_pair_credit": baseline_credit,
        "candidate_pair_credit": candidate_credit,
        "pair_credit_delta": candidate_credit - baseline_credit,
        "improved_pairs": float(improved_pairs),
        "regressed_pairs": float(regressed_pairs),
        "unchanged_pairs": float(unchanged_pairs),
        "net_improved_pairs": float(improved_pairs - regressed_pairs),
    }


def _credit(earlier_risk: float, later_risk: float) -> float:
    if earlier_risk > later_risk:
        return 1.0
    if earlier_risk == later_risk:
        return 0.5
    return 0.0
