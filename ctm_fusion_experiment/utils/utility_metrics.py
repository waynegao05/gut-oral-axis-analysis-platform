from __future__ import annotations

import math
from typing import Any

import numpy as np

from ctm_fusion_experiment.utils.metrics import concordance_index


def risk_utility_metrics(
    time: Any,
    event: Any,
    risk: Any,
    *,
    top_fraction: float = 0.1,
) -> dict[str, float]:
    """Report c-index plus simple risk-stratification utility metrics.

    These metrics are deliberately operational rather than clinical claims:
    they describe whether the model concentrates observed events into the
    highest-risk band and separates high-risk from low-risk samples.
    """

    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=float)
    risk_array = np.asarray(risk, dtype=float)
    if not (len(time_array) == len(event_array) == len(risk_array)):
        raise ValueError("time, event, and risk must have the same length.")
    if len(risk_array) == 0:
        raise ValueError("At least one sample is required.")
    if not 0.0 < float(top_fraction) <= 0.5:
        raise ValueError("top_fraction must be in (0, 0.5].")

    band_size = max(1, int(math.ceil(len(risk_array) * float(top_fraction))))
    high_order = np.argsort(-risk_array, kind="mergesort")
    low_order = np.argsort(risk_array, kind="mergesort")
    top_index = high_order[:band_size]
    bottom_index = low_order[:band_size]
    overall_event_rate = float(np.mean(event_array))
    top_event_rate = float(np.mean(event_array[top_index]))
    bottom_event_rate = float(np.mean(event_array[bottom_index]))
    top_mean_time = float(np.mean(time_array[top_index]))
    bottom_mean_time = float(np.mean(time_array[bottom_index]))
    lift = top_event_rate / overall_event_rate if overall_event_rate > 0.0 else 0.0

    return {
        "c_index": concordance_index(time_array, event_array, risk_array),
        "risk_mean": float(np.mean(risk_array)),
        "risk_std": float(np.std(risk_array)),
        "overall_event_rate": overall_event_rate,
        "top_fraction": float(top_fraction),
        "top_count": float(band_size),
        "top_event_rate": top_event_rate,
        "bottom_event_rate": bottom_event_rate,
        "top_event_lift": float(lift),
        "high_low_event_gap": top_event_rate - bottom_event_rate,
        "top_mean_time": top_mean_time,
        "bottom_mean_time": bottom_mean_time,
        "time_separation": bottom_mean_time - top_mean_time,
    }
