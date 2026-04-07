from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class StructuredInput:
    microbes: Dict[str, float]
    clinical: Dict[str, float]
    metabolites: Dict[str, float]


def _coerce_numeric_map(payload: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def normalize_relative_abundance(microbes: Dict[str, float]) -> Dict[str, float]:
    if not microbes:
        return {}
    total = sum(max(v, 0.0) for v in microbes.values())
    if total <= 0:
        return {k: 0.0 for k in microbes}
    return {k: max(v, 0.0) / total for k, v in microbes.items()}


def zscore_like(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    series = pd.Series(values, dtype=float)
    std = float(series.std(ddof=0))
    if std == 0:
        return {k: float(v) for k, v in values.items()}
    transformed = (series - float(series.mean())) / std
    return transformed.to_dict()


def build_structured_input(payload: Dict[str, Any]) -> StructuredInput:
    microbes = normalize_relative_abundance(_coerce_numeric_map(payload.get("microbes", {})))
    clinical = _coerce_numeric_map(payload.get("clinical", {}))
    metabolites = _coerce_numeric_map(payload.get("metabolites", {}))
    return StructuredInput(microbes=microbes, clinical=clinical, metabolites=metabolites)
