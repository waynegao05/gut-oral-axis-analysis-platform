from __future__ import annotations

from typing import Any, Dict


SMOKING_TRUE = {"yes", "true", "1", "current", "former", "ever"}
FAMILY_HISTORY_TRUE = {"yes", "true", "1", "positive"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_binary(value: Any, truthy: set[str]) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    return 1.0 if text in truthy else 0.0


def _normalize_microbe_payload(microbe_payload: Any) -> Dict[str, float]:
    if isinstance(microbe_payload, dict):
        return {str(k): _to_float(v) for k, v in microbe_payload.items()}

    microbes: Dict[str, float] = {}
    if isinstance(microbe_payload, list):
        for item in microbe_payload:
            if not isinstance(item, dict):
                continue
            name = item.get("taxon") or item.get("name") or item.get("node_name")
            abundance = item.get("abundance") or item.get("relative_abundance") or item.get("value")
            if name is None:
                continue
            microbes[str(name)] = _to_float(abundance)
    return microbes


def standardize_raw_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    demographics = payload.get("demographics", {})
    history = payload.get("history", {})
    oral_microbiome = payload.get("oral_microbiome", {})
    metabolites_raw = payload.get("metabolites", {})
    context = payload.get("clinical_context", {})

    microbes = _normalize_microbe_payload(oral_microbiome.get("taxa", oral_microbiome))

    clinical = {
        "age": _to_float(demographics.get("age")),
        "bmi": _to_float(demographics.get("bmi")),
        "smoking": _to_binary(history.get("smoking"), SMOKING_TRUE),
        "family_history": _to_binary(history.get("family_history_colorectal_or_ibd"), FAMILY_HISTORY_TRUE),
    }

    metabolites = {
        "bile_acids": _to_float(metabolites_raw.get("bile_acids")),
        "scfa": _to_float(metabolites_raw.get("scfa")),
        "tryptophan_metabolism": _to_float(metabolites_raw.get("tryptophan_metabolism")),
    }

    metadata = {
        "sample_id": str(payload.get("sample_id", context.get("sample_id", "unknown"))),
        "sex": demographics.get("sex", "unknown"),
        "chief_complaint": context.get("chief_complaint", ""),
        "suspected_condition": context.get("suspected_condition", "gut_risk_screening"),
        "recent_antibiotics": _to_binary(history.get("recent_antibiotics"), SMOKING_TRUE),
        "recent_probiotics": _to_binary(history.get("recent_probiotics"), SMOKING_TRUE),
    }

    return {
        "microbes": microbes,
        "clinical": clinical,
        "metabolites": metabolites,
        "metadata": metadata,
    }
