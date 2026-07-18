from __future__ import annotations

import math
import re
from typing import Any, Dict


SMOKING_TRUE = {"yes", "true", "1", "current", "former", "ever"}
FAMILY_HISTORY_TRUE = {"yes", "true", "1", "positive"}
SMOKING_FALSE = {"no", "false", "0", "never"}
FAMILY_HISTORY_FALSE = {"no", "false", "0", "negative"}
GENERAL_TRUE = {"yes", "true", "1", "positive", "present"}
GENERAL_FALSE = {"no", "false", "0", "negative", "absent"}
EXPLICIT_NONE_VALUES = {
    "无",
    "没有",
    "否认",
    "none",
    "no",
    "nil",
    "n/a",
    "na",
    "no known allergies",
    "no current medications",
}


def _to_float(value: Any, *, field_name: str, required: bool = False) -> float | None:
    if value is None or value == "":
        if required:
            raise ValueError(f"{field_name} 缺少数值。")
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是有效数字。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有效数字，当前值为 {value!r}。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 不能是 NaN 或 Infinity。")
    return number


def _to_binary(value: Any, truthy: set[str], falsy: set[str], *, field_name: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in truthy:
        return 1.0
    if text in falsy:
        return 0.0
    raise ValueError(f"{field_name} 只能填写明确的是/否、true/false 或 1/0，当前值为 {value!r}。")


def _normalize_microbe_payload(microbe_payload: Any) -> Dict[str, float]:
    if isinstance(microbe_payload, dict):
        return {
            str(name): float(
                _to_float(value, field_name=f"oral_microbiome.{name}", required=True)
            )
            for name, value in microbe_payload.items()
        }

    microbes: Dict[str, float] = {}
    if isinstance(microbe_payload, list):
        for index, item in enumerate(microbe_payload):
            if not isinstance(item, dict):
                raise ValueError(f"oral_microbiome.taxa[{index}] 必须是 JSON 对象。")
            name = item.get("taxon") or item.get("name") or item.get("node_name")
            if name is None:
                raise ValueError(f"oral_microbiome.taxa[{index}] 缺少菌群名称。")
            abundance = next(
                (
                    item[key]
                    for key in ("abundance", "relative_abundance", "value")
                    if key in item
                ),
                None,
            )
            microbes[str(name)] = float(
                _to_float(
                    abundance,
                    field_name=f"oral_microbiome.taxa[{index}].abundance",
                    required=True,
                )
            )
    return microbes


def _set_optional_number(target: Dict[str, float], key: str, value: Any, field_name: str) -> None:
    parsed = _to_float(value, field_name=field_name)
    if parsed is not None:
        target[key] = parsed


def _set_optional_binary(
    target: Dict[str, float],
    key: str,
    value: Any,
    truthy: set[str],
    falsy: set[str],
    field_name: str,
) -> None:
    if value is None or value == "":
        return
    target[key] = _to_binary(value, truthy, falsy, field_name=field_name)


def _to_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = re.split(r"[,，;；\n]", value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{field_name} 必须是字符串列表或逗号分隔文本。")

    result: list[str] = []
    for index, item in enumerate(values):
        if isinstance(item, (dict, list, tuple)):
            raise ValueError(f"{field_name}[{index}] 必须是文本。")
        text = str(item).strip()
        if text and text.lower() not in EXPLICIT_NONE_VALUES:
            result.append(text)
    return result


def _object_section(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} 必须是 JSON 对象。")
    return value


def standardize_raw_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    demographics = _object_section(payload, "demographics")
    history = _object_section(payload, "history")
    oral_microbiome = _object_section(payload, "oral_microbiome")
    metabolites_raw = _object_section(payload, "metabolites")
    context = _object_section(payload, "clinical_context")
    medication_context = _object_section(payload, "medication_context")

    microbes = _normalize_microbe_payload(oral_microbiome.get("taxa", oral_microbiome))

    clinical: Dict[str, float] = {}
    _set_optional_number(clinical, "age", demographics.get("age"), "demographics.age")
    _set_optional_number(clinical, "bmi", demographics.get("bmi"), "demographics.bmi")
    _set_optional_binary(
        clinical,
        "smoking",
        history.get("smoking"),
        SMOKING_TRUE,
        SMOKING_FALSE,
        "history.smoking",
    )
    _set_optional_binary(
        clinical,
        "family_history",
        history.get("family_history_colorectal_or_ibd"),
        FAMILY_HISTORY_TRUE,
        FAMILY_HISTORY_FALSE,
        "history.family_history_colorectal_or_ibd",
    )

    metabolites: Dict[str, float] = {}
    _set_optional_number(
        metabolites,
        "bile_acids",
        metabolites_raw.get("bile_acids"),
        "metabolites.bile_acids",
    )
    _set_optional_number(metabolites, "scfa", metabolites_raw.get("scfa"), "metabolites.scfa")
    _set_optional_number(
        metabolites,
        "tryptophan_metabolism",
        metabolites_raw.get("tryptophan_metabolism"),
        "metabolites.tryptophan_metabolism",
    )

    metadata: Dict[str, Any] = {
        "sample_id": str(payload.get("sample_id", context.get("sample_id", "unknown"))),
        "sex": demographics.get("sex", "unknown"),
        "chief_complaint": context.get("chief_complaint", ""),
        "suspected_condition": context.get("suspected_condition", "gut_risk_screening"),
    }

    list_fields = {
        "current_medications": medication_context.get(
            "current_medications", history.get("current_medications")
        ),
        "drug_allergies": medication_context.get(
            "drug_allergies", history.get("drug_allergies")
        ),
    }
    for field, value in list_fields.items():
        if value is not None:
            metadata[field] = _to_string_list(
                value,
                field_name=f"medication_context.{field}",
            )

    binary_fields = {
        "recent_antibiotics": history.get("recent_antibiotics"),
        "recent_probiotics": history.get("recent_probiotics"),
        "renal_impairment": medication_context.get(
            "renal_impairment", history.get("renal_impairment")
        ),
        "hepatic_impairment": medication_context.get(
            "hepatic_impairment", history.get("hepatic_impairment")
        ),
        "pregnancy": medication_context.get("pregnancy", demographics.get("pregnancy")),
    }
    for field, value in binary_fields.items():
        if value is not None and value != "":
            metadata[field] = _to_binary(
                value,
                GENERAL_TRUE,
                GENERAL_FALSE,
                field_name=f"medication_context.{field}",
            )

    return {
        "microbes": microbes,
        "clinical": clinical,
        "metabolites": metabolites,
        "metadata": metadata,
    }
