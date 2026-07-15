from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


REQUIRED_TOP_LEVEL_KEYS = ["microbes", "clinical", "metabolites"]

CLINICAL_RANGES: dict[str, tuple[float, float, str]] = {
    "age": (1.0, 120.0, "年龄"),
    "bmi": (5.0, 100.0, "BMI"),
}
BINARY_CLINICAL_FIELDS: dict[str, str] = {
    "smoking": "吸烟状态",
    "family_history": "家族史",
}
METABOLITE_FIELDS: dict[str, str] = {
    "bile_acids": "胆汁酸",
    "scfa": "短链脂肪酸（SCFA）",
    "tryptophan_metabolism": "色氨酸代谢",
}


def _finite_number(value: Any, field_path: str, errors: List[str]) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        errors.append(f"{field_path} 必须是有效数字。")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_path} 必须是有效数字，当前值为 {value!r}。")
        return None
    if not math.isfinite(number):
        errors.append(f"{field_path} 不能是 NaN 或 Infinity。")
        return None
    return number


def _validate_microbes(microbes: Dict[str, Any], errors: List[str]) -> None:
    if not microbes:
        errors.append("字段 microbes 不能为空。")
        return

    positive_count = 0
    for raw_name, value in microbes.items():
        name = str(raw_name).strip()
        if not name:
            errors.append("microbes 中的菌群名称不能为空。")
            continue
        field_path = f"microbes.{name}"
        number = _finite_number(value, field_path, errors)
        if number is None:
            continue
        if number < 0.0 or number > 1.0:
            errors.append(
                f"{field_path} 必须是 0 到 1 之间的标准化丰度，当前值为 {number:g}。"
            )
            continue
        if number > 0.0:
            positive_count += 1

    if positive_count == 0 and not any(message.startswith("microbes.") for message in errors):
        errors.append("microbes 至少需要一个大于 0 的菌群丰度。")


def _validate_clinical(clinical: Dict[str, Any], errors: List[str]) -> None:
    known_fields = set(CLINICAL_RANGES) | set(BINARY_CLINICAL_FIELDS)
    for field, value in clinical.items():
        if field not in known_fields:
            _finite_number(value, f"clinical.{field}", errors)

    for field, (minimum, maximum, label) in CLINICAL_RANGES.items():
        if field not in clinical:
            continue
        field_path = f"clinical.{field}"
        number = _finite_number(clinical[field], field_path, errors)
        if number is None:
            continue
        if number < minimum or number > maximum:
            errors.append(
                f"{field_path}（{label}）必须在 {minimum:g} 到 {maximum:g} 之间，当前值为 {number:g}。"
            )

    for field, label in BINARY_CLINICAL_FIELDS.items():
        if field not in clinical:
            continue
        field_path = f"clinical.{field}"
        number = _finite_number(clinical[field], field_path, errors)
        if number is not None and number not in {0.0, 1.0}:
            errors.append(f"{field_path}（{label}）只能是 0 或 1，当前值为 {number:g}。")


def _validate_metabolites(metabolites: Dict[str, Any], errors: List[str]) -> None:
    for field, value in metabolites.items():
        label = METABOLITE_FIELDS.get(field, field)
        field_path = f"metabolites.{field}"
        number = _finite_number(value, field_path, errors)
        if number is None:
            continue
        if number < 0.0 or number > 1.0:
            errors.append(
                f"{field_path}（{label}）必须是 0 到 1 之间的标准化数值，当前值为 {number:g}。"
            )


def validate_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["输入必须是 JSON 对象。"]

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            errors.append(f"缺少顶层字段：{key}。")
        elif not isinstance(payload[key], dict):
            errors.append(f"字段 {key} 必须是 JSON 对象。")

    microbes = payload.get("microbes")
    clinical = payload.get("clinical")
    metabolites = payload.get("metabolites")

    if isinstance(microbes, dict):
        _validate_microbes(microbes, errors)
    if isinstance(clinical, dict):
        _validate_clinical(clinical, errors)
    if isinstance(metabolites, dict):
        _validate_metabolites(metabolites, errors)

    return len(errors) == 0, errors
