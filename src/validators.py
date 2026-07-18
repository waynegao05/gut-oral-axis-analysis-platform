from __future__ import annotations

import math
import re
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
METADATA_BINARY_FIELDS: dict[str, str] = {
    "recent_antibiotics": "近期抗生素暴露",
    "recent_probiotics": "近期益生菌暴露",
    "renal_impairment": "肾功能异常",
    "hepatic_impairment": "肝功能异常",
    "pregnancy": "妊娠状态",
}
METADATA_LIST_FIELDS: dict[str, str] = {
    "current_medications": "当前用药清单",
    "drug_allergies": "药物过敏史",
}
MAX_METADATA_LIST_ITEMS = 100
MAX_METADATA_TEXT_LENGTH = 500
NEGATIVE_MEDICATION_QUANTITY = re.compile(
    r"(?<![\w.])-\s*\d+(?:\.\d+)?\s*(?:mcg|μg|ug|mg|g|ml|iu|units?)\b",
    flags=re.IGNORECASE,
)


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


def _validate_metadata(metadata: Dict[str, Any], errors: List[str]) -> None:
    for field, label in METADATA_BINARY_FIELDS.items():
        if field not in metadata:
            continue
        field_path = f"metadata.{field}"
        number = _finite_number(metadata[field], field_path, errors)
        if number is not None and number not in {0.0, 1.0}:
            errors.append(f"{field_path}（{label}）只能是 0 或 1，当前值为 {number:g}。")

    for field, label in METADATA_LIST_FIELDS.items():
        if field not in metadata:
            continue
        field_path = f"metadata.{field}"
        values = metadata[field]
        if not isinstance(values, list):
            errors.append(f"{field_path}（{label}）必须是字符串列表。")
            continue
        if len(values) > MAX_METADATA_LIST_ITEMS:
            errors.append(
                f"{field_path}（{label}）最多允许 {MAX_METADATA_LIST_ITEMS} 项。"
            )
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field_path}[{index}] 必须是非空文本。")
                continue
            if len(value) > MAX_METADATA_TEXT_LENGTH:
                errors.append(
                    f"{field_path}[{index}] 最多允许 {MAX_METADATA_TEXT_LENGTH} 个字符。"
                )
            if field == "current_medications" and NEGATIVE_MEDICATION_QUANTITY.search(value):
                errors.append(
                    f"{field_path}[{index}] 含有负数剂量或规格，当前值为 {value!r}。"
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
    metadata = payload.get("metadata")

    if isinstance(microbes, dict):
        _validate_microbes(microbes, errors)
    if isinstance(clinical, dict):
        _validate_clinical(clinical, errors)
    if isinstance(metabolites, dict):
        _validate_metabolites(metabolites, errors)
    if metadata is not None and not isinstance(metadata, dict):
        errors.append("字段 metadata 必须是 JSON 对象。")
    elif isinstance(metadata, dict):
        _validate_metadata(metadata, errors)

    return len(errors) == 0, errors
