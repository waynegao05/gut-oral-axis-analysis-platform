from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.drug_knowledge import build_drug_knowledge_review


KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "pharmacy_rules_v3.json"

_REQUIRED_ENGINE_SOURCE_IDS = {
    "AGA_PROBIOTICS_2020",
    "CDC_ANTIBIOTIC_STEWARDSHIP_2025",
    "FDA_CDS_2026",
    "INTERNAL_TOPOLOGY_V6",
    "ONC_HIGH_PRIORITY_DDI_2012",
    "OPENFDA_LABEL_CURRENT",
    "DAILYMED_SPL_CURRENT",
    "RXNORM_API_CURRENT",
    "USPSTF_CRC_2021",
    "WHO_MEDICATION_RECONCILIATION",
}

_MEDICATION_CONTEXT_LABELS = {
    "current_medications": "当前用药清单",
    "drug_allergies": "药物过敏史",
    "recent_antibiotics": "近期抗生素暴露",
    "recent_probiotics": "近期益生菌暴露",
    "renal_impairment": "肾功能异常状态",
    "hepatic_impairment": "肝功能异常状态",
    "pregnancy": "妊娠状态",
}

_MODEL_INPUT_LABELS = {
    "age": "年龄",
    "bmi": "BMI",
    "smoking": "吸烟状态",
    "family_history": "家族史",
    "bile_acids": "胆汁酸指标",
    "scfa": "短链脂肪酸（SCFA）指标",
    "tryptophan_metabolism": "色氨酸代谢指标",
}

_CONDITION_LABELS = {
    "antibiotic_c_difficile_prevention": "抗生素使用期间预防艰难梭菌感染",
    "c_difficile_prevention_during_antibiotics": "抗生素使用期间预防艰难梭菌感染",
    "pouchitis": "已确认储袋炎",
    "c_difficile_infection": "已确认艰难梭菌感染",
    "crohns_disease": "已确认克罗恩病",
    "ulcerative_colitis": "已确认溃疡性结肠炎",
    "irritable_bowel_syndrome": "已确认肠易激综合征",
    "ibs": "已确认肠易激综合征",
    "pediatric_acute_infectious_gastroenteritis": "儿童急性感染性胃肠炎",
}

_DISCLAIMER = (
    "本结果仅用于研究性药学辅助和临床复核，不构成诊断、处方或停换药指令。"
    "任何药物决策必须结合完整病史、检查、药品说明书和具备资质的临床人员判断。"
)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _model_input_label(field: Any) -> str:
    value = str(field).strip()
    normalized = value.removeprefix("clinical.").removeprefix("metabolites.")
    if normalized.startswith("abundance::"):
        return f"{normalized.removeprefix('abundance::')} 菌群丰度"
    return _MODEL_INPUT_LABELS.get(normalized, normalized or "未识别字段")


def _format_measurement(value: Any) -> str | None:
    number = _finite_float(value)
    if number is None:
        return None
    return f"{number:g}"


def _out_of_range_message(
    model_features: Mapping[str, Any],
    out_of_range_inputs: Sequence[str],
) -> str:
    raw_details = model_features.get("out_of_training_range_details", [])
    details = raw_details if isinstance(raw_details, list) else []
    readable_details: list[str] = []
    seen_fields: set[str] = set()
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        field = str(detail.get("field", "")).strip()
        if not field or field in seen_fields:
            continue
        seen_fields.add(field)
        value = _format_measurement(detail.get("value"))
        minimum = _format_measurement(detail.get("training_minimum"))
        maximum = _format_measurement(detail.get("training_maximum"))
        label = _model_input_label(field)
        if value is not None and minimum is not None and maximum is not None:
            readable_details.append(
                f"{label}输入为 {value}，当前模型研究数据范围为 {minimum} 至 {maximum}"
            )
        else:
            readable_details.append(label)

    if readable_details:
        detail_text = "；".join(readable_details)
    else:
        labels = [_model_input_label(field) for field in out_of_range_inputs]
        detail_text = "、".join(labels) if labels else "部分输入"
    return (
        f"以下内容超出当前模型研究数据范围：{detail_text}。"
        "请先核对数值和单位；如果原始值确实如此，不要为了获得结果而修改数据，本版本不适用于该范围。"
    )


def _binary_flag(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key, 0)
    if isinstance(value, bool):
        return value
    number = _finite_float(value)
    return number == 1.0


def _validate_knowledge_payload(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        raise ValueError("Pharmacy knowledge base root must be a JSON object.")

    required = {
        "schema_version",
        "engine_version",
        "last_reviewed",
        "intended_use",
        "calibration",
        "marker_rules",
        "evidence_sources",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Pharmacy knowledge base is missing fields: {', '.join(missing)}")

    calibration = payload["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("Pharmacy calibration must be a JSON object.")
    calibration_fields = {
        "dataset",
        "dataset_scope",
        "value_scale",
        "sample_count",
        "quantile_method",
        "quantile_interpolation",
        "normalization",
        "source_table",
        "required_marker_panel",
    }
    missing_calibration = sorted(calibration_fields.difference(calibration))
    if missing_calibration:
        raise ValueError(
            "Pharmacy calibration is missing fields: "
            f"{', '.join(missing_calibration)}"
        )
    panel = calibration["required_marker_panel"]
    if not isinstance(panel, list) or not panel or any(not str(item).strip() for item in panel):
        raise ValueError("Pharmacy required_marker_panel must contain marker names.")
    panel_names = [str(item) for item in panel]
    if len(panel_names) != len(set(panel_names)):
        raise ValueError("Pharmacy required_marker_panel contains duplicate markers.")

    sources = payload["evidence_sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("Pharmacy evidence_sources must be a non-empty list.")
    source_ids: list[str] = []
    required_source_fields = {
        "source_id",
        "organization",
        "title",
        "year",
        "source_type",
        "url",
        "scope_note",
    }
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Pharmacy evidence source {index} must be a JSON object.")
        missing_source = sorted(required_source_fields.difference(source))
        if missing_source:
            raise ValueError(
                f"Pharmacy evidence source {index} is missing fields: "
                f"{', '.join(missing_source)}"
            )
        source_id = str(source["source_id"]).strip()
        if not source_id:
            raise ValueError(f"Pharmacy evidence source {index} has an empty source_id.")
        source_ids.append(source_id)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Pharmacy evidence_sources contains duplicate source_id values.")
    source_id_set = set(source_ids)

    missing_engine_sources = sorted(_REQUIRED_ENGINE_SOURCE_IDS.difference(source_id_set))
    if missing_engine_sources:
        raise ValueError(
            "Pharmacy knowledge base is missing engine evidence sources: "
            f"{', '.join(missing_engine_sources)}"
        )

    rules = payload["marker_rules"]
    if not isinstance(rules, list):
        raise ValueError("Pharmacy marker_rules must be a list.")
    required_rule_fields = {
        "rule_id",
        "marker",
        "operator",
        "threshold",
        "threshold_quantile",
        "category",
        "title",
        "suggestion",
        "rationale",
        "base_priority",
        "urgency",
        "action_type",
        "evidence_level",
        "evidence_source_ids",
    }
    rule_ids: list[str] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Pharmacy marker rule {index} must be a JSON object.")
        missing_rule = sorted(required_rule_fields.difference(rule))
        if missing_rule:
            raise ValueError(
                f"Pharmacy marker rule {index} is missing fields: "
                f"{', '.join(missing_rule)}"
            )
        rule_id = str(rule["rule_id"]).strip()
        rule_ids.append(rule_id)
        if not rule_id:
            raise ValueError(f"Pharmacy marker rule {index} has an empty rule_id.")
        marker = str(rule["marker"])
        if marker not in panel_names:
            raise ValueError(f"Pharmacy marker rule {rule_id} uses unknown marker {marker}.")
        if rule["operator"] not in {"gt", "lt"}:
            raise ValueError(f"Pharmacy marker rule {rule_id} has an invalid operator.")
        threshold = _finite_float(rule["threshold"])
        priority = _finite_float(rule["base_priority"])
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Pharmacy marker rule {rule_id} has an invalid threshold.")
        if priority is None or not 0.0 <= priority <= 1.0:
            raise ValueError(f"Pharmacy marker rule {rule_id} has an invalid priority.")
        referenced_sources = rule["evidence_source_ids"]
        if not isinstance(referenced_sources, list) or not referenced_sources:
            raise ValueError(f"Pharmacy marker rule {rule_id} has no evidence sources.")
        unknown = sorted(set(referenced_sources).difference(source_id_set))
        if unknown:
            raise ValueError(
                f"Pharmacy rule {rule_id} references unknown sources: "
                f"{', '.join(unknown)}"
            )
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Pharmacy marker_rules contains duplicate rule_id values.")
    return source_id_set


@lru_cache(maxsize=1)
def load_pharmacy_knowledge_base() -> dict[str, Any]:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Pharmacy knowledge base is missing: {KNOWLEDGE_PATH}")
    raw_knowledge = KNOWLEDGE_PATH.read_bytes()
    payload = json.loads(raw_knowledge.decode("utf-8"))
    _validate_knowledge_payload(payload)
    payload["knowledge_sha256"] = hashlib.sha256(raw_knowledge).hexdigest()
    return payload


def _source_index(knowledge: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(source["source_id"]): dict(source)
        for source in knowledge["evidence_sources"]
    }


def _status_label(status: str) -> str:
    return {
        "standard": "信息较完整，可供医生或药师参考",
        "limited": "信息不完整，请先补充或核对",
        "withheld": "暂时无法给出菌群相关建议",
    }[status]


def _quality_context(
    submitted_microbes: Mapping[str, float],
    risk_result: Mapping[str, Any],
    model_features: Mapping[str, Any],
    metadata: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> dict[str, Any]:
    required_panel = [str(value) for value in knowledge["calibration"]["required_marker_panel"]]
    submitted_names = {str(name) for name in submitted_microbes}
    observed_markers = [name for name in required_panel if name in submitted_names]
    missing_markers = [name for name in required_panel if name not in submitted_names]
    panel_completeness = len(observed_markers) / max(len(required_panel), 1)

    panel_values = {
        marker: _finite_float(submitted_microbes.get(marker))
        for marker in required_panel
    }
    panel_total = sum(
        value for value in panel_values.values() if value is not None and value >= 0.0
    )
    valid_panel_values = all(
        value is not None and value >= 0.0 for value in panel_values.values()
    )
    calibration_available = not missing_markers and valid_panel_values and panel_total > 0.0
    calibration_ready = calibration_available
    calibrated_marker_values = (
        {
            marker: round(float(value) / panel_total, 6)
            for marker, value in panel_values.items()
            if value is not None
        }
        if calibration_ready
        else {}
    )

    reliability = str(risk_result.get("prediction_reliability", "unknown"))
    defaulted_inputs = _string_list(model_features.get("defaulted_inputs", []))
    out_of_range_inputs = _string_list(model_features.get("out_of_training_range_inputs", []))
    raw_out_of_range_details = model_features.get("out_of_training_range_details", [])
    out_of_range_details = (
        [dict(detail) for detail in raw_out_of_range_details if isinstance(detail, Mapping)]
        if isinstance(raw_out_of_range_details, list)
        else []
    )
    unsupported_microbes = _string_list(model_features.get("unsupported_microbes_ignored", []))

    reasons: list[dict[str, str]] = []
    hard_limit = False
    soft_limit = False

    if not missing_markers and not calibration_available:
        hard_limit = True
        reasons.append(
            {
                "code": "calibration_unavailable",
                "message": "五项菌群数值无法组成有效比例，请检查是否有空值、负数或总和为 0。",
            }
        )
    if reliability == "caution_out_of_training_range" or out_of_range_inputs:
        hard_limit = True
        reasons.append(
            {
                "code": "out_of_training_range",
                "message": _out_of_range_message(model_features, out_of_range_inputs),
            }
        )
    if reliability in {"caution_defaulted_inputs", "caution_split_disagreement"}:
        soft_limit = True
        reasons.append(
            {
                "code": reliability,
                "message": "模型在不同训练版本之间的结果不够一致，请补充资料并由人工复核。",
            }
        )
    elif reliability not in {"standard", "caution_out_of_training_range"}:
        hard_limit = True
        reasons.append(
            {
                "code": "unverified_model_reliability",
                "message": "当前模型无法确认结果是否可靠，因此暂不展示菌群相关建议。",
            }
        )
    if defaulted_inputs:
        soft_limit = True
        reasons.append(
            {
                "code": "defaulted_inputs",
                "message": "部分健康或代谢信息没有填写，系统使用了参考默认值。",
            }
        )
    if missing_markers:
        soft_limit = True
        reasons.append(
            {
                "code": "incomplete_marker_panel",
                "message": "五项菌群信息没有填全；系统不会把缺失项当作 0。",
            }
        )
    if unsupported_microbes:
        soft_limit = True
        reasons.append(
            {
                "code": "unsupported_microbes",
                "message": "有些菌名当前模型无法识别，结果没有使用这些数据。",
            }
        )
    if _binary_flag(metadata, "recent_antibiotics"):
        soft_limit = True
        reasons.append(
            {
                "code": "recent_antibiotics",
                "message": "近期使用过抗生素，本次菌群结果可能不是稳定状态。",
            }
        )
    missing_medication_context = [
        label
        for key, label in _MEDICATION_CONTEXT_LABELS.items()
        if key not in metadata
    ]
    if missing_medication_context:
        soft_limit = True
        reasons.append(
            {
                "code": "incomplete_medication_context",
                "message": f"还没有确认：{'、'.join(missing_medication_context)}。",
            }
        )

    status = "withheld" if hard_limit else "limited" if soft_limit else "standard"
    return {
        "status": status,
        "status_label": _status_label(status),
        "status_reasons": reasons,
        "required_marker_panel": required_panel,
        "observed_markers": observed_markers,
        "missing_markers": missing_markers,
        "panel_completeness": round(panel_completeness, 4),
        "calibration_ready": calibration_ready,
        "calibration_available": calibration_available,
        "calibration_scale": knowledge["calibration"]["value_scale"],
        "calibration_normalization": knowledge["calibration"]["normalization"],
        "panel_abundance_total": round(panel_total, 6),
        "calibrated_marker_values": calibrated_marker_values,
        "model_reliability": reliability,
        "defaulted_inputs": defaulted_inputs,
        "out_of_training_range_inputs": out_of_range_inputs,
        "out_of_training_range_details": out_of_range_details,
        "unsupported_microbes_ignored": unsupported_microbes,
    }


def _make_card(
    *,
    recommendation_id: str,
    category: str,
    title: str,
    suggestion: str,
    rationale: str,
    priority: float,
    urgency: str,
    evidence_level: str,
    evidence_source_ids: Sequence[str],
    action_steps: Sequence[str] | None = None,
    marker: str = "general",
    panel_composition: float | None = None,
    submitted_abundance: float | None = None,
    trigger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "recommendation_id": recommendation_id,
        "category": category,
        "title": title,
        "suggestion": suggestion,
        "rationale": rationale,
        "priority": round(max(0.0, min(float(priority), 1.0)), 4),
        "urgency": urgency,
        "urgency_label": "优先处理" if urgency == "priority" else "后续核对",
        "action_type": "clinician_review_only",
        "marker": marker,
        "evidence_level": evidence_level,
        "evidence_source_ids": list(evidence_source_ids),
        "requires_clinician_review": True,
        "allows_medication_change": False,
        "action_steps": [
            str(step).strip()
            for step in (action_steps or [suggestion])
            if str(step).strip()
        ],
    }
    if panel_composition is not None:
        card["marker_value"] = round(float(panel_composition), 6)
        card["panel_composition"] = round(float(panel_composition), 6)
    if submitted_abundance is not None:
        card["submitted_abundance"] = round(float(submitted_abundance), 6)
    if trigger is not None:
        card["trigger"] = dict(trigger)
    return card


def _risk_review_card(
    risk_result: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(quality.get("status", "withheld"))
    risk_level = str(risk_result.get("risk_level", "unknown")).lower()
    risk_level_text = {
        "high": "较高风险",
        "medium": "中等风险",
        "low": "较低风险",
    }.get(risk_level, "未明确风险")
    percentile = _finite_float(risk_result.get("risk_percentile", risk_result.get("risk_score")))
    percentile_text = f"（参考队列百分位 {percentile:.2f}%）" if percentile is not None else ""

    if status == "withheld":
        out_of_range_inputs = _string_list(quality.get("out_of_training_range_inputs", []))
        field_labels = [_model_input_label(field) for field in out_of_range_inputs]
        field_text = "、".join(field_labels)
        reasons = quality.get("status_reasons", [])
        reasons = reasons if isinstance(reasons, list) else []
        reason_message = next(
            (
                str(reason.get("message", "")).strip()
                for reason in reasons
                if isinstance(reason, Mapping) and reason.get("message")
            ),
            "当前输入不完整或超出模型可可靠解释的范围。",
        )
        title = (
            f"先核对{field_text}，再重新分析"
            if field_text
            else "先修正输入，再看菌群相关建议"
        )
        return _make_card(
            recommendation_id="model_result_withheld_review",
            category="data_quality",
            title=title,
            suggestion=reason_message,
            rationale="当前输入不完整或超出模型可可靠解释的范围，现在给出菌群相关建议容易误导。",
            priority=1.0,
            urgency="priority",
            evidence_level="safety_gate",
            evidence_source_ids=["FDA_CDS_2026"],
            action_steps=[
                reason_message,
                "核对后重新运行分析，确认页面不再显示“暂时无法给出菌群相关建议”。",
                "在此之前，不要依据本次结果更改药物、补充剂或益生菌。",
            ],
        )

    if risk_level == "high":
        title = "模型提示较高风险，优先安排临床复核"
        suggestion = "整理症状、既往检查、家族史和完整用药清单，带着本结果咨询消化专科或临床药师。"
        action_steps = [
            "记录目前症状、开始时间以及近期是否加重。",
            "准备既往检查结果、家族史和完整用药清单。",
            "把这些资料和本结果交给消化专科或临床药师，判断是否需要进一步检查；不要自行启停药。",
        ]
        priority = 0.96
        urgency = "priority"
    elif risk_level == "medium":
        title = "模型提示中等风险，安排一次有记录的复核"
        suggestion = "把症状变化、近期用药和既往检查整理在一起，供临床人员判断是否需要复查。"
        action_steps = [
            "记录症状变化、近期抗生素或益生菌使用情况。",
            "整理最近一次相关检查的日期和结果。",
            "在常规复诊时请临床人员判断是否需要复查或进一步评估。",
        ]
        priority = 0.76
        urgency = "routine"
    else:
        title = "当前模型风险较低，继续常规观察"
        suggestion = "保留本次结果并观察症状变化；低风险不等于排除疾病。"
        action_steps = [
            "保留本次结果，作为后续比较的基线。",
            "如症状持续、加重或近期用药发生变化，重新评估。",
            "不要因为当前结果较低而跳过原有的复诊或筛查计划。",
        ]
        priority = 0.56
        urgency = "routine"

    return _make_card(
        recommendation_id=f"risk_review_{risk_level}",
        category="risk_follow_up",
        title=title,
        suggestion=suggestion,
        rationale=f"模型结果为{risk_level_text}{percentile_text}。它表示在研究队列中的相对位置，不是诊断，也不是个人绝对发病概率。",
        priority=priority,
        urgency=urgency,
        evidence_level="model_assisted_review",
        evidence_source_ids=["FDA_CDS_2026", "INTERNAL_TOPOLOGY_V6"],
        action_steps=action_steps,
    )


def _medication_context_cards(metadata: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    medications_available = "current_medications" in metadata
    allergies_available = "drug_allergies" in metadata
    medications = _string_list(metadata.get("current_medications", []))
    allergies = _string_list(metadata.get("drug_allergies", []))
    cards: list[dict[str, Any]] = []
    missing_context = [
        label
        for key, label in _MEDICATION_CONTEXT_LABELS.items()
        if key not in metadata
    ]

    if missing_context:
        cards.append(
            _make_card(
                recommendation_id="medication_reconciliation_required",
                category="medication_safety",
                title="用药资料不完整，先补齐再核对",
                suggestion=f"当前还缺少：{'、'.join(missing_context)}。补齐后再查看用药安全提示。",
                rationale="缺少药名、过敏史或特殊人群信息时，系统可能漏掉需要人工处理的风险。",
                priority=0.92,
                urgency="priority",
                evidence_level="medication_safety_guidance",
                evidence_source_ids=["WHO_MEDICATION_RECONCILIATION", "FDA_CDS_2026"],
                action_steps=[
                    "照药盒或处方填写每种药的通用名、剂型、规格和每天用法。",
                    "补充药物过敏名称，以及当时出现了什么反应。",
                    "确认近期抗生素、益生菌、肝肾功能和妊娠状态。",
                ],
            )
        )

    special_flags = [
        label
        for key, label in (
            ("renal_impairment", "肾功能异常"),
            ("hepatic_impairment", "肝功能异常"),
            ("pregnancy", "妊娠"),
        )
        if _binary_flag(metadata, key)
    ]
    if special_flags:
        cards.append(
            _make_card(
                recommendation_id="special_population_medication_review",
                category="medication_safety",
                title="肝肾功能或妊娠信息需要人工核对",
                suggestion="准备最新检查和完整用药资料，在改动药物或补充剂前交给医生或药师核对。",
                rationale=f"已填写的情况包括：{'、'.join(special_flags)}。这些情况可能影响药物选择和剂量，但网页不能据此自动计算处方。",
                priority=0.98,
                urgency="priority",
                evidence_level="medication_safety_guidance",
                evidence_source_ids=["WHO_MEDICATION_RECONCILIATION", "FDA_CDS_2026"],
                action_steps=[
                    "准备最近一次肝肾功能检查的日期和结果；如涉及妊娠，也准备孕周或相关信息。",
                    "同时准备处方药、非处方药、保健品和益生菌清单。",
                    "在任何药物或补充剂调整前，请开药医生或临床药师完成个体化核对。",
                ],
            )
        )

    if _binary_flag(metadata, "recent_antibiotics"):
        cards.append(
            _make_card(
                recommendation_id="recent_antibiotic_exposure_review",
                category="antibiotic_stewardship",
                title="近期用过抗生素，先记录清楚再解释菌群",
                suggestion="补充抗生素名称、使用原因、开始和结束日期，再由临床人员判断本次检测是否需要复查。",
                rationale="近期抗生素可能暂时改变菌群，因此当前结果不一定代表稳定状态。",
                priority=0.94,
                urgency="priority",
                evidence_level="public_health_guidance",
                evidence_source_ids=["CDC_ANTIBIOTIC_STEWARDSHIP_2025", "FDA_CDS_2026"],
                action_steps=[
                    "写下抗生素名称、使用原因、开始日期、结束日期和最后一次用药时间。",
                    "把用药时间与采样日期一起交给临床人员，判断是否需要在稳定状态下复测。",
                    "不要因为菌群结果自行追加、延长或更换抗生素。",
                ],
            )
        )

    if _binary_flag(metadata, "recent_probiotics"):
        cards.append(
            _make_card(
                recommendation_id="recent_probiotic_exposure_review",
                category="microecology_review",
                title="近期用过益生菌，补充具体产品和菌株",
                suggestion="把包装上的产品名、完整菌株、每天用量、使用目的和日期填写清楚。",
                rationale="不同菌株和组合不能互相替代，仅凭某种菌偏高或偏低不能判断该用哪种产品。",
                priority=0.72,
                urgency="routine",
                evidence_level="clinical_guideline_caution",
                evidence_source_ids=["AGA_PROBIOTICS_2020"],
                action_steps=[
                    "记录产品名、完整菌株名称、每天用量和使用目的。",
                    "补充开始日期、结束日期或最后一次使用时间。",
                    "在确认具体临床用途前，不要只凭菌群结果继续或更换产品。",
                ],
            )
        )

    context = {
        "provided_fields": [
            key for key in _MEDICATION_CONTEXT_LABELS if key in metadata
        ],
        "missing_fields": [
            key for key in _MEDICATION_CONTEXT_LABELS if key not in metadata
        ],
        "context_completeness": round(
            sum(key in metadata for key in _MEDICATION_CONTEXT_LABELS)
            / len(_MEDICATION_CONTEXT_LABELS),
            4,
        ),
        "medication_list_available": medications_available,
        "allergy_history_available": allergies_available,
        "current_medications": medications,
        "drug_allergies": allergies,
        "renal_impairment": _binary_flag(metadata, "renal_impairment"),
        "hepatic_impairment": _binary_flag(metadata, "hepatic_impairment"),
        "pregnancy": _binary_flag(metadata, "pregnancy"),
        "recent_antibiotics": _binary_flag(metadata, "recent_antibiotics"),
        "recent_probiotics": _binary_flag(metadata, "recent_probiotics"),
        "interaction_screening_performed": False,
        "comprehensive_interaction_screening_performed": False,
        "label_lookup_performed": False,
        "interaction_screening_note": (
            "当前还没有完成药品知识核对。"
        ),
    }
    return cards, context


def _apply_drug_knowledge_quality(
    quality: dict[str, Any],
    drug_review: Mapping[str, Any],
) -> None:
    reason: dict[str, str] | None = None
    if not bool(drug_review.get("available")):
        reason = {
            "code": "drug_knowledge_database_unavailable",
            "message": "药品资料暂时无法读取，请改由医生或药师人工核对。",
        }
    else:
        normalization = drug_review.get("normalization", {})
        normalization = normalization if isinstance(normalization, Mapping) else {}
        medications = normalization.get("medications", [])
        if isinstance(medications, list) and any(
            isinstance(item, Mapping) and item.get("status") != "matched"
            for item in medications
        ):
            reason = {
                "code": "unresolved_medication_names",
                "message": "有药名无法识别，请照药盒或处方补全后重新核对。",
            }
    if reason is None:
        return
    if reason not in quality["status_reasons"]:
        quality["status_reasons"].append(reason)
    if quality["status"] != "withheld":
        quality["status"] = "limited"
        quality["status_label"] = _status_label("limited")


def _drug_knowledge_cards(drug_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not bool(drug_review.get("available")):
        return [
            _make_card(
                recommendation_id="drug_knowledge_database_unavailable",
                category="medication_safety",
                title="药品资料暂时无法读取，请改由人工核对",
                suggestion="把完整用药和过敏清单交给临床药师，使用医疗机构认可的药品数据库重新核对。",
                rationale="系统当前无法读取药名、说明书或相互作用资料，因此不会用猜测结果代替正式核对。",
                priority=1.0,
                urgency="priority",
                evidence_level="safety_gate",
                evidence_source_ids=["FDA_CDS_2026"],
                action_steps=[
                    "整理处方药、非处方药、保健品和益生菌的完整清单。",
                    "同时准备药物过敏和既往不良反应记录。",
                    "请临床药师使用医疗机构认可的当前药品数据库重新核对。",
                ],
            )
        ]

    cards: list[dict[str, Any]] = []
    interaction = drug_review.get("interaction_screening", {})
    interaction = interaction if isinstance(interaction, Mapping) else {}
    matches = interaction.get("matches", [])
    if isinstance(matches, list):
        for index, match in enumerate(matches):
            if not isinstance(match, Mapping):
                continue
            left = str(match.get("left_input", "药物 A"))
            right = str(match.get("right_input", "药物 B"))
            card = _make_card(
                recommendation_id=f"high_priority_ddi_{match.get('rule_id', index)}_{index + 1}",
                category="drug_interaction_alert",
                title=f"发现需要优先核对的用药组合：{left} + {right}",
                suggestion="不要自行停药或改剂量。请尽快把完整用药清单交给开药医生或临床药师核对。",
                rationale="这组用药命中了系统收录的严重相互作用规则。实际风险还要结合剂型、剂量、使用时间和个人情况判断。",
                priority=1.0,
                urgency="priority",
                evidence_level="high_priority_consensus_subset",
                evidence_source_ids=_string_list(
                    match.get("evidence_source_ids", ["ONC_HIGH_PRIORITY_DDI_2012"])
                ),
                action_steps=[
                    "尽快联系开药医生或临床药师核对这两种药；不要自行停药、换药或改剂量。",
                    f"把 {left} 和 {right} 的剂型、规格、每天用法和最后一次使用时间整理好。",
                    "把完整用药清单一并带上；如果已经明显不适，及时就医。",
                ],
            )
            card["interaction_match"] = dict(match)
            cards.append(card)

    allergy = drug_review.get("allergy_screening", {})
    allergy = allergy if isinstance(allergy, Mapping) else {}
    allergy_matches = allergy.get("matches", [])
    if isinstance(allergy_matches, list):
        for index, match in enumerate(allergy_matches):
            if not isinstance(match, Mapping):
                continue
            medication_input = str(match.get("medication_input", "当前用药"))
            allergy_input = str(match.get("allergy_input", "过敏记录"))
            card = _make_card(
                recommendation_id=f"exact_ingredient_allergy_match_{match.get('drug_id', index)}",
                category="medication_allergy_alert",
                title=f"用药清单与过敏记录出现同一种成分：{medication_input}",
                suggestion="尽快联系医生或药师，确认药名、既往过敏反应和当前处方。",
                rationale=(
                    f"当前用药“{medication_input}”与过敏记录“{allergy_input}”"
                    "被识别为同一种药物成分，需要人工确认是否填写准确。"
                ),
                priority=1.0,
                urgency="priority",
                evidence_level="exact_ingredient_safety_match",
                evidence_source_ids=["RXNORM_API_CURRENT", "FDA_CDS_2026"],
                action_steps=[
                    "核对药盒、处方和过敏记录，确认是否确为同一种成分。",
                    "写下上次过敏时出现的症状、发生时间和严重程度。",
                    "尽快联系医生或药师确认；如已出现过敏反应，及时就医。不要只凭网页自行决定下一次用药。",
                ],
            )
            card["allergy_match"] = dict(match)
            cards.append(card)

    label_lookup = drug_review.get("label_lookup", {})
    label_lookup = label_lookup if isinstance(label_lookup, Mapping) else {}
    unmatched = _string_list(label_lookup.get("unmatched_inputs", []))
    if unmatched:
        cards.append(
            _make_card(
                recommendation_id="unresolved_medication_names",
                category="medication_reconciliation",
                title=f"有 {len(unmatched)} 项药名无法识别，暂时不能完成核对",
                suggestion="照药盒或处方补充通用名、剂型、规格和每天用法，然后重新分析。",
                rationale=f"目前看不清的输入是：{'；'.join(unmatched)}。这些药不能被当作“没有相互作用”。",
                priority=0.96,
                urgency="priority",
                evidence_level="terminology_coverage_limit",
                evidence_source_ids=["RXNORM_API_CURRENT", "FDA_CDS_2026"],
                action_steps=[
                    "照药盒或处方填写通用名、剂型、规格和每天用法。",
                    "如只知道商品名，把商品名和生产厂家一并填写。",
                    "补全后重新分析；在此之前，请药师人工核对未识别的药。",
                ],
            )
        )

    probiotic = drug_review.get("probiotic_decision_support", {})
    probiotic = probiotic if isinstance(probiotic, Mapping) else {}
    condition_code = str(probiotic.get("condition_code", "")).strip()
    condition_label = _CONDITION_LABELS.get(condition_code, "表单中选择的临床情况")
    independent_basis = (
        f"这条提醒只根据你在表单中选择的“{condition_label}”和临床指南生成，"
        "不使用本次菌群模型分数。"
    )
    candidate_count = int(probiotic.get("candidate_count", 0) or 0)
    if candidate_count:
        card = _make_card(
            recommendation_id=f"probiotic_options_{probiotic.get('rule_id', 'guideline')}",
            category="probiotic_evidence",
            title="该临床情境有可供医生核对的特定益生菌方案",
            suggestion="先确认临床用途，再核对产品是否包含完全相同的菌株；网页不替患者选择剂量或疗程。",
            rationale=str(probiotic.get("evidence_scope", "指南中的益生菌方案只适用于特定疾病和特定菌株。")),
            priority=0.84,
            urgency="routine",
            evidence_level="conditional_guideline_option",
            evidence_source_ids=["AGA_PROBIOTICS_2020", "FDA_CDS_2026"],
            action_steps=[
                "先由临床人员确认是否真正符合该指南情境，而不是只看菌群结果。",
                "核对产品标签上的完整菌株名称；同属产品不能互相替代。",
                "由临床人员结合安全风险决定是否使用以及剂量和疗程，不要据此自行开始。",
            ],
        )
        card["probiotic_candidates"] = list(probiotic.get("candidates", []))
        card["independent_of_model_result"] = True
        card["decision_basis"] = independent_basis
        cards.append(card)
    elif probiotic.get("status") == "no_routine_candidate":
        stance = str(probiotic.get("stance", ""))
        if stance == "clinical_trial_only":
            title = f"{condition_label}：当前不提供常规益生菌方案"
        elif stance == "suggest_against":
            title = f"{condition_label}：当前指南不建议常规使用益生菌"
        else:
            title = f"{condition_label}：没有常规推荐的益生菌方案"
        card = _make_card(
            recommendation_id=f"probiotic_no_routine_candidate_{probiotic.get('rule_id', 'guideline')}",
            category="probiotic_evidence",
            title=title,
            suggestion=(
                "先确认表单中的诊断是否已经由临床人员明确；"
                "未确诊时请改回“尚未明确，仅做风险筛查”。"
            ),
            rationale=(
                f"{independent_basis} "
                f"{str(probiotic.get('evidence_scope', '现有指南没有支持该情境下的常规益生菌方案。'))}"
            ),
            priority=0.82,
            urgency="routine",
            evidence_level="clinical_guideline_caution",
            evidence_source_ids=["AGA_PROBIOTICS_2020", "FDA_CDS_2026"],
            action_steps=[
                f"确认“{condition_label}”是否已经由临床人员明确诊断；如果没有，请修改表单选项。",
                "如果正在使用益生菌，记录完整菌株、产品名、每天用量和已经使用的时间。",
                "复诊时让医生或药师判断是否继续；不要只根据菌群高低自行购买、更换或停用。",
            ],
        )
        card["independent_of_model_result"] = True
        card["decision_basis"] = independent_basis
        cards.append(card)
    return cards


def _screening_context_card(clinical: Mapping[str, float]) -> dict[str, Any] | None:
    age = _finite_float(clinical.get("age"))
    if age is None or not 45.0 <= age <= 85.0:
        return None
    family_history = _finite_float(clinical.get("family_history")) == 1.0
    priority = 0.88 if family_history else 0.74
    rationale = (
        f"年龄为 {age:g} 岁，且已记录家族史；筛查起始时间和方式需按个人风险由临床人员决定。"
        if family_history
        else f"年龄为 {age:g} 岁；应核对是否已按适用指南完成结直肠癌筛查。"
    )
    return _make_card(
        recommendation_id="guideline_screening_status_review",
        category="preventive_care",
        title="核对是否按计划完成结直肠癌筛查",
        suggestion="查清上一次筛查的日期、方式和结果，再请临床人员结合年龄、症状和家族史判断下一步。",
        rationale=rationale,
        priority=priority,
        urgency="priority" if family_history else "routine",
        evidence_level="preventive_guideline",
        evidence_source_ids=["USPSTF_CRC_2021", "FDA_CDS_2026"],
        action_steps=[
            "查找上一次结直肠癌筛查的日期、方式和结果。",
            "补充近期消化道症状和结直肠癌家族史。",
            "把这些信息交给临床人员确认下一次筛查安排；菌群结果本身不能决定是否做结肠镜。",
        ],
    )


def _marker_cards(
    submitted_microbes: Mapping[str, float],
    risk_result: Mapping[str, Any],
    quality: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if quality["status"] == "withheld" or not quality["calibration_ready"]:
        return []

    calibrated = quality["calibrated_marker_values"]
    risk_level = str(risk_result.get("risk_level", "unknown")).lower()
    risk_boost = {"high": 0.12, "medium": 0.05}.get(risk_level, 0.0)
    cards: list[dict[str, Any]] = []
    for rule in knowledge["marker_rules"]:
        marker = str(rule["marker"])
        if marker not in submitted_microbes:
            continue
        value = _finite_float(calibrated.get(marker))
        submitted_value = _finite_float(submitted_microbes.get(marker))
        threshold = float(rule["threshold"])
        operator = str(rule["operator"])
        triggered = value is not None and (
            (operator == "gt" and value > threshold)
            or (operator == "lt" and value < threshold)
        )
        if not triggered:
            continue
        cards.append(
            _make_card(
                recommendation_id=str(rule["rule_id"]),
                category=str(rule["category"]),
                title=str(rule["title"]),
                suggestion=str(rule["suggestion"]),
                rationale=str(rule["rationale"]),
                priority=float(rule["base_priority"]) + risk_boost,
                urgency=str(rule["urgency"]),
                evidence_level=str(rule["evidence_level"]),
                evidence_source_ids=rule["evidence_source_ids"],
                action_steps=_string_list(rule.get("action_steps", [])),
                marker=marker,
                panel_composition=value,
                submitted_abundance=submitted_value,
                trigger={
                    "operator": operator,
                    "threshold": threshold,
                    "threshold_quantile": rule["threshold_quantile"],
                    "value_scale": knowledge["calibration"]["value_scale"],
                    "normalization": knowledge["calibration"]["normalization"],
                    "dataset": knowledge["calibration"]["dataset"],
                    "source_table": knowledge["calibration"]["source_table"],
                },
            )
        )
    return cards


def _plain_language_summary(
    cards: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    medication_context: Mapping[str, Any],
    drug_review: Mapping[str, Any],
) -> dict[str, Any]:
    priority_cards = [card for card in cards if card.get("urgency") == "priority"]
    independent_cards = [
        card for card in cards if bool(card.get("independent_of_model_result"))
    ]
    routine_cards = [
        card
        for card in cards
        if card.get("urgency") != "priority"
        and not bool(card.get("independent_of_model_result"))
    ]
    interaction = drug_review.get("interaction_screening", {})
    interaction = interaction if isinstance(interaction, Mapping) else {}
    allergy = drug_review.get("allergy_screening", {})
    allergy = allergy if isinstance(allergy, Mapping) else {}
    normalization = drug_review.get("normalization", {})
    normalization = normalization if isinstance(normalization, Mapping) else {}
    label_lookup = drug_review.get("label_lookup", {})
    label_lookup = label_lookup if isinstance(label_lookup, Mapping) else {}

    interaction_count = int(interaction.get("match_count", 0) or 0)
    allergy_count = int(allergy.get("match_count", 0) or 0)
    if interaction_count or allergy_count:
        headline = "发现需要优先核对的用药安全提示"
    elif quality.get("status") == "withheld":
        out_of_range_inputs = _string_list(
            quality.get("out_of_training_range_inputs", [])
        )
        if out_of_range_inputs:
            field_text = "、".join(
                _model_input_label(field) for field in out_of_range_inputs
            )
            headline = f"{field_text}超出当前模型研究范围，请先核对"
        else:
            headline = "当前信息不足，先修正输入再看建议"
    elif priority_cards:
        headline = f"有 {len(priority_cards)} 项需要优先核对"
    else:
        headline = "未发现需要立即优先核对的已收录问题"

    what_to_do_now: list[str] = []
    selected_cards = priority_cards if priority_cards else routine_cards
    for card in selected_cards:
        steps = card.get("action_steps", [])
        if not isinstance(steps, list):
            steps = []
        candidate = str(steps[0] if steps else card.get("suggestion", "")).strip()
        if candidate and candidate not in what_to_do_now:
            what_to_do_now.append(candidate)
        if len(what_to_do_now) == 4:
            break

    input_count = int(normalization.get("input_count", 0) or 0)
    matched_count = int(normalization.get("matched_count", 0) or 0)
    label_count = int(label_lookup.get("record_count", 0) or 0)
    what_was_checked: list[str] = []
    if input_count:
        what_was_checked.append(f"已识别 {matched_count}/{input_count} 项当前用药名称")
    else:
        what_was_checked.append("尚未填写当前用药，未进行药名核对")
    if interaction.get("interaction_screening_performed"):
        what_was_checked.append("已核对系统收录的最高风险药物组合")
    else:
        what_was_checked.append("当前条件不足，尚未完成最高风险药物组合核对")
    if medication_context.get("medication_list_available") and medication_context.get(
        "allergy_history_available"
    ):
        what_was_checked.append("已比较可识别的当前用药与药物过敏记录")
    if label_count:
        what_was_checked.append(f"已有 {label_count} 份当前用药说明书可查看")

    what_was_not_checked = [
        "未收录在当前知识库中的其他药物相互作用",
        "患者个人应使用的具体药物、剂量、疗程或停换药方案",
        "需要结合病历、检验和体格检查判断的完整禁忌证与特殊风险",
    ]
    return {
        "headline": headline,
        "urgent_count": len(priority_cards),
        "routine_count": len(routine_cards),
        "independent_guidance_count": len(independent_cards),
        "what_to_do_now": what_to_do_now,
        "what_was_checked": what_was_checked,
        "what_was_not_checked": what_was_not_checked,
        "safety_note": "这是给医生或药师复核的辅助清单，不是诊断或处方。不要只凭网页自行启停药、换药或改剂量。",
    }


def build_pharmacy_assessment(
    *,
    submitted_microbes: Mapping[str, float],
    clinical: Mapping[str, float],
    risk_result: Mapping[str, Any],
    model_features: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    knowledge = load_pharmacy_knowledge_base()
    metadata = metadata if isinstance(metadata, Mapping) else {}
    quality = _quality_context(
        submitted_microbes,
        risk_result,
        model_features,
        metadata,
        knowledge,
    )

    medication_cards, medication_context = _medication_context_cards(metadata)
    drug_review = build_drug_knowledge_review(
        current_medications=medication_context["current_medications"],
        drug_allergies=medication_context["drug_allergies"],
        metadata=metadata,
    )
    interaction_screening = drug_review.get("interaction_screening", {})
    interaction_screening = (
        interaction_screening
        if isinstance(interaction_screening, Mapping)
        else {}
    )
    label_lookup = drug_review.get("label_lookup", {})
    label_lookup = label_lookup if isinstance(label_lookup, Mapping) else {}
    normalization = drug_review.get("normalization", {})
    normalization = normalization if isinstance(normalization, Mapping) else {}
    medication_context.update(
        {
            "interaction_screening_performed": bool(
                interaction_screening.get("interaction_screening_performed")
            ),
            "interaction_screening_scope": interaction_screening.get(
                "screening_scope"
            ),
            "comprehensive_interaction_screening_performed": bool(
                interaction_screening.get(
                    "comprehensive_interaction_screening_performed"
                )
            ),
            "interaction_screening_note": interaction_screening.get("note"),
            "label_lookup_performed": bool(label_lookup.get("performed")),
            "label_record_count": int(label_lookup.get("record_count", 0) or 0),
            "drug_name_normalization_coverage": normalization.get("coverage"),
            "drug_knowledge_available": bool(drug_review.get("available")),
        }
    )
    _apply_drug_knowledge_quality(quality, drug_review)

    cards = [_risk_review_card(risk_result, quality)]
    cards.extend(medication_cards)
    cards.extend(_drug_knowledge_cards(drug_review))
    screening_card = _screening_context_card(clinical)
    if screening_card is not None:
        cards.append(screening_card)
    cards.extend(
        _marker_cards(
            submitted_microbes,
            risk_result,
            quality,
            knowledge,
        )
    )
    cards.sort(key=lambda card: (-float(card["priority"]), str(card["recommendation_id"])))

    sources = _source_index(knowledge)
    used_source_ids = sorted(
        {
            source_id
            for card in cards
            for source_id in card.get("evidence_source_ids", [])
        }
    )
    evidence_sources = [sources[source_id] for source_id in used_source_ids]
    risk_percentile = _finite_float(
        risk_result.get("risk_percentile", risk_result.get("risk_score"))
    )
    marker_trigger_count = sum("trigger" in card for card in cards)
    priority_card_count = sum(card["urgency"] == "priority" for card in cards)
    probiotic_support = drug_review.get("probiotic_decision_support", {})
    probiotic_support = (
        probiotic_support if isinstance(probiotic_support, Mapping) else {}
    )

    return {
        "engine_version": knowledge["engine_version"],
        "knowledge_schema_version": knowledge["schema_version"],
        "knowledge_last_reviewed": knowledge["last_reviewed"],
        "knowledge_sha256": knowledge["knowledge_sha256"],
        "intended_use": knowledge["intended_use"],
        "status": quality["status"],
        "status_label": quality["status_label"],
        "quality": quality,
        "risk_context": {
            "risk_level": str(risk_result.get("risk_level", "unknown")),
            "risk_percentile": round(risk_percentile, 4) if risk_percentile is not None else None,
            "prediction_reliability": str(
                risk_result.get("prediction_reliability", "unknown")
            ),
            "split_disagreement": _finite_float(risk_result.get("split_disagreement")),
            "model_release": risk_result.get("model_release"),
        },
        "medication_context": medication_context,
        "drug_knowledge": drug_review,
        "plain_language_summary": _plain_language_summary(
            cards,
            quality,
            medication_context,
            drug_review,
        ),
        "summary": {
            "recommendation_count": len(cards),
            "marker_trigger_count": marker_trigger_count,
            "priority_card_count": priority_card_count,
            "medication_history_complete": (
                medication_context["medication_list_available"]
                and medication_context["allergy_history_available"]
            ),
            "medication_context_complete": not medication_context["missing_fields"],
            "interaction_screening_performed": medication_context[
                "interaction_screening_performed"
            ],
            "interaction_screening_scope": medication_context.get(
                "interaction_screening_scope"
            ),
            "comprehensive_interaction_screening_performed": medication_context[
                "comprehensive_interaction_screening_performed"
            ],
            "high_priority_interaction_match_count": int(
                interaction_screening.get("match_count", 0) or 0
            ),
            "label_lookup_performed": medication_context["label_lookup_performed"],
            "label_record_count": medication_context["label_record_count"],
            "medication_candidate_generated": False,
            "patient_specific_dose_selected": False,
            "treatment_duration_selected": False,
            "probiotic_candidate_count": int(
                probiotic_support.get("candidate_count", 0) or 0
            ),
            "medication_change_allowed": False,
        },
        "recommendations": cards,
        "evidence_sources": evidence_sources,
        "prohibited_actions": [
            "不得仅凭本结果诊断疾病。",
            "不得仅凭本结果开始、停止或更改药物、补充剂或益生菌。",
            "不得把说明书中的一般用法当作当前患者的具体剂量或疗程。",
            "未命中当前有限相互作用规则，不代表用药组合一定安全。",
            "模型推断的拓扑信息不是实验室实际测量结果。",
        ],
        "disclaimer": _DISCLAIMER,
    }
