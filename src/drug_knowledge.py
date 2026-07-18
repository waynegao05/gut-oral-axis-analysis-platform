from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
LABEL_DATABASE_PATH = (
    ROOT / "data" / "pharmacy_knowledge" / "openfda_label_evidence_v1.json"
)
DDI_DATABASE_PATH = (
    ROOT / "data" / "pharmacy_knowledge" / "high_priority_ddi_v1.json"
)
PROBIOTIC_DATABASE_PATH = (
    ROOT / "data" / "pharmacy_knowledge" / "probiotic_guidance_v1.json"
)

_SECTION_LABELS_ZH = {
    "boxed_warning": "重要警告",
    "indications_and_usage": "适用于什么情况",
    "purpose": "适用于什么情况",
    "uses": "适用于什么情况",
    "dosage_and_administration": "说明书中的一般用法（不可直接照此改剂量）",
    "directions": "说明书中的一般用法（不可直接照此改剂量）",
    "dosage_forms_and_strengths": "剂型与规格",
    "contraindications": "哪些情况不能用",
    "warnings": "使用时要注意什么",
    "warnings_and_cautions": "使用时要注意什么",
    "do_not_use": "哪些情况不能用",
    "ask_doctor": "使用前需要向医生确认什么",
    "ask_doctor_or_pharmacist": "使用前需要向医生或药师确认什么",
    "stop_use": "说明书要求停用并咨询的情况",
    "drug_interactions": "和其他药一起使用要注意什么",
    "use_in_specific_populations": "特殊人群注意事项",
    "pregnancy": "妊娠期注意事项",
    "pregnancy_or_breast_feeding": "妊娠或哺乳期注意事项",
    "pediatric_use": "儿童使用注意事项",
    "geriatric_use": "老年人使用注意事项",
    "overdosage": "用药过量时的说明"
}

_BASE_LABEL_SECTIONS = (
    "boxed_warning",
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "do_not_use",
    "drug_interactions",
    "indications_and_usage",
    "uses",
    "dosage_and_administration",
    "directions",
)
_SPECIAL_POPULATION_SECTIONS = (
    "use_in_specific_populations",
    "pregnancy",
    "pregnancy_or_breast_feeding",
    "pediatric_use",
    "geriatric_use",
)
_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Drug knowledge root must be a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validate_label_database(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "dataset_id",
        "generated_at",
        "sources",
        "coverage",
        "records_sha256",
        "records",
        "failures",
        "safety_contract",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Label database is missing: {', '.join(missing)}")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("Label database records must be a list.")
    if _sha256(records) != str(payload["records_sha256"]):
        raise ValueError("Label database records_sha256 does not match its records.")

    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Label record {index} must be an object.")
        drug_id = str(record.get("drug_id", "")).strip()
        if not drug_id or drug_id in seen:
            raise ValueError(f"Label record has invalid or duplicate drug_id: {drug_id}")
        seen.add(drug_id)
        for field in ("display_name", "aliases", "rxnorm", "openfda", "label_sections"):
            if field not in record:
                raise ValueError(f"Label record {drug_id} is missing {field}.")
        expected_hash = str(record.get("record_sha256", ""))
        hash_payload = dict(record)
        hash_payload.pop("record_sha256", None)
        if not expected_hash or _sha256(hash_payload) != expected_hash:
            raise ValueError(f"Label record {drug_id} failed its hash check.")


def _validate_ddi_database(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "dataset_id",
        "source",
        "screening_contract",
        "drug_groups",
        "rules",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"DDI database is missing: {', '.join(missing)}")
    groups = payload["drug_groups"]
    rules = payload["rules"]
    if not isinstance(groups, list) or not isinstance(rules, list):
        raise ValueError("DDI drug_groups and rules must be lists.")
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Each DDI group must be an object.")
        group_id = str(group.get("group_id", "")).strip()
        members = _string_list(group.get("members", []))
        if not group_id or group_id in group_ids or not members:
            raise ValueError(f"Invalid DDI group: {group_id}")
        group_ids.add(group_id)
    rule_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each DDI rule must be an object.")
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id or rule_id in rule_ids:
            raise ValueError(f"Invalid or duplicate DDI rule: {rule_id}")
        rule_ids.add(rule_id)
        if bool(rule.get("screening_supported")):
            for side in ("left_group_id", "right_group_id"):
                if str(rule.get(side, "")) not in group_ids:
                    raise ValueError(f"DDI rule {rule_id} references unknown {side}.")


def _validate_probiotic_database(payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "dataset_id", "source", "safety_contract", "rules"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Probiotic database is missing: {', '.join(missing)}")
    rules = payload["rules"]
    if not isinstance(rules, list):
        raise ValueError("Probiotic rules must be a list.")
    rule_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each probiotic rule must be an object.")
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id or rule_id in rule_ids:
            raise ValueError(f"Invalid or duplicate probiotic rule: {rule_id}")
        if not _string_list(rule.get("condition_codes", [])):
            raise ValueError(f"Probiotic rule {rule_id} has no condition codes.")
        rule_ids.add(rule_id)


@lru_cache(maxsize=1)
def load_label_database() -> dict[str, Any]:
    payload, file_sha256 = _read_json(LABEL_DATABASE_PATH)
    _validate_label_database(payload)
    payload["file_sha256"] = file_sha256
    return payload


@lru_cache(maxsize=1)
def load_ddi_database() -> dict[str, Any]:
    payload, file_sha256 = _read_json(DDI_DATABASE_PATH)
    _validate_ddi_database(payload)
    payload["file_sha256"] = file_sha256
    return payload


@lru_cache(maxsize=1)
def load_probiotic_database() -> dict[str, Any]:
    payload, file_sha256 = _read_json(PROBIOTIC_DATABASE_PATH)
    _validate_probiotic_database(payload)
    payload["file_sha256"] = file_sha256
    return payload


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("_", " ").replace("’", "'")
    text = _NON_WORD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _contains_alias(normalized_input: str, normalized_alias: str) -> bool:
    if not normalized_input or not normalized_alias:
        return False
    return f" {normalized_alias} " in f" {normalized_input} "


def _record_index(label_database: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["drug_id"]): dict(record)
        for record in label_database["records"]
        if isinstance(record, dict)
    }


def _alias_index(label_database: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    aliases: list[tuple[str, str, str]] = []
    for record in label_database["records"]:
        drug_id = str(record["drug_id"])
        for raw_alias in [drug_id, *_string_list(record.get("aliases", []))]:
            normalized = _normalize_text(raw_alias)
            if len(normalized) >= 2:
                aliases.append((normalized, drug_id, str(raw_alias)))
    aliases.sort(key=lambda item: (-len(item[0]), item[1], item[0]))
    return aliases


def normalize_medication_inputs(
    medication_inputs: Sequence[str],
    label_database: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    database = label_database or load_label_database()
    records = _record_index(database)
    aliases = _alias_index(database)
    normalized_results: list[dict[str, Any]] = []
    for raw_input in medication_inputs:
        raw_text = str(raw_input).strip()
        normalized_input = _normalize_text(raw_text)
        matches = [
            (alias, drug_id, source_alias)
            for alias, drug_id, source_alias in aliases
            if _contains_alias(normalized_input, alias)
        ]
        if not matches:
            normalized_results.append(
                {
                    "input": raw_text,
                    "normalized_input": normalized_input,
                    "status": "unmatched",
                    "drug_id": None,
                    "display_name": None,
                    "rxcui": None,
                    "matched_alias": None,
                }
            )
            continue

        best_length = len(matches[0][0])
        best = [item for item in matches if len(item[0]) == best_length]
        drug_ids = sorted({item[1] for item in best})
        if len(drug_ids) > 1:
            normalized_results.append(
                {
                    "input": raw_text,
                    "normalized_input": normalized_input,
                    "status": "ambiguous",
                    "candidate_drug_ids": drug_ids,
                    "drug_id": None,
                    "display_name": None,
                    "rxcui": None,
                    "matched_alias": None,
                }
            )
            continue

        alias, drug_id, source_alias = best[0]
        record = records[drug_id]
        rxnorm = record.get("rxnorm", {})
        rxnorm = rxnorm if isinstance(rxnorm, dict) else {}
        normalized_results.append(
            {
                "input": raw_text,
                "normalized_input": normalized_input,
                "status": "matched",
                "drug_id": drug_id,
                "display_name": str(record["display_name"]),
                "rxcui": str(rxnorm.get("rxcui", "")) or None,
                "matched_alias": source_alias,
                "matched_alias_normalized": alias,
            }
        )
    return normalized_results


def _excerpt(values: Any, max_characters: int = 1200) -> dict[str, Any] | None:
    text = "\n".join(_string_list(values)).strip()
    if not text:
        return None
    truncated = len(text) > max_characters
    excerpt = text[:max_characters]
    if truncated:
        boundary = excerpt.rfind(" ")
        if boundary >= max_characters // 2:
            excerpt = excerpt[:boundary]
        excerpt = excerpt.rstrip() + " ..."
    return {
        "excerpt": excerpt,
        "truncated": truncated,
        "full_character_count": len(text),
    }


def _label_evidence_for_record(
    normalized_medication: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    special_population_review: bool,
) -> dict[str, Any]:
    openfda = record.get("openfda", {})
    openfda = openfda if isinstance(openfda, dict) else {}
    label_sections = record.get("label_sections", {})
    label_sections = label_sections if isinstance(label_sections, dict) else {}
    requested_sections = list(_BASE_LABEL_SECTIONS)
    if special_population_review:
        requested_sections.extend(_SPECIAL_POPULATION_SECTIONS)
    sections: dict[str, Any] = {}
    for section_name in requested_sections:
        section = _excerpt(label_sections.get(section_name, []))
        if section is not None:
            section["label_zh"] = _SECTION_LABELS_ZH.get(section_name, section_name)
            sections[section_name] = section

    dosing_section_name = next(
        (
            name
            for name in ("dosage_and_administration", "directions")
            if name in sections
        ),
        None,
    )
    return {
        "input": normalized_medication["input"],
        "drug_id": record["drug_id"],
        "display_name": record["display_name"],
        "review_prompt": "先核对实际药名、剂型、规格和每天用法是否与这份说明书一致。",
        "rxcui": normalized_medication.get("rxcui"),
        "label_identity": {
            "generic_names": _string_list(openfda.get("generic_names", [])),
            "brand_names": _string_list(openfda.get("brand_names", [])),
            "manufacturer_names": _string_list(openfda.get("manufacturer_names", [])),
            "product_types": _string_list(openfda.get("product_types", [])),
            "routes": _string_list(openfda.get("routes", [])),
            "spl_set_id": openfda.get("set_id"),
            "spl_version": openfda.get("version"),
            "effective_time": openfda.get("effective_time"),
        },
        "source": {
            "source_id": "OPENFDA_LABEL_CURRENT",
            "openfda_query_url": openfda.get("query_url"),
            "dailymed_url": openfda.get("dailymed_url"),
            "record_sha256": record.get("record_sha256"),
        },
        "sections": sections,
        "dose_and_course_reference": {
            "available": dosing_section_name is not None,
            "source_section": dosing_section_name,
            "patient_specific_dose_selected": False,
            "treatment_duration_selected": False,
            "interpretation": (
                "这里只展示产品说明书中的一般信息，并未替当前患者选择剂量、疗程或用药方案。"
            ),
        },
        "product_specific_label": True,
        "allows_medication_change": False,
    }


def _ddi_group_index(ddi_database: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        str(group["group_id"]): {
            _normalize_text(member) for member in _string_list(group.get("members", []))
        }
        for group in ddi_database["drug_groups"]
        if isinstance(group, dict)
    }


def _ddi_member_ids(
    medication: Mapping[str, Any],
    all_members: set[str],
) -> set[str]:
    member_ids: set[str] = set()
    drug_id = medication.get("drug_id")
    if drug_id:
        normalized_id = _normalize_text(drug_id)
        if normalized_id in all_members:
            member_ids.add(normalized_id)
    normalized_input = str(medication.get("normalized_input", ""))
    for member in all_members:
        if _contains_alias(normalized_input, member):
            member_ids.add(member)
    return member_ids


def screen_high_priority_interactions(
    normalized_medications: Sequence[Mapping[str, Any]],
    ddi_database: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    database = ddi_database or load_ddi_database()
    groups = _ddi_group_index(database)
    all_members = set().union(*groups.values()) if groups else set()
    medication_members = [
        _ddi_member_ids(medication, all_members)
        for medication in normalized_medications
    ]
    matches: list[dict[str, Any]] = []
    seen_matches: set[tuple[str, int, int]] = set()
    for rule in database["rules"]:
        if not bool(rule.get("screening_supported")):
            continue
        left_members = groups[str(rule["left_group_id"])]
        right_members = groups[str(rule["right_group_id"])]
        for left_index, left_resolved in enumerate(medication_members):
            if not left_resolved.intersection(left_members):
                continue
            for right_index, right_resolved in enumerate(medication_members):
                if left_index == right_index or not right_resolved.intersection(right_members):
                    continue
                pair = tuple(sorted((left_index, right_index)))
                key = (str(rule["rule_id"]), pair[0], pair[1])
                if key in seen_matches:
                    continue
                seen_matches.add(key)
                matches.append(
                    {
                        "rule_id": rule["rule_id"],
                        "source_number": rule["source_number"],
                        "alert_title": rule["alert_title_zh"],
                        "severity": "critical_consensus_subset",
                        "left_input": normalized_medications[left_index]["input"],
                        "right_input": normalized_medications[right_index]["input"],
                        "left_members": sorted(left_resolved.intersection(left_members)),
                        "right_members": sorted(right_resolved.intersection(right_members)),
                        "required_action": "urgent_clinician_or_pharmacist_review",
                        "allows_automatic_medication_change": False,
                        "evidence_source_ids": [
                            str(database["source"]["source_id"]),
                            "OPENFDA_LABEL_CURRENT",
                        ],
                    }
                )

    supported_rules = [
        rule for rule in database["rules"] if bool(rule.get("screening_supported"))
    ]
    unsupported_rules = [
        str(rule["rule_id"])
        for rule in database["rules"]
        if not bool(rule.get("screening_supported"))
    ]
    input_count = len(normalized_medications)
    matched_input_count = sum(
        medication.get("status") == "matched" for medication in normalized_medications
    )
    resolved_for_subset_count = sum(
        medication.get("status") == "matched" or bool(member_ids)
        for medication, member_ids in zip(normalized_medications, medication_members)
    )
    performed = input_count >= 2 and resolved_for_subset_count == input_count
    if input_count < 2:
        screening_status = "insufficient_medication_count"
    elif performed:
        screening_status = "completed_for_minimum_subset"
    else:
        screening_status = "incomplete_unresolved_medication_names"
    return {
        "interaction_screening_performed": performed,
        "screening_status": screening_status,
        "screening_scope": "onc_2012_minimum_high_priority_subset",
        "comprehensive_interaction_screening_performed": False,
        "negative_result_excludes_other_interactions": False,
        "input_medication_count": input_count,
        "resolved_for_subset_count": resolved_for_subset_count,
        "screened_pair_count": input_count * (input_count - 1) // 2 if performed else 0,
        "normalization_coverage": (
            round(matched_input_count / input_count, 4) if input_count else None
        ),
        "source_rule_count": len(database["rules"]),
        "implemented_rule_count": len(supported_rules),
        "unsupported_rule_ids": unsupported_rules,
        "match_count": len(matches),
        "matches": matches,
        "source": dict(database["source"]),
        "note": (
            "这里只核对了系统收录的最高风险组合。没有命中不代表不存在其他相互作用。"
        ),
    }


def _screen_exact_ingredient_allergies(
    normalized_medications: Sequence[Mapping[str, Any]],
    normalized_allergies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current_by_id = {
        str(item["drug_id"]): item
        for item in normalized_medications
        if item.get("status") == "matched" and item.get("drug_id")
    }
    allergy_by_id = {
        str(item["drug_id"]): item
        for item in normalized_allergies
        if item.get("status") == "matched" and item.get("drug_id")
    }
    matches = [
        {
            "drug_id": drug_id,
            "medication_input": current_by_id[drug_id]["input"],
            "allergy_input": allergy_by_id[drug_id]["input"],
            "severity": "urgent_review",
            "allows_automatic_medication_change": False,
        }
        for drug_id in sorted(set(current_by_id).intersection(allergy_by_id))
    ]
    return {
        "exact_ingredient_screening_performed": bool(normalized_allergies),
        "class_cross_reactivity_screening_performed": False,
        "match_count": len(matches),
        "matches": matches,
        "unmatched_allergy_inputs": [
            item["input"]
            for item in normalized_allergies
            if item.get("status") != "matched"
        ],
        "negative_result_excludes_allergy_or_cross_reactivity": False,
    }


def build_probiotic_decision_support(
    metadata: Mapping[str, Any],
    probiotic_database: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    database = probiotic_database or load_probiotic_database()
    condition = _normalize_text(metadata.get("suspected_condition", ""))
    condition_code = condition.replace(" ", "_")
    recent_antibiotics = bool(metadata.get("recent_antibiotics") == 1 or metadata.get("recent_antibiotics") is True)
    matched_rule: Mapping[str, Any] | None = None
    for rule in database["rules"]:
        codes = {
            _normalize_text(code).replace(" ", "_")
            for code in _string_list(rule.get("condition_codes", []))
        }
        if condition_code in codes:
            matched_rule = rule
            break

    base = {
        "condition_code": condition_code or None,
        "microbiome_marker_alone_is_an_indication": False,
        "patient_specific_dose_selected": False,
        "treatment_duration_selected": False,
        "allows_automatic_start_or_stop": False,
        "evidence_source_ids": [str(database["source"]["source_id"])],
        "source": dict(database["source"]),
    }
    if matched_rule is None:
        return {
            **base,
            "status": "not_generated_no_supported_indication",
            "stance": "no_rule_for_submitted_context",
            "candidate_count": 0,
            "candidates": [],
            "note": "风险模型或菌群高低本身不能证明需要使用益生菌。",
        }
    if bool(matched_rule.get("requires_recent_antibiotics")) and not recent_antibiotics:
        return {
            **base,
            "status": "withheld_required_context_missing",
            "stance": matched_rule["stance"],
            "rule_id": matched_rule["rule_id"],
            "candidate_count": 0,
            "candidates": [],
            "note": "该指南情境需要先确认患者正在同时接受抗生素治疗。",
        }

    candidates = []
    for candidate in matched_rule.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "strains": _string_list(candidate.get("strains", [])),
                "candidate_type": "guideline_option_for_clinician_review",
                "dose_selected": False,
                "duration_selected": False,
                "product_interchangeable": False,
                "requires_clinician_review": True,
                "allows_automatic_start_or_stop": False,
            }
        )
    return {
        **base,
        "status": "guideline_options_available" if candidates else "no_routine_candidate",
        "stance": matched_rule["stance"],
        "rule_id": matched_rule["rule_id"],
        "evidence_scope": matched_rule["evidence_scope_zh"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "required_safety_review": [
            "先独立确认临床诊断和使用目的，不能只看菌群模型。",
            "核对免疫状态、严重疾病、侵入性器械、年龄、妊娠情况和产品质量。",
            "核对完整菌株或菌株组合；不同产品不能默认互换。",
            "由临床人员根据当前适用指南和具体产品证据选择剂量与疗程。",
        ],
    }


def _database_freshness(label_database: Mapping[str, Any]) -> dict[str, Any]:
    generated_text = str(label_database.get("generated_at", ""))
    age_days: int | None = None
    try:
        generated = datetime.fromisoformat(generated_text.replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        age_days = max(0, (datetime.now(timezone.utc) - generated).days)
    except ValueError:
        pass
    return {
        "generated_at": generated_text or None,
        "age_days": age_days,
        "refresh_due": age_days is None or age_days > 35,
        "recommended_refresh_interval_days": 35,
    }


def build_drug_knowledge_review(
    *,
    current_medications: Sequence[str],
    drug_allergies: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        label_database = load_label_database()
        label_error = None
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        label_database = None
        label_error = str(exc)

    if label_database is None:
        return {
            "available": False,
            "status": "knowledge_database_unavailable",
            "error": label_error,
            "label_lookup": {"performed": False, "records": []},
            "interaction_screening": {
                "interaction_screening_performed": False,
                "comprehensive_interaction_screening_performed": False,
            },
            "allergy_screening": {
                "exact_ingredient_screening_performed": False,
                "class_cross_reactivity_screening_performed": False,
            },
            "candidate_therapy_support": {
                "medication_candidates_generated": False,
                "patient_specific_dose_selected": False,
                "treatment_duration_selected": False,
            },
        }

    normalized_medications = normalize_medication_inputs(
        list(current_medications), label_database
    )
    normalized_allergies = normalize_medication_inputs(
        list(drug_allergies), label_database
    )
    records = _record_index(label_database)
    special_population_review = any(
        metadata.get(field) == 1 or metadata.get(field) is True
        for field in ("renal_impairment", "hepatic_impairment", "pregnancy")
    )
    label_evidence = [
        _label_evidence_for_record(
            item,
            records[str(item["drug_id"])],
            special_population_review=special_population_review,
        )
        for item in normalized_medications
        if item.get("status") == "matched" and item.get("drug_id") in records
    ]

    try:
        ddi_database = load_ddi_database()
        interaction_screening = screen_high_priority_interactions(
            normalized_medications, ddi_database
        )
        ddi_sha256 = ddi_database["file_sha256"]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        interaction_screening = {
            "interaction_screening_performed": False,
            "comprehensive_interaction_screening_performed": False,
            "status": "ddi_database_unavailable",
            "error": str(exc),
            "matches": [],
            "match_count": 0,
        }
        ddi_sha256 = None

    try:
        probiotic_database = load_probiotic_database()
        probiotic_support = build_probiotic_decision_support(
            metadata, probiotic_database
        )
        probiotic_sha256 = probiotic_database["file_sha256"]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        probiotic_support = {
            "status": "probiotic_database_unavailable",
            "error": str(exc),
            "candidate_count": 0,
            "candidates": [],
            "patient_specific_dose_selected": False,
            "treatment_duration_selected": False,
            "allows_automatic_start_or_stop": False,
        }
        probiotic_sha256 = None

    matched_count = sum(item["status"] == "matched" for item in normalized_medications)
    return {
        "available": True,
        "status": "limited_clinical_decision_support",
        "database": {
            "dataset_id": label_database["dataset_id"],
            "file_sha256": label_database["file_sha256"],
            "records_sha256": label_database["records_sha256"],
            "record_count": label_database["coverage"]["label_record_count"],
            "comprehensive_drug_coverage": False,
            "freshness": _database_freshness(label_database),
            "nlm_attribution": label_database.get("nlm_attribution"),
            "openfda_scope_note": (
                "这里只收录部分具体产品的说明书记录，不能代表所有厂家、剂型、给药途径或当前在售产品。"
            ),
            "ddi_file_sha256": ddi_sha256,
            "probiotic_file_sha256": probiotic_sha256,
        },
        "normalization": {
            "input_count": len(current_medications),
            "matched_count": matched_count,
            "coverage": (
                round(matched_count / len(current_medications), 4)
                if current_medications
                else None
            ),
            "medications": normalized_medications,
            "allergies": normalized_allergies,
        },
        "label_lookup": {
            "performed": bool(current_medications),
            "source_scope": "selected_product_specific_openfda_spl",
            "record_count": len(label_evidence),
            "records": label_evidence,
            "unmatched_inputs": [
                item["input"]
                for item in normalized_medications
                if item["status"] != "matched"
            ],
            "negative_result_excludes_warning_or_contraindication": False,
        },
        "interaction_screening": interaction_screening,
        "allergy_screening": _screen_exact_ingredient_allergies(
            normalized_medications, normalized_allergies
        ),
        "probiotic_decision_support": probiotic_support,
        "candidate_therapy_support": {
            "medication_candidates_generated": False,
            "status": "withheld_without_confirmed_indication_and_required_parameters",
            "official_label_evidence_returned_for_current_medications": bool(label_evidence),
            "patient_specific_dose_selected": False,
            "treatment_duration_selected": False,
            "reason": (
                "现有风险结果和用药背景不足以确认诊断，也不足以安全选择具体药物、剂量或疗程。"
            ),
            "required_before_specific_candidate_review": [
                "已确认的诊断和治疗目的",
                "治疗目标和既往治疗反应",
                "完整的药物、过敏和补充剂清单",
                "年龄、体重以及肝肾功能检验结果",
                "适用时的妊娠或哺乳状态",
                "相关合并疾病、生命体征和检验监测",
                "当前适用指南、医院药品目录和具体产品说明书复核",
            ],
        },
        "safety_boundary": {
            "research_and_clinician_review_only": True,
            "not_a_diagnosis": True,
            "not_a_prescription": True,
            "allows_start_stop_or_dose_change": False,
            "no_match_means_safe": False,
        },
        "evidence_source_ids": [
            "RXNORM_API_CURRENT",
            "OPENFDA_LABEL_CURRENT",
            "DAILYMED_SPL_CURRENT",
            "ONC_HIGH_PRIORITY_DDI_2012",
            "AGA_PROBIOTICS_2020",
        ],
    }
