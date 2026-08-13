from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("datasets.json")

FULL_MULTIMODAL_MODALITIES = frozenset(
    {
        "oral_microbiome",
        "gut_microbiome",
        "clinical_covariates",
        "metabolomics",
    }
)
SURVIVAL_CORE_MODALITIES = frozenset({"gut_microbiome", "clinical_covariates"})
DOWNLOAD_READY_ACCESS_MODES = frozenset({"open", "open_processed"})


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_registry(payload)
    return payload


def validate_registry(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Public dataset registry schema_version must be 1.")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Public dataset registry must contain a non-empty datasets list.")

    seen_ids: set[str] = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"Dataset entry {index} must be an object.")

        required = {
            "id",
            "title",
            "access",
            "modalities",
            "same_subject_modalities",
            "outcome",
            "recommended_role",
            "sources",
        }
        missing = sorted(required.difference(dataset))
        if missing:
            raise ValueError(f"Dataset entry {index} is missing keys: {missing}")

        dataset_id = dataset["id"]
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError(f"Dataset entry {index} has an invalid id.")
        if dataset_id in seen_ids:
            raise ValueError(f"Duplicate public dataset id: {dataset_id}")
        seen_ids.add(dataset_id)

        if not isinstance(dataset["modalities"], list):
            raise ValueError(f"{dataset_id}.modalities must be a list.")
        if not isinstance(dataset["same_subject_modalities"], bool):
            raise ValueError(f"{dataset_id}.same_subject_modalities must be boolean.")

        access = dataset["access"]
        outcome = dataset["outcome"]
        if not isinstance(access, dict) or "mode" not in access:
            raise ValueError(f"{dataset_id}.access must include mode.")
        outcome_keys = {"type", "individual_time", "individual_event", "right_censoring"}
        if not isinstance(outcome, dict) or not outcome_keys.issubset(outcome):
            raise ValueError(f"{dataset_id}.outcome is incomplete.")

        sources = dataset["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{dataset_id}.sources must be a non-empty list.")
        for source in sources:
            if not isinstance(source, dict) or not {"label", "url"}.issubset(source):
                raise ValueError(f"{dataset_id} contains an invalid source entry.")
            if not str(source["url"]).startswith("https://"):
                raise ValueError(f"{dataset_id} source URLs must use HTTPS.")


def _has_exact_survival(dataset: dict[str, Any]) -> bool:
    outcome = dataset["outcome"]
    return bool(
        outcome["type"] == "right_censored_time_to_event"
        and outcome["individual_time"]
        and outcome["individual_event"]
        and outcome["right_censoring"]
    )


def assess_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    modalities = set(dataset["modalities"])
    exact_survival = _has_exact_survival(dataset)
    same_subject = bool(dataset["same_subject_modalities"])
    access_mode = str(dataset["access"]["mode"])
    download_ready = access_mode in DOWNLOAD_READY_ACCESS_MODES

    missing_full = sorted(FULL_MULTIMODAL_MODALITIES.difference(modalities))
    missing_survival_core = sorted(SURVIVAL_CORE_MODALITIES.difference(modalities))
    if not exact_survival:
        missing_full.append("right_censored_time_to_event")
        missing_survival_core.append("right_censored_time_to_event")

    full_contract_satisfied = not missing_full and same_subject
    survival_core_satisfied = not missing_survival_core and same_subject

    return {
        "id": dataset["id"],
        "title": dataset["title"],
        "access_mode": access_mode,
        "download_ready": download_ready,
        "same_subject_modalities": same_subject,
        "exact_right_censored_survival": exact_survival,
        "full_contract_satisfied": full_contract_satisfied,
        "full_replacement_ready": full_contract_satisfied and download_ready,
        "survival_core_satisfied": survival_core_satisfied,
        "survival_core_ready": survival_core_satisfied and download_ready,
        "missing_full_contract": missing_full,
        "missing_survival_core": missing_survival_core,
        "recommended_role": dataset["recommended_role"],
    }


def build_audit_report(registry: dict[str, Any]) -> dict[str, Any]:
    validate_registry(registry)
    assessments = [assess_dataset(dataset) for dataset in registry["datasets"]]
    full_ready = [item["id"] for item in assessments if item["full_replacement_ready"]]
    survival_ready = [item["id"] for item in assessments if item["survival_core_ready"]]
    survival_after_access = [
        item["id"]
        for item in assessments
        if item["survival_core_satisfied"] and not item["download_ready"]
    ]
    return {
        "schema_version": 1,
        "current_dataset_policy": registry["current_dataset_policy"],
        "cross_cohort_patient_join_allowed": False,
        "strict_full_replacement_available": bool(full_ready),
        "strict_full_replacement_dataset_ids": full_ready,
        "download_ready_survival_core_dataset_ids": survival_ready,
        "survival_core_dataset_ids_after_access": survival_after_access,
        "assessments": assessments,
    }
