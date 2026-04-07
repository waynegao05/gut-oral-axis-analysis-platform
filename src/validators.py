from __future__ import annotations

from typing import Any, Dict, List, Tuple


REQUIRED_TOP_LEVEL_KEYS = ["microbes", "clinical", "metabolites"]


def validate_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["Payload must be a JSON object."]

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            errors.append(f"Missing top-level field: {key}")
        elif not isinstance(payload[key], dict):
            errors.append(f"Field '{key}' must be an object.")

    microbes = payload.get("microbes", {})
    if isinstance(microbes, dict) and len(microbes) == 0:
        errors.append("Field 'microbes' must not be empty.")

    return len(errors) == 0, errors
