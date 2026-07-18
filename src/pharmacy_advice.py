from __future__ import annotations

from typing import Dict, List


def build_pharmacy_assistance(
    report: Dict[str, object],
    metadata: Dict[str, object],
) -> List[Dict[str, object]]:
    """Return cards from the unified assessment for legacy report callers."""
    del metadata
    assessment = report.get("pharmacy_assessment", {})
    if not isinstance(assessment, dict):
        return []
    recommendations = assessment.get("recommendations", [])
    if not isinstance(recommendations, list):
        return []
    return [dict(item) for item in recommendations if isinstance(item, dict)]
