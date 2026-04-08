from __future__ import annotations

from typing import Dict, List


def build_clinical_report(standardized: Dict[str, object], model_report: Dict[str, object], pharmacy_advice: List[Dict[str, str]]) -> Dict[str, object]:
    metadata = standardized.get("metadata", {})
    top_microbes = model_report.get("top_microbes", [])
    risk_result = model_report.get("risk_result", {})
    recommendations = model_report.get("recommendations", [])

    return {
        "patient_summary": {
            "sample_id": metadata.get("sample_id"),
            "sex": metadata.get("sex"),
            "chief_complaint": metadata.get("chief_complaint"),
            "suspected_condition": metadata.get("suspected_condition"),
        },
        "risk_assessment": risk_result,
        "microbiome_findings": {
            "top_microbes": top_microbes[:5],
            "rule_trigger_count": len(recommendations),
            "model_recommendations": recommendations,
        },
        "pharmacological_assistance": pharmacy_advice,
        "disclaimer": "This output is a research-oriented pharmacological assistance report. It does not replace physician diagnosis, confirmatory testing, or formal prescribing decisions.",
    }
