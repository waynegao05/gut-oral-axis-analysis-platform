from __future__ import annotations

from typing import Dict, List


def build_pharmacy_assistance(report: Dict[str, object], metadata: Dict[str, object]) -> List[Dict[str, str]]:
    risk_result = report.get("risk_result", {})
    risk_level = str(risk_result.get("risk_level", "unknown"))
    recommendations = report.get("recommendations", [])

    advice: List[Dict[str, str]] = []

    if risk_level == "high":
        advice.append(
            {
                "category": "follow_up_priority",
                "suggestion": "Recommend prompt gastroenterology evaluation and confirmatory examination planning.",
                "basis": "High-risk stratification from oral microbiome network features and fused host variables.",
            }
        )
        advice.append(
            {
                "category": "pharmacological_assistance",
                "suggestion": "Before empirical medication adjustment, review inflammatory status, prior antimicrobial exposure, and microbiome-disrupting drug history.",
                "basis": "High-risk subjects may show altered microbiome-linked drug response and ecological fragility.",
            }
        )
    elif risk_level == "medium":
        advice.append(
            {
                "category": "follow_up_priority",
                "suggestion": "Recommend structured follow-up and combined assessment with symptoms, inflammatory markers, and clinician judgment.",
                "basis": "Intermediate-risk stratification suggests non-negligible gut disorder possibility.",
            }
        )
    else:
        advice.append(
            {
                "category": "follow_up_priority",
                "suggestion": "Recommend routine monitoring and repeat screening if symptoms persist or exposure history changes.",
                "basis": "Current model output indicates relatively low risk under existing input conditions.",
            }
        )

    if metadata.get("recent_antibiotics", 0) == 1.0:
        advice.append(
            {
                "category": "medication_history",
                "suggestion": "Interpret current microbiome pattern cautiously because recent antibiotic exposure may distort baseline ecological structure.",
                "basis": "Recent antibiotic use is a major microbiome confounder and may affect downstream pharmacological interpretation.",
            }
        )

    if metadata.get("recent_probiotics", 0) == 1.0:
        advice.append(
            {
                "category": "microecology_interpretation",
                "suggestion": "Record recent probiotic use when reviewing risk and response, as it may partially shift abundance-based findings.",
                "basis": "Recent microbiome modulation can affect apparent community balance and intervention interpretation.",
            }
        )

    if recommendations:
        top = recommendations[0]
        advice.append(
            {
                "category": "marker_driven_assistance",
                "suggestion": f"Prioritize review of marker-associated disturbance around {top.get('marker', 'key taxa')} before final medication discussion.",
                "basis": str(top.get("rationale", "Top-ranked microbiome marker was triggered in the rule base.")),
            }
        )

    return advice
