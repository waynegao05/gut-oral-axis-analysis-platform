from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


RULE_PATH = Path(__file__).resolve().parents[1] / "data" / "microbe_drug_rules.json"


def load_rules() -> List[Dict[str, object]]:
    if not RULE_PATH.exists():
        return []
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    return payload.get("rules", [])


def generate_recommendations(
    microbes: Dict[str, float],
    risk_score: float,
    risk_level: str,
) -> List[Dict[str, object]]:
    recommendations: List[Dict[str, object]] = []
    for rule in load_rules():
        marker = str(rule.get("marker", ""))
        direction = str(rule.get("direction", "increase"))
        abundance = float(microbes.get(marker, 0.0))
        triggered = abundance > 0.1 if direction == "increase" else abundance < 0.03
        if triggered:
            priority = float(rule.get("priority", 0.5)) + (0.2 if risk_level == "high" else 0.0)
            recommendations.append(
                {
                    "marker": marker,
                    "abundance": round(abundance, 4),
                    "priority": round(priority, 4),
                    "suggestion": rule.get("suggestion"),
                    "rationale": rule.get("rationale"),
                }
            )

    recommendations.sort(key=lambda x: x["priority"], reverse=True)
    if not recommendations:
        recommendations.append(
            {
                "marker": "general",
                "abundance": 0.0,
                "priority": 0.3,
                "suggestion": "Maintain routine follow-up and continue longitudinal microbiome monitoring.",
                "rationale": f"No strong prototype rule was triggered; current risk level is {risk_level}.",
            }
        )
    return recommendations
