from __future__ import annotations

from typing import Dict, List

from src.pharmacy_engine import build_pharmacy_assessment, load_pharmacy_knowledge_base


def load_rules() -> List[Dict[str, object]]:
    """Return v2 marker rules for compatibility with earlier callers."""
    return [dict(rule) for rule in load_pharmacy_knowledge_base()["marker_rules"]]


def generate_recommendations(
    microbes: Dict[str, float],
    risk_score: float,
    risk_level: str,
) -> List[Dict[str, object]]:
    """Compatibility wrapper around the unified pharmacy-assistance engine.

    Without model calibration and reliability metadata, marker-driven rules are
    intentionally withheld. The returned cards remain safe for older callers.
    """
    assessment = build_pharmacy_assessment(
        submitted_microbes=microbes,
        clinical={},
        risk_result={
            "risk_score": risk_score,
            "risk_percentile": risk_score,
            "risk_level": risk_level,
            "prediction_reliability": "unknown",
        },
        model_features={},
        metadata={},
    )
    return list(assessment["recommendations"])
