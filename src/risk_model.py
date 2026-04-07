from __future__ import annotations

from typing import Dict

from config.settings import (
    DEFAULT_CLINICAL_WEIGHTS,
    DEFAULT_METABOLITE_WEIGHTS,
    DEFAULT_MICROBE_WEIGHTS,
    RISK_THRESHOLDS,
)


class CoxStyleRiskModel:
    def score(
        self,
        gnn_features: Dict[str, float],
        microbes: Dict[str, float],
        clinical: Dict[str, float],
        metabolites: Dict[str, float],
    ) -> Dict[str, float | str]:
        score = 0.0
        score += 0.7 * float(gnn_features.get("gnn_signal", 0.0))
        score += 0.5 * float(gnn_features.get("centrality_signal", 0.0))
        score += 0.4 * float(gnn_features.get("module_signal", 0.0))

        for name, weight in DEFAULT_MICROBE_WEIGHTS.items():
            score += weight * float(microbes.get(name, 0.0))
        for name, weight in DEFAULT_CLINICAL_WEIGHTS.items():
            score += weight * float(clinical.get(name, 0.0))
        for name, weight in DEFAULT_METABOLITE_WEIGHTS.items():
            score += weight * float(metabolites.get(name, 0.0))

        if score < RISK_THRESHOLDS["low"]:
            level = "low"
        elif score < RISK_THRESHOLDS["medium"]:
            level = "medium"
        else:
            level = "high"
        return {"risk_score": round(score, 4), "risk_level": level}
