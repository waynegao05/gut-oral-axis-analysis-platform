from __future__ import annotations

from typing import Dict, List


def build_report(
    microbes: Dict[str, float],
    gnn_features: Dict[str, object],
    risk_result: Dict[str, object],
    recommendations: List[Dict[str, object]],
) -> Dict[str, object]:
    ranked_microbes = sorted(microbes.items(), key=lambda x: x[1], reverse=True)
    return {
        "top_microbes": ranked_microbes[:10],
        "gnn_features": gnn_features,
        "risk_result": risk_result,
        "recommendations": recommendations,
    }
