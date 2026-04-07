from __future__ import annotations

from typing import Dict, List


def build_report(
    microbes: Dict[str, float],
    gnn_features: Dict[str, float],
    risk_result: Dict[str, float | str],
    recommendations: List[Dict[str, object]],
) -> Dict[str, object]:
    ranked_microbes = sorted(microbes.items(), key=lambda x: x[1], reverse=True)
    return {
        "top_microbes": ranked_microbes[:10],
        "gnn_features": gnn_features,
        "risk_result": risk_result,
        "recommendations": recommendations,
    }
