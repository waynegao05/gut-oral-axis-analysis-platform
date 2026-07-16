from __future__ import annotations

from typing import Any, Dict

from config.settings import WEB_MODEL_BACKEND
from src.graph_builder import build_microbe_graph, graph_topology_features
from src.preprocess import build_structured_input
from src.recommendation import generate_recommendations
from src.report import build_report


def _get_model_bridge() -> Any:
    if WEB_MODEL_BACKEND == "temporal_topology":
        from src.temporal_topology_bridge import get_temporal_topology_model_bridge

        return get_temporal_topology_model_bridge()
    if WEB_MODEL_BACKEND == "legacy_cox":
        from archive.legacy_web_backends.cox_ensemble_v1 import get_research_model_bridge

        return get_research_model_bridge()
    raise RuntimeError(
        "GOA_MODEL_BACKEND must be temporal_topology or legacy_cox; "
        f"received {WEB_MODEL_BACKEND!r}."
    )


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, object]:
    structured = build_structured_input(payload)
    graph = build_microbe_graph(structured.microbes)
    graph_features = graph_topology_features(graph)
    model_bridge = _get_model_bridge()
    model_prediction = model_bridge.score(
        structured.microbes,
        structured.clinical,
        structured.metabolites,
    )
    gnn_features = {**graph_features, **model_prediction.model_features}
    risk_result = model_prediction.risk_result
    recommendations = generate_recommendations(
        structured.microbes,
        float(risk_result["risk_score"]),
        str(risk_result["risk_level"]),
    )
    return build_report(structured.microbes, gnn_features, risk_result, recommendations)
