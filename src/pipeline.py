from __future__ import annotations

from typing import Any, Dict

from src.graph_builder import build_microbe_graph, graph_topology_features
from src.gnn_encoder import LightweightGNNEncoder
from src.preprocess import build_structured_input
from src.recommendation import generate_recommendations
from src.report import build_report
from src.risk_model import CoxStyleRiskModel


encoder = LightweightGNNEncoder()
risk_model = CoxStyleRiskModel()


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, object]:
    structured = build_structured_input(payload)
    graph = build_microbe_graph(structured.microbes)
    graph_features = graph_topology_features(graph)
    gnn_features = encoder.encode(graph)
    gnn_features.update(graph_features)
    risk_result = risk_model.score(
        gnn_features,
        structured.microbes,
        structured.clinical,
        structured.metabolites,
    )
    recommendations = generate_recommendations(
        structured.microbes,
        float(risk_result["risk_score"]),
        str(risk_result["risk_level"]),
    )
    return build_report(structured.microbes, gnn_features, risk_result, recommendations)
