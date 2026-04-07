from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

APP_NAME = "Gut-Oral Axis Prototype"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True

DEFAULT_MICROBE_WEIGHTS = {
    "Fusobacterium": 1.2,
    "Porphyromonas": 1.1,
    "Prevotella": 0.7,
    "Streptococcus": 0.6,
    "Lactobacillus": -0.4,
}
DEFAULT_CLINICAL_WEIGHTS = {"age": 0.02, "bmi": 0.03, "smoking": 0.4, "family_history": 0.5}
DEFAULT_METABOLITE_WEIGHTS = {"bile_acids": 0.15, "scfa": -0.12, "tryptophan_metabolism": 0.11}
RISK_THRESHOLDS = {"low": 0.6, "medium": 1.2}
KNOWN_RELATIONS = {
    ("Fusobacterium", "Porphyromonas"): 0.8,
    ("Fusobacterium", "Prevotella"): 0.5,
    ("Streptococcus", "Prevotella"): 0.45,
    ("Lactobacillus", "Fusobacterium"): -0.3,
}

RULE_PATH = Path(__file__).resolve().parent / "data" / "microbe_drug_rules.json"


def _coerce_numeric_map(payload: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def normalize_relative_abundance(microbes: Dict[str, float]) -> Dict[str, float]:
    if not microbes:
        return {}
    total = sum(max(v, 0.0) for v in microbes.values())
    if total <= 0:
        return {k: 0.0 for k in microbes}
    return {k: max(v, 0.0) / total for k, v in microbes.items()}


def build_microbe_graph(microbes: Dict[str, float]) -> nx.Graph:
    graph = nx.Graph()
    for node, abundance in microbes.items():
        graph.add_node(node, abundance=float(abundance))
    for left, right in combinations(microbes.keys(), 2):
        weight = KNOWN_RELATIONS.get((left, right), KNOWN_RELATIONS.get((right, left)))
        if weight is None:
            weight = (microbes[left] + microbes[right]) / 2.0
        graph.add_edge(left, right, weight=float(weight))
    return graph


def encode_graph(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {"gnn_signal": 0.0, "centrality_signal": 0.0, "abundance_signal": 0.0, "module_signal": 0.0}
    adjacency = nx.to_numpy_array(graph, weight="weight", dtype=float)
    features = np.array([[float(graph.nodes[n].get("abundance", 0.0))] for n in graph.nodes()], dtype=float)
    identity = np.eye(adjacency.shape[0])
    adjacency_hat = adjacency + identity
    degree = np.sum(np.abs(adjacency_hat), axis=1)
    degree[degree == 0] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    normalized = d_inv_sqrt @ adjacency_hat @ d_inv_sqrt
    hidden = normalized @ features
    centrality = nx.degree_centrality(graph)
    return {
        "gnn_signal": float(hidden.mean()),
        "centrality_signal": float(np.mean(list(centrality.values()))) if centrality else 0.0,
        "abundance_signal": float(features.mean()) if features.size else 0.0,
        "module_signal": float(nx.average_clustering(graph)) if graph.number_of_edges() > 0 else 0.0,
        "density": float(nx.density(graph)),
    }


def score_risk(gnn_features: Dict[str, float], microbes: Dict[str, float], clinical: Dict[str, float], metabolites: Dict[str, float]) -> Dict[str, float | str]:
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


def load_rules() -> List[Dict[str, object]]:
    if not RULE_PATH.exists():
        return []
    return json.loads(RULE_PATH.read_text(encoding="utf-8")).get("rules", [])


def generate_recommendations(microbes: Dict[str, float], risk_level: str) -> List[Dict[str, object]]:
    recommendations: List[Dict[str, object]] = []
    for rule in load_rules():
        marker = str(rule.get("marker", ""))
        direction = str(rule.get("direction", "increase"))
        abundance = float(microbes.get(marker, 0.0))
        triggered = abundance > 0.1 if direction == "increase" else abundance < 0.03
        if triggered:
            priority = float(rule.get("priority", 0.5)) + (0.2 if risk_level == "high" else 0.0)
            recommendations.append({
                "marker": marker,
                "abundance": round(abundance, 4),
                "priority": round(priority, 4),
                "suggestion": rule.get("suggestion"),
                "rationale": rule.get("rationale"),
            })
    recommendations.sort(key=lambda x: x["priority"], reverse=True)
    if not recommendations:
        recommendations.append({
            "marker": "general",
            "abundance": 0.0,
            "priority": 0.3,
            "suggestion": "Maintain routine follow-up and continue longitudinal microbiome monitoring.",
            "rationale": f"No strong prototype rule was triggered; current risk level is {risk_level}.",
        })
    return recommendations


def run_pipeline(payload: Dict[str, Any]) -> Dict[str, object]:
    microbes = normalize_relative_abundance(_coerce_numeric_map(payload.get("microbes", {})))
    clinical = _coerce_numeric_map(payload.get("clinical", {}))
    metabolites = _coerce_numeric_map(payload.get("metabolites", {}))
    graph = build_microbe_graph(microbes)
    gnn_features = encode_graph(graph)
    risk_result = score_risk(gnn_features, microbes, clinical, metabolites)
    recommendations = generate_recommendations(microbes, str(risk_result["risk_level"]))
    ranked_microbes = sorted(microbes.items(), key=lambda x: x[1], reverse=True)
    return {
        "top_microbes": ranked_microbes[:10],
        "gnn_features": gnn_features,
        "risk_result": risk_result,
        "recommendations": recommendations,
    }


INDEX_HTML = """
<!doctype html>
<html lang='en'>
  <head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>{{ app_name }}</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 920px; margin: 32px auto; padding: 0 16px; }
      textarea { width: 100%; min-height: 260px; font-family: monospace; }
      button { margin-top: 12px; padding: 10px 18px; }
      pre { background: #f5f5f5; padding: 16px; overflow: auto; }
    </style>
  </head>
  <body>
    <h1>{{ app_name }}</h1>
    <p>Paste JSON payload and click Analyze.</p>
    <textarea id='payload'>{
  "microbes": {
    "Fusobacterium": 0.18,
    "Porphyromonas": 0.15,
    "Prevotella": 0.10,
    "Streptococcus": 0.09,
    "Lactobacillus": 0.02
  },
  "clinical": {
    "age": 52,
    "bmi": 24.5,
    "smoking": 1,
    "family_history": 1
  },
  "metabolites": {
    "bile_acids": 0.8,
    "scfa": 0.3,
    "tryptophan_metabolism": 0.7
  }
}</textarea>
    <br>
    <button onclick='runAnalysis()'>Analyze</button>
    <h2>Output</h2>
    <pre id='output'></pre>
    <script>
      async function runAnalysis() {
        const payload = JSON.parse(document.getElementById('payload').value);
        const res = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      }
    </script>
  </body>
</html>
"""

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, app_name=APP_NAME)

@app.route('/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(force=True)
    return jsonify(run_pipeline(payload))

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
