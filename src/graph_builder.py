from __future__ import annotations

from itertools import combinations
from typing import Dict

import networkx as nx


KNOWN_RELATIONS = {
    ("Fusobacterium", "Porphyromonas"): 0.8,
    ("Fusobacterium", "Prevotella"): 0.5,
    ("Streptococcus", "Prevotella"): 0.45,
    ("Lactobacillus", "Fusobacterium"): -0.3,
}


def build_microbe_graph(microbes: Dict[str, float]) -> nx.Graph:
    graph = nx.Graph()
    for node, abundance in microbes.items():
        graph.add_node(node, abundance=float(abundance))

    for left, right in combinations(microbes.keys(), 2):
        key = (left, right)
        reverse_key = (right, left)
        weight = KNOWN_RELATIONS.get(key, KNOWN_RELATIONS.get(reverse_key))
        if weight is None:
            weight = (microbes[left] + microbes[right]) / 2.0
        if abs(weight) > 0:
            graph.add_edge(left, right, weight=float(weight))
    return graph


def graph_topology_features(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {"density": 0.0, "avg_degree": 0.0, "avg_clustering": 0.0}
    density = nx.density(graph)
    avg_degree = sum(dict(graph.degree()).values()) / max(graph.number_of_nodes(), 1)
    avg_clustering = nx.average_clustering(graph) if graph.number_of_edges() > 0 else 0.0
    return {
        "density": float(density),
        "avg_degree": float(avg_degree),
        "avg_clustering": float(avg_clustering),
    }
