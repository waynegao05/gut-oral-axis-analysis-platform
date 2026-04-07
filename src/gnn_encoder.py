from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np


class LightweightGNNEncoder:
    """Runnable prototype GNN-style encoder.

    This encoder approximates a graph convolution style aggregation so the
    repository can run end-to-end before validated model weights are introduced.
    """

    def __init__(self, hidden_dim: int = 4):
        self.hidden_dim = hidden_dim

    def encode(self, graph: nx.Graph) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {
                "gnn_signal": 0.0,
                "centrality_signal": 0.0,
                "abundance_signal": 0.0,
                "module_signal": 0.0,
            }

        adjacency = nx.to_numpy_array(graph, weight="weight", dtype=float)
        features = np.array(
            [[float(graph.nodes[n].get("abundance", 0.0))] for n in graph.nodes()],
            dtype=float,
        )
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
        }
