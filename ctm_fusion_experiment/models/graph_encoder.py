from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from ctm_fusion_experiment.models.cox_head import CoxHead


class GraphOnlyGATCoxEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        heads: int,
        edge_hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.node_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
        )
        self.first_conv = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_hidden_dim,
            dropout=dropout,
        )
        self.second_conv = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_hidden_dim,
            dropout=dropout,
        )
        self.first_norm = nn.LayerNorm(hidden_dim)
        self.second_norm = nn.LayerNorm(hidden_dim)
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = CoxHead(embedding_dim)

    def _edge_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        weights = edge_attr.view(-1, 1)
        return self.edge_encoder(torch.cat([weights, weights.pow(2), torch.log1p(weights.abs())], dim=1))

    def forward(self, batch) -> dict[str, torch.Tensor]:
        edge_features = self._edge_features(batch.edge_attr)
        nodes = self.node_projection(batch.x)
        nodes = self.first_norm(nodes + self.first_conv(nodes, batch.edge_index, edge_features))
        nodes = F.dropout(F.gelu(nodes), p=self.dropout, training=self.training)
        nodes = self.second_norm(nodes + self.second_conv(nodes, batch.edge_index, edge_features))
        nodes = F.dropout(F.gelu(nodes), p=self.dropout, training=self.training)
        pooled = torch.cat(
            [
                global_mean_pool(nodes, batch.batch),
                global_max_pool(nodes, batch.batch),
            ],
            dim=1,
        )
        graph_embedding = self.embedding_projection(pooled)
        return {
            "graph_embedding": graph_embedding,
            "risk": self.risk_head(graph_embedding),
        }
