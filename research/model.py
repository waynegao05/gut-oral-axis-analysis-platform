from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATCoxModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        fusion_dim = hidden_dim + clinical_dim + metabolite_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch):
        x = self.gat1(batch.x, batch.edge_index, batch.edge_attr)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, batch.edge_index, batch.edge_attr)
        x = self.graph_norm(x)
        graph_embedding = global_mean_pool(x, batch.batch)

        clinical = batch.clinical.view(graph_embedding.size(0), -1)
        metabolites = batch.metabolites.view(graph_embedding.size(0), -1)
        fused = torch.cat([graph_embedding, clinical, metabolites], dim=1)
        latent = self.fusion(fused)
        risk = self.risk_head(latent).squeeze(-1)
        return {
            "risk": risk,
            "graph_embedding": graph_embedding,
            "latent": latent,
        }
