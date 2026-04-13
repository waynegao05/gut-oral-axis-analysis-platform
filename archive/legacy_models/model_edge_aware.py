from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GlobalAttention, global_max_pool, global_mean_pool


class EdgeAwareGATCoxModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        hidden_dim: int = 48,
        heads: int = 4,
        dropout: float = 0.3,
        edge_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
        )

        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_hidden_dim,
        )
        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_hidden_dim,
        )

        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.graph_norm1 = nn.LayerNorm(hidden_dim * heads)
        self.graph_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        pooled_dim = hidden_dim * 3
        fusion_input_dim = pooled_dim + clinical_dim + metabolite_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch):
        edge_attr = self.edge_encoder(batch.edge_attr)
        x0 = self.node_proj(batch.x)

        x1 = self.gat1(x0, batch.edge_index, edge_attr)
        x1 = self.graph_norm1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.gat2(x1, batch.edge_index, edge_attr)
        x2 = self.graph_norm2(x2 + self.residual_proj(x0))
        x2 = torch.relu(x2)
        x2 = self.dropout(x2)

        mean_pool = global_mean_pool(x2, batch.batch)
        max_pool = global_max_pool(x2, batch.batch)
        att_pool = self.att_pool(x2, batch.batch)
        graph_embedding = torch.cat([mean_pool, max_pool, att_pool], dim=1)

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
