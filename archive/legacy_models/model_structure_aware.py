from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GlobalAttention, global_max_pool, global_mean_pool


def _compute_node_structure_features(batch) -> tuple[torch.Tensor, torch.Tensor]:
    device = batch.x.device
    num_nodes = batch.x.size(0)
    edge_index = batch.edge_index
    edge_weight = batch.edge_attr.view(-1)

    degree = torch.zeros(num_nodes, device=device)
    weighted_degree = torch.zeros(num_nodes, device=device)
    ones = torch.ones(edge_index.size(1), device=device)
    degree.index_add_(0, edge_index[0], ones)
    weighted_degree.index_add_(0, edge_index[0], edge_weight)

    clustering = torch.zeros(num_nodes, device=device)
    triangle_score = torch.zeros(num_nodes, device=device)
    bridge_proxy = torch.zeros(num_nodes, device=device)
    graph_targets = []

    num_graphs = int(batch.batch.max().item()) + 1 if batch.batch.numel() > 0 else 1
    for graph_id in range(num_graphs):
        node_mask = batch.batch == graph_id
        node_ids = torch.nonzero(node_mask, as_tuple=False).view(-1)
        local_n = node_ids.numel()
        if local_n == 0:
            graph_targets.append(torch.tensor(0.0, device=device))
            continue

        local_map = {int(node_id.item()): i for i, node_id in enumerate(node_ids)}
        adj = torch.zeros((local_n, local_n), device=device)
        wadj = torch.zeros((local_n, local_n), device=device)

        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        local_edges = edge_index[:, edge_mask]
        local_weights = edge_weight[edge_mask]
        for edge_pos in range(local_edges.size(1)):
            src = int(local_edges[0, edge_pos].item())
            dst = int(local_edges[1, edge_pos].item())
            ls = local_map[src]
            ld = local_map[dst]
            if ls == ld:
                continue
            adj[ls, ld] = 1.0
            wadj[ls, ld] = max(wadj[ls, ld], local_weights[edge_pos])

        adj = torch.maximum(adj, adj.t())
        wadj = torch.maximum(wadj, wadj.t())
        local_deg = adj.sum(dim=1)
        tri_diag = torch.diagonal(adj @ adj @ adj) / 2.0
        denom = torch.clamp(local_deg * (local_deg - 1.0) / 2.0, min=1.0)
        local_clustering = tri_diag / denom
        local_bridge = local_deg * (1.0 - local_clustering)

        clustering[node_ids] = local_clustering
        triangle_score[node_ids] = tri_diag / max(float(local_n), 1.0)
        bridge_proxy[node_ids] = local_bridge / max(float(local_n), 1.0)

        graph_target = (
            0.45 * local_clustering.mean()
            + 0.35 * (wadj.sum(dim=1).mean() / max(float(local_n), 1.0))
            + 0.20 * local_bridge.mean()
        )
        graph_targets.append(graph_target)

    max_deg = torch.clamp(degree.max(), min=1.0)
    max_wdeg = torch.clamp(weighted_degree.max(), min=1.0)
    max_bridge = torch.clamp(bridge_proxy.max(), min=1.0)
    node_struct = torch.stack(
        [
            degree / max_deg,
            weighted_degree / max_wdeg,
            clustering,
            triangle_score,
            bridge_proxy / max_bridge,
        ],
        dim=1,
    )
    graph_target_tensor = torch.stack(graph_targets).view(-1, 1)
    return node_struct, graph_target_tensor


class StructureAwareGATCoxModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.25,
        edge_hidden_dim: int = 12,
    ) -> None:
        super().__init__()
        self.structure_dim = 5
        self.node_proj = nn.Sequential(
            nn.Linear(node_feature_dim + self.structure_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
        )

        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout, edge_dim=edge_hidden_dim)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout, edge_dim=edge_hidden_dim)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=edge_hidden_dim)

        self.res1 = nn.Linear(hidden_dim, hidden_dim * heads)
        self.res2 = nn.Linear(hidden_dim * heads, hidden_dim * heads)
        self.res3 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        graph_dim = hidden_dim * 3
        fusion_dim = graph_dim + clinical_dim + metabolite_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(hidden_dim // 2, 1)
        self.structure_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.aux_loss_fn = nn.MSELoss()

    def forward(self, batch):
        node_struct, graph_target = _compute_node_structure_features(batch)
        x = torch.cat([batch.x, node_struct], dim=1)
        x = self.node_proj(x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        h1 = self.gat1(x, batch.edge_index, edge_attr)
        h1 = self.norm1(h1 + self.res1(x))
        h1 = torch.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.gat2(h1, batch.edge_index, edge_attr)
        h2 = self.norm2(h2 + self.res2(h1))
        h2 = torch.relu(h2)
        h2 = self.dropout(h2)

        h3 = self.gat3(h2, batch.edge_index, edge_attr)
        h3 = self.norm3(h3 + self.res3(h2))
        h3 = torch.relu(h3)
        h3 = self.dropout(h3)

        mean_pool = global_mean_pool(h3, batch.batch)
        max_pool = global_max_pool(h3, batch.batch)
        att_pool = self.att_pool(h3, batch.batch)
        graph_embedding = torch.cat([mean_pool, max_pool, att_pool], dim=1)

        clinical = batch.clinical.view(graph_embedding.size(0), -1)
        metabolites = batch.metabolites.view(graph_embedding.size(0), -1)
        fused = torch.cat([graph_embedding, clinical, metabolites], dim=1)
        latent = self.fusion(fused)
        risk = self.risk_head(latent).squeeze(-1)

        structure_pred = self.structure_head(graph_embedding)
        aux_loss = self.aux_loss_fn(structure_pred, graph_target)
        return {
            "risk": risk,
            "graph_embedding": graph_embedding,
            "latent": latent,
            "aux_loss": aux_loss,
            "graph_target": graph_target,
        }
