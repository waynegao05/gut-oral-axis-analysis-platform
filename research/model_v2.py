from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation


def _compute_structure_targets(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    node_targets = torch.zeros((num_nodes, 4), device=device)
    graph_targets: List[torch.Tensor] = []
    graph_cluster_targets: List[torch.Tensor] = []

    num_graphs = int(batch.batch.max().item()) + 1 if batch.batch.numel() > 0 else 1
    for graph_id in range(num_graphs):
        node_mask = batch.batch == graph_id
        node_ids = torch.nonzero(node_mask, as_tuple=False).view(-1)
        local_n = node_ids.numel()
        if local_n == 0:
            graph_targets.append(torch.tensor(0.0, device=device))
            graph_cluster_targets.append(torch.tensor(0.0, device=device))
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
        local_weighted = wadj.sum(dim=1)

        clustering[node_ids] = local_clustering
        triangle_score[node_ids] = tri_diag / max(float(local_n), 1.0)
        bridge_proxy[node_ids] = local_bridge / max(float(local_n), 1.0)

        max_local_deg = torch.clamp(local_deg.max(), min=1.0)
        max_local_weighted = torch.clamp(local_weighted.max(), min=1.0)
        max_local_bridge = torch.clamp(local_bridge.max(), min=1.0)
        node_targets[node_ids] = torch.stack(
            [
                local_deg / max_local_deg,
                local_weighted / max_local_weighted,
                local_clustering,
                local_bridge / max_local_bridge,
            ],
            dim=1,
        )

        graph_structure_target = (
            0.35 * local_clustering.mean()
            + 0.25 * (local_weighted.mean() / max(float(local_n), 1.0))
            + 0.20 * local_bridge.mean()
            + 0.20 * (adj.sum() / max(float(local_n * max(local_n - 1, 1)), 1.0))
        )
        graph_cluster_target = 0.6 * local_clustering.mean() + 0.4 * torch.std(local_clustering, unbiased=False)
        graph_targets.append(graph_structure_target)
        graph_cluster_targets.append(graph_cluster_target)

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
    return node_struct, node_targets, torch.stack(graph_targets).view(-1, 1), torch.stack(graph_cluster_targets).view(-1, 1)


class DeepStructureAwareGATCoxModelV2(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        clinical_dim: int,
        metabolite_dim: int,
        hidden_dim: int = 96,
        heads: int = 4,
        dropout: float = 0.25,
        edge_hidden_dim: int = 24,
        num_layers: int = 4,
        layer_attn_heads: int = 4,
        contrastive_temperature: float = 0.2,
        survival_head_type: str = "cox",
        num_time_bins: int = 12,
        use_layer_attention: bool = False,
    ) -> None:
        super().__init__()
        self.structure_dim = 5
        self.num_layers = num_layers
        self.contrastive_temperature = contrastive_temperature
        self.survival_head_type = survival_head_type
        self.num_time_bins = num_time_bins
        self.use_layer_attention = use_layer_attention
        self.dropout = nn.Dropout(dropout)

        self.node_proj = nn.Sequential(
            nn.Linear(node_feature_dim + self.structure_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
        )

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layer_pool_gates = nn.ModuleList()
        self.layer_pool_projs = nn.ModuleList()
        self.layer_dims: List[int] = []

        in_dim = hidden_dim
        for layer_idx in range(num_layers):
            out_dim = hidden_dim
            concat = layer_idx != num_layers - 1
            layer_heads = heads if concat else 1
            conv = GATConv(
                in_dim,
                out_dim,
                heads=layer_heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_hidden_dim,
            )
            final_dim = out_dim * layer_heads if concat else out_dim

            self.convs.append(conv)
            self.residuals.append(nn.Linear(in_dim, final_dim))
            self.norms.append(nn.LayerNorm(final_dim))
            self.layer_dims.append(final_dim)

            self.layer_pool_gates.append(
                AttentionalAggregation(
                    gate_nn=nn.Sequential(
                        nn.Linear(final_dim, max(16, final_dim // 2)),
                        nn.GELU(),
                        nn.Linear(max(16, final_dim // 2), 1),
                    )
                )
            )

            in_dim = final_dim

        self.final_node_dim = in_dim
        self.graph_dim = self.final_node_dim * 3

        for layer_dim in self.layer_dims:
            pooled_dim = layer_dim * 3
            self.layer_pool_projs.append(
                nn.Sequential(
                    nn.Linear(pooled_dim, self.graph_dim),
                    nn.LayerNorm(self.graph_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        if self.use_layer_attention:
            self.layer_attention = nn.MultiheadAttention(
                self.graph_dim,
                num_heads=layer_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.jump_proj = nn.Sequential(
                nn.Linear(self.graph_dim * 2, self.graph_dim),
                nn.LayerNorm(self.graph_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.layer_attention = None
            self.jump_proj = None

        fusion_dim = self.graph_dim + clinical_dim + metabolite_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if survival_head_type == "cox":
            self.risk_head = nn.Linear(hidden_dim, 1)
            self.time_head = None
        elif survival_head_type == "discrete_time":
            self.risk_head = None
            self.time_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_time_bins),
            )
        else:
            raise ValueError(f"Unsupported survival_head_type: {survival_head_type}")

        self.graph_structure_head = nn.Sequential(
            nn.Linear(self.graph_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_cluster_head = nn.Sequential(
            nn.Linear(self.graph_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_target_head = nn.Sequential(
            nn.Linear(self.final_node_dim + self.structure_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.aux_loss_fn = nn.MSELoss()

    def _edge_features(self, edge_attr: torch.Tensor, augment: bool = False) -> torch.Tensor:
        base = edge_attr.view(-1, 1)
        edge_feat = torch.cat([base, base.pow(2), torch.log1p(base.abs())], dim=1)
        if augment:
            edge_feat = edge_feat + 0.03 * torch.randn_like(edge_feat)
        return self.edge_encoder(edge_feat)

    def _pool_graph(
        self,
        node_repr: torch.Tensor,
        batch_index: torch.Tensor,
        pool_gate: nn.Module,
        pool_proj: nn.Module,
    ) -> torch.Tensor:
        mean_pool = global_mean_pool(node_repr, batch_index)
        max_pool = global_max_pool(node_repr, batch_index)
        att_pool = pool_gate(node_repr, index=batch_index)
        pooled = torch.cat([mean_pool, max_pool, att_pool], dim=1)
        return pool_proj(pooled)

    def _encode(self, batch, augment: bool = False):
        node_struct, node_targets, graph_targets, graph_cluster_targets = _compute_structure_targets(batch)
        x = torch.cat([batch.x, node_struct], dim=1)
        if augment:
            x = x + 0.03 * torch.randn_like(x)
            x = F.dropout(x, p=0.08, training=True)
        x = self.node_proj(x)
        edge_emb = self._edge_features(batch.edge_attr, augment=augment)

        pooled_layers = []
        layer_states = []
        for conv, residual, norm, pool_gate, pool_proj in zip(
            self.convs,
            self.residuals,
            self.norms,
            self.layer_pool_gates,
            self.layer_pool_projs,
        ):
            h = conv(x, batch.edge_index, edge_emb)
            h = norm(h + residual(x))
            h = F.gelu(h)
            h = self.dropout(h)
            x = h
            layer_states.append(h)
            pooled_layers.append(self._pool_graph(h, batch.batch, pool_gate, pool_proj))

        if self.use_layer_attention:
            layer_stack = torch.stack(pooled_layers, dim=1)
            attn_layers, _ = self.layer_attention(layer_stack, layer_stack, layer_stack, need_weights=False)
            jump_embedding = attn_layers.mean(dim=1)
            graph_embedding = self.jump_proj(torch.cat([pooled_layers[-1], jump_embedding], dim=1))
        else:
            graph_embedding = pooled_layers[-1]
        return graph_embedding, layer_states[-1], node_struct, node_targets, graph_targets, graph_cluster_targets

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.size(0) < 2:
            return torch.tensor(0.0, device=z1.device)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.t() / self.contrastive_temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, batch, compute_contrastive: bool = False) -> Dict[str, torch.Tensor]:
        graph_embedding, final_node, node_struct, node_targets, graph_targets, graph_cluster_targets = self._encode(
            batch,
            augment=False,
        )
        clinical = batch.clinical.view(graph_embedding.size(0), -1)
        metabolites = batch.metabolites.view(graph_embedding.size(0), -1)
        fused = torch.cat([graph_embedding, clinical, metabolites], dim=1)
        latent = self.fusion(fused)
        if self.survival_head_type == "cox":
            risk = self.risk_head(latent).squeeze(-1)
            time_logits = None
        else:
            time_logits = self.time_head(latent)
            hazard = torch.sigmoid(time_logits).clamp(min=1e-6, max=1.0 - 1e-6)
            survival = torch.cumprod(1.0 - hazard, dim=1)
            risk = -survival.sum(dim=1)

        graph_aux_loss = self.aux_loss_fn(self.graph_structure_head(graph_embedding), graph_targets) + self.aux_loss_fn(
            self.graph_cluster_head(graph_embedding), graph_cluster_targets
        )
        node_aux_loss = self.aux_loss_fn(self.node_target_head(torch.cat([final_node, node_struct], dim=1)), node_targets)

        if compute_contrastive:
            z1, _, _, _, _, _ = self._encode(batch, augment=True)
            z2, _, _, _, _, _ = self._encode(batch, augment=True)
            contrastive_loss = self._contrastive_loss(z1, z2)
        else:
            contrastive_loss = torch.tensor(0.0, device=graph_embedding.device)

        aux_loss = graph_aux_loss + node_aux_loss
        return {
            "risk": risk,
            "time_logits": time_logits,
            "graph_embedding": graph_embedding,
            "latent": latent,
            "graph_aux_loss": graph_aux_loss,
            "node_aux_loss": node_aux_loss,
            "contrastive_loss": contrastive_loss,
            "aux_loss": aux_loss,
            "graph_target": graph_targets,
            "graph_cluster_target": graph_cluster_targets,
            "node_target": node_targets,
        }
