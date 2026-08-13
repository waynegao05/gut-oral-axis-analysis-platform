from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments.topology_v7_compositional_temporal_v1.model import (
    EdgeState,
    canonical_node_tensor,
)
from experiments.topology_v7_internalized_edge_v2.model import (
    internal_complete_graph_structure,
)
from experiments.topology_v7_internalized_edge_v2.runner import (
    canonical_precomputed_edge_target,
)


@dataclass(frozen=True)
class AnalyticEdgeParameters:
    logit_mean: torch.Tensor
    logit_scale: torch.Tensor
    intercept: torch.Tensor
    slope: torch.Tensor
    fit_report: dict[str, float]


def _canonical_abundance(
    data_set: Sequence[Any],
    *,
    num_node_types: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for item in data_set:
        values = torch.zeros(num_node_types, dtype=torch.float32)
        values[item.node_type.long()] = item.x[:, 0].detach().float()
        rows.append(values)
    return torch.stack(rows, dim=0)


def fit_analytic_edge_parameters(
    data_set: Sequence[Any],
    *,
    num_node_types: int,
) -> AnalyticEdgeParameters:
    if len(data_set) < 2:
        raise ValueError("At least two training graphs are required.")
    abundance = _canonical_abundance(
        data_set,
        num_node_types=num_node_types,
    ).clamp(min=1e-6, max=1.0 - 1e-6)
    logit = torch.log(abundance / (1.0 - abundance))
    mean = logit.mean(dim=0)
    scale = logit.std(dim=0, unbiased=False)
    scale = torch.where(scale <= 1e-8, torch.ones_like(scale), scale)
    standardized = (logit - mean) / scale
    upper = torch.triu_indices(
        num_node_types,
        num_node_types,
        offset=1,
    )
    signal = torch.tanh(
        standardized[:, upper[0]]
        * standardized[:, upper[1]]
        / 2.0
    )
    targets = torch.stack(
        [
            canonical_precomputed_edge_target(
                item,
                num_node_types=num_node_types,
            )
            for item in data_set
        ],
        dim=0,
    ).float()
    design = torch.stack(
        [torch.ones_like(signal), signal],
        dim=2,
    )
    coefficients: list[torch.Tensor] = []
    for pair_index in range(signal.size(1)):
        coefficients.append(
            torch.linalg.lstsq(
                design[:, pair_index, :],
                targets[:, pair_index],
            ).solution
        )
    coefficient = torch.stack(coefficients, dim=0)
    prediction = torch.clamp(
        coefficient[:, 0].unsqueeze(0)
        + signal * coefficient[:, 1].unsqueeze(0),
        min=0.02,
        max=0.98,
    )
    target_np = targets.numpy()
    prediction_np = prediction.numpy()
    residual = target_np - prediction_np
    denominator = float(
        np.sum((target_np - target_np.mean(axis=0, keepdims=True)) ** 2)
    )
    fit_report = {
        "r2": float(
            1.0 - np.sum(residual**2) / max(denominator, 1e-12)
        ),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "correlation": float(
            np.corrcoef(target_np.ravel(), prediction_np.ravel())[0, 1]
        ),
    }
    return AnalyticEdgeParameters(
        logit_mean=mean,
        logit_scale=scale,
        intercept=coefficient[:, 0],
        slope=coefficient[:, 1],
        fit_report=fit_report,
    )


class AnalyticInternalEdgeGenerator(nn.Module):
    def __init__(
        self,
        *,
        num_node_types: int,
        parameters: AnalyticEdgeParameters,
    ) -> None:
        super().__init__()
        self.num_node_types = int(num_node_types)
        upper = torch.triu_indices(
            self.num_node_types,
            self.num_node_types,
            offset=1,
        )
        self.register_buffer("pair_source", upper[0], persistent=False)
        self.register_buffer("pair_target", upper[1], persistent=False)
        lookup = torch.full(
            (self.num_node_types, self.num_node_types),
            -1,
            dtype=torch.long,
        )
        for pair_id, (source, target) in enumerate(upper.t().tolist()):
            lookup[source, target] = pair_id
            lookup[target, source] = pair_id
        self.register_buffer("pair_lookup", lookup, persistent=False)
        self.register_buffer(
            "logit_mean",
            parameters.logit_mean.detach().float().clone(),
        )
        self.register_buffer(
            "logit_scale",
            parameters.logit_scale.detach().float().clone(),
        )
        self.register_buffer(
            "intercept",
            parameters.intercept.detach().float().clone(),
        )
        self.register_buffer(
            "slope",
            parameters.slope.detach().float().clone(),
        )
        self.fit_report = dict(parameters.fit_report)

    @property
    def num_pairs(self) -> int:
        return int(self.pair_source.numel())

    def forward(self, batch: Any) -> EdgeState:
        canonical = canonical_node_tensor(
            batch,
            num_node_types=self.num_node_types,
        )
        abundance = canonical[:, :, 0].clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
        logit = torch.log(abundance / (1.0 - abundance))
        standardized = (logit - self.logit_mean) / self.logit_scale
        signal = torch.tanh(
            standardized[:, self.pair_source]
            * standardized[:, self.pair_target]
            / 2.0
        )
        weights = torch.clamp(
            self.intercept.unsqueeze(0)
            + signal * self.slope.unsqueeze(0),
            min=0.02,
            max=0.98,
        )

        source_node = batch.edge_index[0].long()
        target_node = batch.edge_index[1].long()
        source_graph = batch.batch[source_node].long()
        target_graph = batch.batch[target_node].long()
        if not torch.equal(source_graph, target_graph):
            raise ValueError("Cross-graph edges are prohibited.")
        source_type = batch.node_type[source_node].long()
        target_type = batch.node_type[target_node].long()
        pair_id = self.pair_lookup[source_type, target_type]
        if torch.any(pair_id < 0):
            raise ValueError("Invalid analytic edge pair.")
        directed = weights[source_graph, pair_id]
        matrix = weights.new_zeros(
            (
                weights.size(0),
                self.num_node_types,
                self.num_node_types,
            )
        )
        matrix[:, self.pair_source, self.pair_target] = weights
        matrix[:, self.pair_target, self.pair_source] = weights
        return EdgeState(
            pair_logits=signal,
            pair_weights=weights,
            directed_weights=directed,
            edge_matrix=matrix,
            delta_regularization=weights.new_zeros(()),
            saturation_regularization=weights.new_zeros(()),
        )


class AnalyticInternalRelationModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_node_types: int,
        site_feature_dim: int,
        parameters: AnalyticEdgeParameters,
        use_linear_site_residual: bool,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_node_types = int(num_node_types)
        self.site_feature_dim = int(site_feature_dim)
        self.use_linear_site_residual = bool(use_linear_site_residual)
        self.residual_scale = float(residual_scale)
        self.edge_generator = AnalyticInternalEdgeGenerator(
            num_node_types=num_node_types,
            parameters=parameters,
        )
        self.site_residual = nn.Linear(site_feature_dim, 1)
        nn.init.zeros_(self.site_residual.weight)
        nn.init.zeros_(self.site_residual.bias)

    def forward(
        self,
        batch: Any,
        compute_contrastive: bool = False,
    ) -> dict[str, torch.Tensor]:
        edge_state = self.edge_generator(batch)
        internal_batch = batch.clone()
        internal_batch.edge_attr = edge_state.directed_weights.view(-1, 1)
        (
            internal_batch.node_struct,
            internal_batch.node_targets,
            internal_batch.graph_targets,
            internal_batch.graph_cluster_targets,
        ) = internal_complete_graph_structure(
            internal_batch,
            edge_state,
            num_node_types=self.num_node_types,
        )
        output = self.base_model(
            internal_batch,
            compute_contrastive=compute_contrastive,
        )
        if self.use_linear_site_residual:
            site = batch.site_features.view(-1, self.site_feature_dim)
            residual = self.residual_scale * torch.tanh(
                self.site_residual(site).squeeze(-1)
            )
        else:
            residual = torch.zeros_like(output["risk"])
        output["base_risk"] = output["risk"]
        output["risk"] = output["risk"] + residual
        output.update(
            {
                "site_risk_residual": residual,
                "pair_edge_weights": edge_state.pair_weights,
                "directed_edge_weights": edge_state.directed_weights,
            }
        )
        return output


def evaluate_edge_emulation(
    generator: AnalyticInternalEdgeGenerator,
    data_set: Sequence[Any],
    *,
    device: torch.device,
) -> dict[str, float]:
    from torch_geometric.loader import DataLoader

    loader = DataLoader(
        data_set,
        batch_size=len(data_set),
        shuffle=False,
    )
    generator.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        prediction = generator(batch).pair_weights.cpu().numpy()
    target = np.vstack(
        [
            canonical_precomputed_edge_target(
                item,
                num_node_types=generator.num_node_types,
            ).numpy()
            for item in data_set
        ]
    )
    residual = target - prediction
    denominator = float(
        np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    )
    return {
        "r2": float(
            1.0 - np.sum(residual**2) / max(denominator, 1e-12)
        ),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "correlation": float(
            np.corrcoef(target.ravel(), prediction.ravel())[0, 1]
        ),
    }
