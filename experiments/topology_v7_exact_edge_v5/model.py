from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares

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
class ExactEdgeParameters:
    logit_mean: torch.Tensor
    logit_scale: torch.Tensor
    center: torch.Tensor
    amplitude: torch.Tensor
    tanh_coefficients: torch.Tensor
    fit_report: dict[str, float]


def _canonical_abundance(
    data_set: Sequence[Any],
    *,
    num_node_types: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for item in data_set:
        values = torch.zeros(num_node_types, dtype=torch.float64)
        values[item.node_type.long()] = item.x[:, 0].detach().double()
        rows.append(values)
    return torch.stack(rows, dim=0)


def _design_matrix(
    standardized: np.ndarray,
    pair_source: np.ndarray,
    pair_target: np.ndarray,
) -> np.ndarray:
    source = standardized[:, pair_source]
    target = standardized[:, pair_target]
    return np.stack(
        [
            np.ones_like(source),
            source,
            target,
            source * target,
        ],
        axis=2,
    )


def _predict_pair(
    parameters: np.ndarray,
    design: np.ndarray,
) -> np.ndarray:
    center = parameters[0]
    amplitude = parameters[1]
    argument = design @ parameters[2:]
    return np.clip(
        center + amplitude * np.tanh(argument),
        0.02,
        0.98,
    )


def fit_exact_edge_parameters(
    data_set: Sequence[Any],
    *,
    num_node_types: int,
    maximum_function_evaluations: int = 3000,
) -> ExactEdgeParameters:
    """Fit the V7 bounded bilinear edge rule without using outcomes."""
    if len(data_set) < 8:
        raise ValueError("At least eight training graphs are required.")

    abundance = _canonical_abundance(
        data_set,
        num_node_types=num_node_types,
    ).clamp(min=1e-6, max=1.0 - 1e-6)
    logit = torch.log(abundance / (1.0 - abundance))
    mean = logit.mean(dim=0)
    scale = logit.std(dim=0, unbiased=False)
    scale = torch.where(scale <= 1e-8, torch.ones_like(scale), scale)
    standardized = ((logit - mean) / scale).numpy()

    upper = torch.triu_indices(
        num_node_types,
        num_node_types,
        offset=1,
    )
    pair_source = upper[0].numpy()
    pair_target = upper[1].numpy()
    design = _design_matrix(
        standardized,
        pair_source,
        pair_target,
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
    ).double().numpy()

    coefficients: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    lower = np.asarray(
        [0.0, 0.0, -12.0, -12.0, -12.0, -12.0],
        dtype=float,
    )
    upper_bound = np.asarray(
        [1.0, 0.5, 12.0, 12.0, 12.0, 12.0],
        dtype=float,
    )
    for pair_index in range(targets.shape[1]):
        target = targets[:, pair_index]
        amplitude = float(
            np.clip(
                (
                    np.quantile(target, 0.95)
                    - np.quantile(target, 0.05)
                )
                / 2.0,
                0.02,
                0.30,
            )
        )
        best_error = float("inf")
        best_parameters: np.ndarray | None = None
        pair_design = design[:, pair_index, :]
        for interaction_sign in (-1.0, 1.0):
            initial = np.asarray(
                [
                    float(np.mean(target)),
                    amplitude,
                    0.0,
                    0.0,
                    0.0,
                    interaction_sign,
                ],
                dtype=float,
            )
            result = least_squares(
                lambda values: (
                    _predict_pair(values, pair_design) - target
                ),
                initial,
                bounds=(lower, upper_bound),
                max_nfev=int(maximum_function_evaluations),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            prediction = _predict_pair(result.x, pair_design)
            error = float(np.mean((prediction - target) ** 2))
            if error < best_error:
                best_error = error
                best_parameters = result.x
        if best_parameters is None:
            raise RuntimeError(
                f"Exact edge fit failed for pair {pair_index}."
            )
        coefficients.append(best_parameters)
        predictions.append(
            _predict_pair(best_parameters, pair_design)
        )

    coefficient = np.stack(coefficients, axis=0)
    prediction = np.column_stack(predictions)
    residual = targets - prediction
    denominator = float(
        np.sum(
            (
                targets
                - targets.mean(axis=0, keepdims=True)
            )
            ** 2
        )
    )
    fit_report = {
        "r2": float(
            1.0
            - np.sum(residual**2)
            / max(denominator, 1e-12)
        ),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "correlation": float(
            np.corrcoef(
                targets.ravel(),
                prediction.ravel(),
            )[0, 1]
        ),
        "uses_time_or_event": False,
        "num_parameters_per_edge": 6,
    }
    return ExactEdgeParameters(
        logit_mean=mean.float(),
        logit_scale=scale.float(),
        center=torch.from_numpy(coefficient[:, 0]).float(),
        amplitude=torch.from_numpy(coefficient[:, 1]).float(),
        tanh_coefficients=torch.from_numpy(
            coefficient[:, 2:]
        ).float(),
        fit_report=fit_report,
    )


class ExactInternalEdgeGenerator(nn.Module):
    def __init__(
        self,
        *,
        num_node_types: int,
        parameters: ExactEdgeParameters,
    ) -> None:
        super().__init__()
        self.num_node_types = int(num_node_types)
        pair_indices = torch.triu_indices(
            self.num_node_types,
            self.num_node_types,
            offset=1,
        )
        self.register_buffer(
            "pair_source",
            pair_indices[0],
            persistent=False,
        )
        self.register_buffer(
            "pair_target",
            pair_indices[1],
            persistent=False,
        )
        lookup = torch.full(
            (self.num_node_types, self.num_node_types),
            -1,
            dtype=torch.long,
        )
        for pair_id, (source, target) in enumerate(
            pair_indices.t().tolist()
        ):
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
            "center",
            parameters.center.detach().float().clone(),
        )
        self.register_buffer(
            "amplitude",
            parameters.amplitude.detach().float().clone(),
        )
        self.register_buffer(
            "tanh_coefficients",
            parameters.tanh_coefficients.detach().float().clone(),
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
        source = standardized[:, self.pair_source]
        target = standardized[:, self.pair_target]
        design = torch.stack(
            [
                torch.ones_like(source),
                source,
                target,
                source * target,
            ],
            dim=2,
        )
        argument = torch.sum(
            design * self.tanh_coefficients.unsqueeze(0),
            dim=2,
        )
        weights = torch.clamp(
            self.center.unsqueeze(0)
            + self.amplitude.unsqueeze(0) * torch.tanh(argument),
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
            raise ValueError("Invalid exact analytic edge pair.")
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
            pair_logits=argument,
            pair_weights=weights,
            directed_weights=directed,
            edge_matrix=matrix,
            delta_regularization=weights.new_zeros(()),
            saturation_regularization=weights.new_zeros(()),
        )


class ExactInternalRelationModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_node_types: int,
        parameters: ExactEdgeParameters,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_node_types = int(num_node_types)
        self.edge_generator = ExactInternalEdgeGenerator(
            num_node_types=num_node_types,
            parameters=parameters,
        )

    def forward(
        self,
        batch: Any,
        compute_contrastive: bool = False,
    ) -> dict[str, torch.Tensor]:
        edge_state = self.edge_generator(batch)
        internal_batch = batch.clone()
        internal_batch.edge_attr = (
            edge_state.directed_weights.view(-1, 1)
        )
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
        output.update(
            {
                "pair_edge_weights": edge_state.pair_weights,
                "directed_edge_weights": edge_state.directed_weights,
            }
        )
        return output


def evaluate_edge_emulation(
    generator: ExactInternalEdgeGenerator,
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
        np.sum(
            (
                target
                - target.mean(axis=0, keepdims=True)
            )
            ** 2
        )
    )
    return {
        "r2": float(
            1.0
            - np.sum(residual**2)
            / max(denominator, 1e-12)
        ),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "correlation": float(
            np.corrcoef(
                target.ravel(),
                prediction.ravel(),
            )[0, 1]
        ),
    }
