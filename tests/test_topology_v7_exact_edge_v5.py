from __future__ import annotations

import copy

import torch
from torch_geometric.data import Batch, Data

from experiments.topology_v7_exact_edge_v5.model import (
    ExactInternalEdgeGenerator,
    ExactInternalRelationModel,
    evaluate_edge_emulation,
    fit_exact_edge_parameters,
)
from research.model_v2 import DeepStructureAwareGATCoxModelV2


NUM_NODE_TYPES = 5


def _edges() -> torch.Tensor:
    source: list[int] = []
    target: list[int] = []
    for left in range(NUM_NODE_TYPES):
        for right in range(left + 1, NUM_NODE_TYPES):
            source.extend([left, right])
            target.extend([right, left])
    return torch.tensor([source, target], dtype=torch.long)


def _graphs(count: int = 50) -> list[Data]:
    edge_index = _edges()
    upper = torch.triu_indices(
        NUM_NODE_TYPES,
        NUM_NODE_TYPES,
        offset=1,
    )
    abundance_rows = []
    for index in range(count):
        abundance_rows.append(
            torch.sigmoid(
                torch.linspace(-1.8, 1.4, NUM_NODE_TYPES)
                + 0.35 * torch.sin(torch.tensor(index / 3.0))
                + 0.025
                * index
                * torch.arange(NUM_NODE_TYPES)
            )
        )
    abundance_matrix = torch.stack(abundance_rows)
    logit = torch.log(
        abundance_matrix / (1.0 - abundance_matrix)
    )
    reference_z = (
        logit - logit.mean(dim=0)
    ) / logit.std(dim=0, unbiased=False)
    result: list[Data] = []
    for index in range(count):
        abundance = abundance_matrix[index]
        source = reference_z[index, upper[0]]
        target = reference_z[index, upper[1]]
        pair_id = torch.arange(upper.size(1), dtype=torch.float32)
        argument = (
            -0.2
            + 0.08 * pair_id
            + 0.15 * source
            - 0.12 * target
            + (-1.0) ** pair_id * source * target / 2.0
        )
        pair_weights = torch.clamp(
            0.35
            + 0.01 * pair_id
            + 0.12 * torch.tanh(argument),
            0.02,
            0.98,
        )
        directed = torch.empty(20)
        directed[0::2] = pair_weights
        directed[1::2] = pair_weights
        result.append(
            Data(
                x=torch.stack(
                    [
                        abundance,
                        torch.linspace(0.2, 0.8, NUM_NODE_TYPES),
                    ],
                    dim=1,
                ),
                edge_index=edge_index,
                edge_attr=directed.view(-1, 1),
                node_type=torch.arange(NUM_NODE_TYPES),
                node_struct=torch.rand(NUM_NODE_TYPES, 5),
                node_targets=torch.rand(NUM_NODE_TYPES, 4),
                graph_targets=torch.rand(1),
                graph_cluster_targets=torch.rand(1),
                clinical=torch.tensor([0.2, -0.1, 0.3, 0.5]),
                metabolites=torch.tensor([0.1, 0.4, 0.7]),
                site_features=torch.linspace(-0.5, 0.5, 12),
                time=torch.tensor(24.0 + index),
                event=torch.tensor(float(index % 2)),
                generation_group_id=torch.tensor(index % 5),
                sample_id=f"sample_{index}",
            )
        )
    return result


def _model(data_set: list[Data]) -> ExactInternalRelationModel:
    base = DeepStructureAwareGATCoxModelV2(
        node_feature_dim=2,
        clinical_dim=4,
        metabolite_dim=3,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
        edge_hidden_dim=8,
        num_layers=2,
        layer_attn_heads=2,
        survival_head_type="cox",
        num_time_bins=4,
        use_layer_attention=False,
        num_node_types=NUM_NODE_TYPES,
        node_identity_dim=4,
        identity_readout_dim=8,
        pool_every_layer=False,
    )
    parameters = fit_exact_edge_parameters(
        data_set,
        num_node_types=NUM_NODE_TYPES,
    )
    return ExactInternalRelationModel(
        base,
        num_node_types=NUM_NODE_TYPES,
        parameters=parameters,
    )


def test_exact_fit_generalizes_to_held_out_graphs() -> None:
    data_set = _graphs()
    parameters = fit_exact_edge_parameters(
        data_set[:40],
        num_node_types=NUM_NODE_TYPES,
    )
    generator = ExactInternalEdgeGenerator(
        num_node_types=NUM_NODE_TYPES,
        parameters=parameters,
    )
    held_out = evaluate_edge_emulation(
        generator,
        data_set[40:],
        device=torch.device("cpu"),
    )

    assert parameters.fit_report["uses_time_or_event"] is False
    assert parameters.fit_report["num_parameters_per_edge"] == 6
    assert held_out["r2"] > 0.999
    assert held_out["mae"] < 1e-4


def test_forward_ignores_external_structure_and_outcomes() -> None:
    data_set = _graphs()
    model = _model(data_set[:40]).eval()
    original = Batch.from_data_list(data_set[40:42])
    changed = copy.deepcopy(original)
    changed.edge_attr = torch.rand_like(changed.edge_attr) * 100.0
    changed.node_struct = torch.rand_like(changed.node_struct) * 100.0
    changed.node_targets = torch.rand_like(changed.node_targets) * 100.0
    changed.graph_targets = torch.rand_like(changed.graph_targets) * 100.0
    changed.graph_cluster_targets = (
        torch.rand_like(changed.graph_cluster_targets) * 100.0
    )
    changed.time = changed.time * 100.0
    changed.event = 1.0 - changed.event
    changed.generation_group_id = changed.generation_group_id + 20

    with torch.no_grad():
        first = model(original)
        second = model(changed)

    assert torch.equal(first["risk"], second["risk"])
    assert torch.equal(
        first["pair_edge_weights"],
        second["pair_edge_weights"],
    )


def test_node_abundance_controls_internal_edges() -> None:
    data_set = _graphs()
    model = _model(data_set[:40]).eval()
    first_batch = Batch.from_data_list(data_set[40:42])
    second_batch = copy.deepcopy(first_batch)
    second_batch.x[:, 0] = torch.clamp(
        second_batch.x[:, 0] + 0.08,
        max=0.95,
    )

    with torch.no_grad():
        first = model(first_batch)
        second = model(second_batch)

    assert first["pair_edge_weights"].shape == (2, 10)
    assert torch.all(first["pair_edge_weights"] >= 0.02)
    assert torch.all(first["pair_edge_weights"] <= 0.98)
    assert not torch.allclose(
        first["pair_edge_weights"],
        second["pair_edge_weights"],
    )
