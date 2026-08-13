from __future__ import annotations

import copy

import torch
from torch_geometric.data import Batch, Data

from experiments.topology_v7_compositional_temporal_v1.data_adapter import (
    sanitize_graph_item,
)
from experiments.topology_v7_compositional_temporal_v1.losses import (
    discrete_time_nll,
    dual_survival_objective,
    fit_discrete_time_cutpoints,
)
from experiments.topology_v7_compositional_temporal_v1.model import (
    InternalEdgeGATDualSurvivalModel,
)


NUM_NODE_TYPES = 5


def _complete_directed_edges() -> torch.Tensor:
    source: list[int] = []
    target: list[int] = []
    for left in range(NUM_NODE_TYPES):
        for right in range(left + 1, NUM_NODE_TYPES):
            source.extend([left, right])
            target.extend([right, left])
    return torch.tensor([source, target], dtype=torch.long)


def _graph(offset: float, *, event: float = 1.0) -> Data:
    node_type = torch.arange(NUM_NODE_TYPES, dtype=torch.long)
    abundance = torch.tensor(
        [0.10, 0.20, 0.25, 0.15, 0.30], dtype=torch.float32
    )
    function = torch.tensor(
        [0.40, 0.55, 0.60, 0.45, 0.70], dtype=torch.float32
    )
    x = torch.stack(
        [abundance + offset * 0.01, function + offset * 0.02], dim=1
    )
    edges = _complete_directed_edges()
    return Data(
        x=x,
        edge_index=edges,
        edge_attr=torch.linspace(0.1, 0.9, edges.size(1)).view(-1, 1),
        node_type=node_type,
        node_struct=torch.ones(NUM_NODE_TYPES, 5),
        node_targets=torch.ones(NUM_NODE_TYPES, 4),
        graph_targets=torch.ones(1),
        graph_cluster_targets=torch.ones(1),
        clinical=torch.tensor(
            [0.2 + offset, -0.1, 0.3, 1.0], dtype=torch.float32
        ),
        metabolites=torch.tensor(
            [0.1, 0.4 + offset, 0.7], dtype=torch.float32
        ),
        time=torch.tensor(24.0 + 12.0 * offset),
        event=torch.tensor(event),
        generation_group_id=torch.tensor(int(offset) % 5),
        provenance_code=torch.tensor(100.0 + offset),
        sample_id=f"sample_{offset}",
    )


def _model(edge_mode: str = "node_context") -> InternalEdgeGATDualSurvivalModel:
    torch.manual_seed(41)
    return InternalEdgeGATDualSurvivalModel(
        node_feature_dim=2,
        clinical_dim=4,
        metabolite_dim=3,
        num_node_types=NUM_NODE_TYPES,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
        edge_hidden_dim=8,
        node_identity_dim=4,
        edge_mode=edge_mode,
        num_time_bins=4,
    )


def test_adapter_removes_all_precomputed_edge_and_structure_values() -> None:
    sanitized = sanitize_graph_item(
        _graph(0.0), num_node_types=NUM_NODE_TYPES
    )

    for attribute in (
        "edge_attr",
        "node_struct",
        "node_targets",
        "graph_targets",
        "graph_cluster_targets",
    ):
        assert attribute not in sanitized
    assert sanitized.internal_edge_policy == "computed_inside_forward"


def test_internal_edges_are_symmetric_bounded_and_ignore_csv_weights() -> None:
    model = _model().eval()
    first = Batch.from_data_list([_graph(0.0), _graph(1.0)])
    second = copy.deepcopy(first)
    second.edge_attr = torch.rand_like(second.edge_attr) * 1000.0

    with torch.no_grad():
        output_first = model(first)
        output_second = model(second)

    assert torch.allclose(
        output_first["risk"], output_second["risk"], atol=0.0, rtol=0.0
    )
    assert torch.allclose(
        output_first["pair_edge_weights"],
        output_second["pair_edge_weights"],
        atol=0.0,
        rtol=0.0,
    )
    pair_weights = output_first["pair_edge_weights"]
    directed = output_first["directed_edge_weights"].view(2, -1)
    assert torch.all(pair_weights >= 0.02)
    assert torch.all(pair_weights <= 1.0)
    assert torch.allclose(directed[:, 0::2], directed[:, 1::2])


def test_outcomes_groups_and_audit_metadata_cannot_change_forward_output() -> None:
    model = _model().eval()
    original = Batch.from_data_list([_graph(0.0), _graph(1.0)])
    changed = copy.deepcopy(original)
    changed.time = original.time.flip(0) * 100.0
    changed.event = 1.0 - original.event
    changed.generation_group_id = original.generation_group_id + 99
    changed.provenance_code = original.provenance_code * -1000.0

    with torch.no_grad():
        first = model(original)
        second = model(changed)

    assert torch.allclose(first["risk"], second["risk"], atol=0.0, rtol=0.0)
    assert torch.allclose(
        first["pair_edge_weights"],
        second["pair_edge_weights"],
        atol=0.0,
        rtol=0.0,
    )


def test_same_sample_has_same_edges_alone_or_in_mixed_batch() -> None:
    model = _model().eval()
    alone = Batch.from_data_list([_graph(0.0)])
    mixed = Batch.from_data_list([_graph(0.0), _graph(3.0)])

    with torch.no_grad():
        alone_output = model(alone)
        mixed_output = model(mixed)

    assert torch.allclose(
        alone_output["pair_edge_weights"][0],
        mixed_output["pair_edge_weights"][0],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        alone_output["risk"][0],
        mixed_output["risk"][0],
        atol=1e-6,
        rtol=0.0,
    )


def test_node_and_edge_storage_permutation_preserves_prediction() -> None:
    model = _model().eval()
    original = _graph(0.0)
    permutation = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(NUM_NODE_TYPES)
    permuted = copy.deepcopy(original)
    permuted.x = original.x[permutation]
    permuted.node_type = original.node_type[permutation]
    permuted.edge_index = inverse[original.edge_index][:, torch.randperm(20)]
    permuted.edge_attr = torch.rand(20, 1)

    with torch.no_grad():
        first = model(Batch.from_data_list([original]))
        second = model(Batch.from_data_list([permuted]))

    assert torch.allclose(
        first["pair_edge_weights"],
        second["pair_edge_weights"],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(first["risk"], second["risk"], atol=1e-6, rtol=0.0)


def test_invalid_topology_fails_immediately() -> None:
    model = _model().eval()
    graph = _graph(0.0)
    graph.edge_index = graph.edge_index[:, :-1]
    graph.edge_attr = graph.edge_attr[:-1]

    try:
        model(Batch.from_data_list([graph]))
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("A missing edge direction must be rejected.")


def test_dual_head_backward_reaches_internal_edge_parameters() -> None:
    model = _model().train()
    batch = Batch.from_data_list(
        [
            _graph(0.0, event=1.0),
            _graph(1.0, event=0.0),
            _graph(2.0),
            _graph(3.0, event=0.0),
        ]
    )
    cutpoints = fit_discrete_time_cutpoints(
        batch.time, batch.event, num_bins=4
    )
    output = model(batch)
    losses = dual_survival_objective(
        output,
        time=batch.time,
        event=batch.event,
        cutpoints=cutpoints,
        discrete_weight=0.2,
    )
    losses["total"].backward()

    gradients = [
        parameter.grad
        for parameter in model.edge_generator.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.any(gradient != 0) for gradient in gradients)


def test_discrete_time_loss_is_finite() -> None:
    logits = torch.tensor(
        [[-2.0, 2.0, -1.0], [-3.0, -2.0, -1.0]],
        requires_grad=True,
    )
    loss = discrete_time_nll(
        logits,
        time=torch.tensor([2.0, 5.0]),
        event=torch.tensor([1.0, 0.0]),
        cutpoints=torch.tensor([1.5, 3.5]),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
