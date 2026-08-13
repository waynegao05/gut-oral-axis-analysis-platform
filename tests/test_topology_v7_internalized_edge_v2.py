from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from experiments.topology_v7_internalized_edge_v2.model import (
    InternalizedEdgeDropInModel,
    internal_complete_graph_structure,
    lognormal_aft_nll,
)
from experiments.topology_v7_internalized_edge_v2.runner import (
    canonical_precomputed_edge_target,
)
from research.model_v2 import DeepStructureAwareGATCoxModelV2


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments/topology_v7_internalized_edge_v2/experiment_plan.json"
)
NUM_NODE_TYPES = 5


def _edges() -> torch.Tensor:
    source: list[int] = []
    target: list[int] = []
    for left in range(NUM_NODE_TYPES):
        for right in range(left + 1, NUM_NODE_TYPES):
            source.extend([left, right])
            target.extend([right, left])
    return torch.tensor([source, target], dtype=torch.long)


def _graph(offset: float = 0.0) -> Data:
    edge_index = _edges()
    undirected = torch.linspace(0.08, 0.42, 10)
    edge_attr = torch.repeat_interleave(undirected, 2).view(-1, 1)
    return Data(
        x=torch.stack(
            [
                torch.linspace(0.1, 0.5, NUM_NODE_TYPES) + offset,
                torch.linspace(0.3, 0.7, NUM_NODE_TYPES),
            ],
            dim=1,
        ),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=torch.arange(NUM_NODE_TYPES),
        node_struct=torch.rand(NUM_NODE_TYPES, 5),
        node_targets=torch.rand(NUM_NODE_TYPES, 4),
        graph_targets=torch.rand(1),
        graph_cluster_targets=torch.rand(1),
        clinical=torch.tensor([0.2, -0.1, 0.3, 0.5]),
        metabolites=torch.tensor([0.1, 0.4, 0.7]),
        time=torch.tensor(42.0),
        event=torch.tensor(1.0),
        generation_group_id=torch.tensor(0),
        edge_supervision_target=undirected,
        sample_id=f"sample_{offset}",
    )


def _model() -> InternalizedEdgeDropInModel:
    torch.manual_seed(13)
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
    model = InternalizedEdgeDropInModel(
        base,
        node_feature_dim=2,
        clinical_dim=4,
        metabolite_dim=3,
        num_node_types=NUM_NODE_TYPES,
        edge_mode="node_context",
        edge_hidden_dim=12,
        node_identity_dim=4,
    )
    model.initialize_aft_location(4.0)
    return model


def test_v2_uses_fresh_development_and_audit_seeds() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["future_cohorts"]["development_generation_seed"] == 20261008
    assert plan["future_cohorts"]["audit_generation_seed"] == 20261009
    assert plan["inference_contract"]["edge_weight_csv_read_by_model"] is False
    assert (
        plan["inference_contract"][
            "edge_supervision_target_available_at_inference"
        ]
        is False
    )


def test_canonical_training_target_is_ordered_and_symmetric() -> None:
    target = canonical_precomputed_edge_target(
        _graph(), num_node_types=NUM_NODE_TYPES
    )

    assert target.shape == (10,)
    assert torch.allclose(target, torch.linspace(0.08, 0.42, 10))


def test_drop_in_forward_ignores_csv_edges_cached_structure_and_outcomes() -> None:
    model = _model().eval()
    original = Batch.from_data_list([_graph(0.0), _graph(0.1)])
    changed = copy.deepcopy(original)
    changed.edge_attr = torch.rand_like(changed.edge_attr) * 100.0
    changed.node_struct = torch.rand_like(changed.node_struct) * 100.0
    changed.node_targets = torch.rand_like(changed.node_targets) * 100.0
    changed.graph_targets = torch.rand_like(changed.graph_targets) * 100.0
    changed.graph_cluster_targets = (
        torch.rand_like(changed.graph_cluster_targets) * 100.0
    )
    changed.edge_supervision_target = (
        torch.rand_like(changed.edge_supervision_target) * 100.0
    )
    changed.time = changed.time * 100.0
    changed.event = 1.0 - changed.event
    changed.generation_group_id = changed.generation_group_id + 10

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


def test_internal_structure_has_expected_complete_graph_shapes() -> None:
    model = _model().eval()
    batch = Batch.from_data_list([_graph(0.0), _graph(0.1)])
    with torch.no_grad():
        state = model.edge_generator(batch)
        structure = internal_complete_graph_structure(
            batch, state, num_node_types=NUM_NODE_TYPES
        )

    node_struct, node_targets, graph_targets, cluster_targets = structure
    assert node_struct.shape == (10, 5)
    assert node_targets.shape == (10, 4)
    assert graph_targets.shape == (2, 1)
    assert cluster_targets.shape == (2, 1)
    assert torch.all(node_struct[:, 0] == 1.0)
    assert torch.all(node_struct[:, 2] == 1.0)


def test_lognormal_aft_loss_is_finite_and_differentiable() -> None:
    location = torch.tensor([3.5, 4.2], requires_grad=True)
    loss = lognormal_aft_nll(
        location,
        torch.tensor(-0.3, requires_grad=True),
        torch.tensor([36.0, 72.0]),
        torch.tensor([1.0, 0.0]),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(location.grad).all()
