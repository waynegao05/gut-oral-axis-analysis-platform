from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch_geometric.data import Batch, Data

from research.losses import cox_ph_loss
from research.model_v2 import (
    DeepStructureAwareGATCoxModelV2,
    _compute_structure_targets,
    compute_single_graph_structure,
)
from research.train_v2 import build_run_provenance


ROOT = Path(__file__).resolve().parents[1]


def _graph(node_type: torch.Tensor) -> Data:
    x = torch.tensor([[0.2, 0.4], [0.8, 0.6]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[0.5], [0.5]], dtype=torch.float32)
    node_struct, node_targets, graph_targets, graph_cluster_targets = compute_single_graph_structure(
        x,
        edge_index,
        edge_attr,
    )
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        node_struct=node_struct,
        node_targets=node_targets,
        graph_targets=graph_targets,
        graph_cluster_targets=graph_cluster_targets,
        clinical=torch.tensor([0.1, -0.2], dtype=torch.float32),
        metabolites=torch.tensor([0.3, -0.4], dtype=torch.float32),
        time=torch.tensor(12.0),
        event=torch.tensor(1.0),
    )


def test_breslow_cox_loss_is_invariant_within_tied_event_times() -> None:
    risk = torch.tensor([1.0, 0.2, -0.3, 0.4], dtype=torch.float32, requires_grad=True)
    time = torch.tensor([1.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    event = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    permutation = torch.tensor([1, 0, 2, 3])

    original = cox_ph_loss(risk, time, event, ties_method="breslow")
    permuted = cox_ph_loss(
        risk[permutation],
        time[permutation],
        event[permutation],
        ties_method="breslow",
    )

    assert original.item() == pytest.approx(permuted.item())
    original.backward()
    assert torch.isfinite(risk.grad).all()


def test_precomputed_structure_targets_are_reused_after_batching() -> None:
    data = _graph(torch.tensor([0, 1], dtype=torch.long))
    batch = Batch.from_data_list([data])

    node_struct, node_targets, graph_targets, graph_cluster_targets = _compute_structure_targets(batch)

    torch.testing.assert_close(node_struct, data.node_struct)
    torch.testing.assert_close(node_targets, data.node_targets)
    torch.testing.assert_close(graph_targets, data.graph_targets)
    torch.testing.assert_close(graph_cluster_targets, data.graph_cluster_targets)


def test_node_identity_embedding_changes_graph_risk_when_labels_are_swapped() -> None:
    torch.manual_seed(42)
    model = DeepStructureAwareGATCoxModelV2(
        node_feature_dim=2,
        clinical_dim=2,
        metabolite_dim=2,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
        edge_hidden_dim=8,
        num_layers=2,
        num_node_types=2,
        node_identity_dim=4,
        identity_readout_dim=8,
        graph_projection_dim=8,
        tabular_projection_dim=8,
    )
    model.eval()

    original = Batch.from_data_list([_graph(torch.tensor([0, 1], dtype=torch.long))])
    swapped = Batch.from_data_list([_graph(torch.tensor([1, 0], dtype=torch.long))])
    with torch.no_grad():
        original_risk = model(original)["risk"]
        swapped_risk = model(swapped)["risk"]

    assert not torch.allclose(original_risk, swapped_risk, atol=1e-7, rtol=1e-7)


def test_node_identity_configuration_must_be_complete() -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        DeepStructureAwareGATCoxModelV2(
            node_feature_dim=2,
            clinical_dim=2,
            metabolite_dim=2,
            num_node_types=2,
            node_identity_dim=0,
        )


def test_training_provenance_verifies_v7_manifest_hashes() -> None:
    config_path = ROOT / "config/research/research_config_v7_gnn_fullrisk.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    provenance = build_run_provenance(config, config_path, split_seed=42)

    assert provenance["dataset"]["generator_version"] == "topology_v7_hybrid_generator_v2"
    assert provenance["dataset"]["declared_output_hashes_verified"] is True
    assert set(provenance["dataset"]["inputs"]) == {
        "graph_csv",
        "clinical_csv",
        "metabolite_csv",
        "label_csv",
        "provenance_csv",
    }
