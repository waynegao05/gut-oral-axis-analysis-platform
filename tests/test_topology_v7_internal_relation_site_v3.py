from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from experiments.topology_v7_internal_relation_site_v3.features import (
    PROHIBITED_FEATURE_TOKENS,
    attach_site_features,
    build_site_feature_table,
    fit_site_standardizer,
)
from experiments.topology_v7_internal_relation_site_v3.model import (
    InternalRelationSiteModel,
)
from research.model_v2 import DeepStructureAwareGATCoxModelV2


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments/topology_v7_internal_relation_site_v3/"
    "experiment_plan.json"
)
NUM_NODE_TYPES = 5
SITE_DIM = 14


def _edges() -> torch.Tensor:
    source: list[int] = []
    target: list[int] = []
    for left in range(NUM_NODE_TYPES):
        for right in range(left + 1, NUM_NODE_TYPES):
            source.extend([left, right])
            target.extend([right, left])
    return torch.tensor([source, target], dtype=torch.long)


def _graph(offset: float = 0.0) -> Data:
    return Data(
        x=torch.stack(
            [
                torch.linspace(0.1, 0.5, NUM_NODE_TYPES) + offset,
                torch.linspace(0.3, 0.7, NUM_NODE_TYPES),
            ],
            dim=1,
        ),
        edge_index=_edges(),
        edge_attr=torch.linspace(0.05, 0.95, 20).view(-1, 1),
        node_type=torch.arange(NUM_NODE_TYPES),
        node_struct=torch.rand(NUM_NODE_TYPES, 5),
        node_targets=torch.rand(NUM_NODE_TYPES, 4),
        graph_targets=torch.rand(1),
        graph_cluster_targets=torch.rand(1),
        clinical=torch.tensor([0.2, -0.1, 0.3, 0.5]),
        metabolites=torch.tensor([0.1, 0.4, 0.7]),
        site_features=torch.linspace(-0.8, 0.8, SITE_DIM) + offset,
        time=torch.tensor(42.0),
        event=torch.tensor(1.0),
        generation_group_id=torch.tensor(0),
        sample_id=f"sample_{offset}",
    )


def _model(
    *,
    edge_mode: str = "site_context",
    use_site_residual: bool = True,
) -> InternalRelationSiteModel:
    torch.manual_seed(17)
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
    return InternalRelationSiteModel(
        base,
        node_feature_dim=2,
        clinical_dim=4,
        metabolite_dim=3,
        site_feature_dim=SITE_DIM,
        num_node_types=NUM_NODE_TYPES,
        edge_mode=edge_mode,
        use_site_residual=use_site_residual,
        edge_hidden_dim=12,
        site_hidden_dim=12,
    )


def test_v3_plan_locks_nested_logo_and_one_time_audit() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["development_generation_seed"] == 20261010
    assert plan["audit_generation_seed"] == 20261011
    assert plan["development_model_seeds"] == [42]
    assert plan["audit_model_seeds"] == [7, 21, 42, 123, 2026]
    assert (
        plan["inference_contract"]["edge_weights_computed_inside_forward"]
        is True
    )
    assert (
        plan["audit_policy"]["audit_generated_only_after_development_gate"]
        is True
    )
    assert plan["development_gate"]["minimum_macro_c_index_gain"] == 0.0015
    assert plan["development_gate"]["minimum_integrated_auc_gain"] == 0.003
    assert any(
        candidate.get("aft_weight") == 0.05
        for candidate in plan["candidates"]
    )


def test_site_features_are_outcome_free_and_training_standardized(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for sample_index, sample_id in enumerate(("a", "b", "c")):
        for taxon_index, taxon in enumerate(("A", "B")):
            rows.append(
                {
                    "sample_id": sample_id,
                    "taxon": taxon,
                    "saliva_relative_abundance": (
                        0.1 + 0.03 * sample_index + 0.04 * taxon_index
                    ),
                    "stool_relative_abundance": (
                        0.2 + 0.02 * sample_index + 0.03 * taxon_index
                    ),
                    "time": 10 + sample_index,
                    "event": sample_index % 2,
                }
            )
    path = tmp_path / "oral_gut.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    table = build_site_feature_table(path)
    lowered = " ".join(table.feature_columns).lower()
    assert all(token not in lowered for token in PROHIBITED_FEATURE_TOKENS)
    assert table.frame.shape == (3, 16)

    standardizer = fit_site_standardizer(table, ["a", "b"])
    items = [
        Data(sample_id=sample_id)
        for sample_id in ("a", "b", "c")
    ]
    attach_site_features([items[:2], items[2:]], table, standardizer)
    train = torch.stack([item.site_features for item in items[:2]])
    assert torch.allclose(train.mean(dim=0), torch.zeros(16), atol=1e-5)
    assert np.isfinite(items[2].site_features.numpy()).all()


def test_internal_model_ignores_csv_edges_cached_structure_and_outcomes() -> None:
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
    changed.time = changed.time * 100.0
    changed.event = 1.0 - changed.event
    changed.generation_group_id = changed.generation_group_id + 20

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


def test_site_context_changes_internal_weights_and_risk() -> None:
    model = _model().eval()
    with torch.no_grad():
        model.edge_generator.pair_network[-1].weight.fill_(0.05)
        model.site_residual_head[-1].weight.fill_(0.05)
    first_batch = Batch.from_data_list([_graph(0.0), _graph(0.1)])
    second_batch = copy.deepcopy(first_batch)
    second_batch.site_features = second_batch.site_features + 1.5

    with torch.no_grad():
        first = model(first_batch)
        second = model(second_batch)

    assert first["pair_edge_weights"].shape == (2, 10)
    assert torch.all(first["pair_edge_weights"] > np.exp(-0.5))
    assert torch.all(first["pair_edge_weights"] < np.exp(0.5))
    assert not torch.allclose(
        first["pair_edge_weights"],
        second["pair_edge_weights"],
    )
    assert not torch.allclose(first["risk"], second["risk"])
    assert torch.isfinite(first["aft_location"]).all()
    assert torch.isfinite(first["aft_log_scale"])


def test_equal_internal_edges_are_exactly_one() -> None:
    model = _model(
        edge_mode="equal",
        use_site_residual=False,
    ).eval()
    batch = Batch.from_data_list([_graph(0.0), _graph(0.1)])

    with torch.no_grad():
        output = model(batch)

    assert torch.equal(
        output["pair_edge_weights"],
        torch.ones((2, 10)),
    )
