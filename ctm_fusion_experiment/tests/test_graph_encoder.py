from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from ctm_fusion_experiment.models.graph_encoder import GraphOnlyGATCoxEncoder


def _graph(offset: float) -> Data:
    return Data(
        x=torch.tensor(
            [
                [0.1 + offset, 0.2],
                [0.3 + offset, 0.4],
                [0.5 + offset, 0.6],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.tensor([[0.5], [0.5], [0.7], [0.7]], dtype=torch.float32),
    )


def test_graph_only_encoder_emits_one_embedding_and_risk_per_graph() -> None:
    model = GraphOnlyGATCoxEncoder(
        node_feature_dim=2,
        hidden_dim=8,
        heads=2,
        edge_hidden_dim=4,
        embedding_dim=10,
        dropout=0.0,
    )
    batch = Batch.from_data_list([_graph(0.0), _graph(0.1)])

    output = model(batch)

    assert output["graph_embedding"].shape == (2, 10)
    assert output["risk"].shape == (2,)
