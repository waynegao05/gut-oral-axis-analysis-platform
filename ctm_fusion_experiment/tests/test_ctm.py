from __future__ import annotations

import torch

from ctm_fusion_experiment.models.ctm import CTM, NeuronLevelModels, Synchronization
from ctm_fusion_experiment.models.ctm_fusion import CTMFusionModel


def test_neuron_level_models_process_private_histories_in_parallel() -> None:
    nlm = NeuronLevelModels(d_model=3, memory_length=4, hidden_dim=2, dropout=0.0)
    histories = torch.randn(5, 3, 4)

    output = nlm(histories)

    assert output.shape == (5, 3)
    assert nlm.input_weight.shape == (4, 4, 3)
    assert nlm.output_weight.shape == (2, 2, 3)


def test_synchronization_updates_recursively() -> None:
    synchronization = Synchronization(d_model=4, n_pairs=2, n_self_pairs=0)
    synchronization.left_indices.copy_(torch.tensor([0, 1]))
    synchronization.right_indices.copy_(torch.tensor([2, 3]))
    synchronization.decay_params.data.zero_()
    state = torch.tensor([[2.0, 3.0, 5.0, 7.0]])

    first, alpha, beta = synchronization(state)
    second, _, _ = synchronization(state, alpha=alpha, beta=beta)

    assert first.tolist() == [[10.0, 21.0]]
    assert torch.allclose(second, torch.tensor([[20.0, 42.0]]) / torch.sqrt(torch.tensor(2.0)))


def test_ctm_emits_representation_and_attention_for_each_tick() -> None:
    model = CTM(
        d_model=12,
        d_input=8,
        iterations=3,
        memory_length=3,
        nlm_hidden_dim=4,
        n_heads=2,
        n_synch_action=6,
        n_synch_out=7,
        n_self_pairs=2,
        synapse_depth=2,
        dropout=0.0,
    )
    tokens = torch.randn(4, 3, 8)

    output = model(tokens, track_attention=True)

    assert output.representations.shape == (4, 3, 7)
    assert output.attention_weights is not None
    assert output.attention_weights.shape == (4, 3, 2, 3)


def test_ctm_fusion_emits_risk_for_each_tick() -> None:
    model = CTMFusionModel(
        graph_dim=9,
        clinical_dim=4,
        metabolite_dim=3,
        d_input=8,
        d_model=12,
        iterations=3,
        memory_length=3,
        nlm_hidden_dim=4,
        n_heads=2,
        n_synch_action=6,
        n_synch_out=7,
        n_self_pairs=2,
        synapse_depth=2,
        dropout=0.0,
    )

    output = model(
        graph_features=torch.randn(5, 9),
        clinical_features=torch.randn(5, 4),
        metabolite_features=torch.randn(5, 3),
        track_attention=True,
    )

    assert output["risk_per_tick"].shape == (5, 3)
    assert output["attention_weights"].shape == (5, 3, 2, 3)
