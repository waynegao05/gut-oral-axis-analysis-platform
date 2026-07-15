from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynapseModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_input: int,
        depth: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive.")
        layers: list[nn.Module] = []
        input_dim = d_model + d_input
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(input_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = d_model
        self.network = nn.Sequential(*layers)

    def forward(self, activated_state: torch.Tensor, attention_output: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([activated_state, attention_output], dim=-1))


class NeuronLevelModels(nn.Module):
    def __init__(
        self,
        d_model: int,
        memory_length: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_weight = nn.Parameter(torch.empty(memory_length, hidden_dim * 2, d_model))
        self.input_bias = nn.Parameter(torch.zeros(1, d_model, hidden_dim * 2))
        self.output_weight = nn.Parameter(torch.empty(hidden_dim, 2, d_model))
        self.output_bias = nn.Parameter(torch.zeros(1, d_model, 2))
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.xavier_uniform_(self.output_weight)

    def forward(self, histories: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("bdm,mhd->bdh", self.dropout(histories), self.input_weight)
        hidden = F.glu(hidden + self.input_bias, dim=-1)
        output = torch.einsum("bdh,hod->bdo", hidden, self.output_weight)
        return F.glu(output + self.output_bias, dim=-1).squeeze(-1)


class Synchronization(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_pairs: int,
        n_self_pairs: int = 0,
    ) -> None:
        super().__init__()
        if n_pairs <= 0:
            raise ValueError("n_pairs must be positive.")
        if n_self_pairs < 0 or n_self_pairs > n_pairs:
            raise ValueError("n_self_pairs must be between zero and n_pairs.")

        left = torch.randint(0, d_model, (n_pairs,), dtype=torch.long)
        right = torch.randint(0, d_model, (n_pairs,), dtype=torch.long)
        right[:n_self_pairs] = left[:n_self_pairs]
        self.register_buffer("left_indices", left)
        self.register_buffer("right_indices", right)
        self.decay_params = nn.Parameter(torch.zeros(n_pairs))

    def forward(
        self,
        activated_state: torch.Tensor,
        alpha: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        products = activated_state[:, self.left_indices] * activated_state[:, self.right_indices]
        if alpha is None or beta is None:
            alpha = products
            beta = torch.ones_like(products)
        else:
            decay = torch.exp(-torch.clamp(self.decay_params, min=0.0, max=15.0)).unsqueeze(0)
            alpha = decay * alpha + products
            beta = decay * beta + 1.0
        return alpha / torch.sqrt(beta), alpha, beta


@dataclass(frozen=True)
class CTMOutput:
    representations: torch.Tensor
    attention_weights: torch.Tensor | None


class CTM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_input: int,
        iterations: int,
        memory_length: int,
        nlm_hidden_dim: int,
        n_heads: int,
        n_synch_action: int,
        n_synch_out: int,
        n_self_pairs: int,
        synapse_depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.start_activated_state = nn.Parameter(torch.zeros(d_model))
        self.start_trace = nn.Parameter(torch.zeros(d_model, memory_length))
        nn.init.uniform_(self.start_activated_state, -0.05, 0.05)
        nn.init.uniform_(self.start_trace, -0.05, 0.05)
        self.action_sync = Synchronization(d_model, n_synch_action, n_self_pairs)
        self.output_sync = Synchronization(d_model, n_synch_out, n_self_pairs)
        self.query_projection = nn.Linear(n_synch_action, d_input)
        self.attention = nn.MultiheadAttention(d_input, n_heads, dropout=dropout, batch_first=True)
        self.synapse_model = SynapseModel(d_model, d_input, depth=synapse_depth, dropout=dropout)
        self.neuron_level_models = NeuronLevelModels(
            d_model=d_model,
            memory_length=memory_length,
            hidden_dim=nlm_hidden_dim,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor, track_attention: bool = False) -> CTMOutput:
        batch_size = tokens.size(0)
        state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1)
        trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1)
        action_alpha = action_beta = None
        output_alpha = output_beta = None
        representations = []
        attention_by_tick = []

        for _ in range(self.iterations):
            action, action_alpha, action_beta = self.action_sync(state, action_alpha, action_beta)
            query = self.query_projection(action).unsqueeze(1)
            attended, weights = self.attention(
                query,
                tokens,
                tokens,
                need_weights=track_attention,
                average_attn_weights=False,
            )
            pre_activation = self.synapse_model(state, attended.squeeze(1))
            trace = torch.cat([trace[:, :, 1:], pre_activation.unsqueeze(-1)], dim=-1)
            state = self.neuron_level_models(trace)
            output, output_alpha, output_beta = self.output_sync(state, output_alpha, output_beta)
            representations.append(output)
            if track_attention:
                attention_by_tick.append(weights.squeeze(2))

        stacked_representations = torch.stack(representations, dim=1)
        stacked_attention = torch.stack(attention_by_tick, dim=1) if track_attention else None
        return CTMOutput(
            representations=stacked_representations,
            attention_weights=stacked_attention,
        )
