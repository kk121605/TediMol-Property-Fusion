"""Method-specific property fusion modules. Learning notes live outside code."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from property_common import PropertyBatch, PropertyNormalizer

class PropertyTokenEncoder(nn.Module):
    """Encode K named scalar properties into K hidden tokens."""

    def __init__(
        self,
        property_names: Sequence[str],
        hidden_size: int,
        means: Sequence[float],
        stds: Sequence[float],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.property_names = tuple(property_names)
        if len(set(self.property_names)) != len(self.property_names):
            raise ValueError("property names must be unique and ordered")
        if len(self.property_names) != len(means):
            raise ValueError("property names, means, and stds must have the same length")
        self.hidden_size = int(hidden_size)
        self.normalizer = PropertyNormalizer(means, stds)
        self.value_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
                for _ in self.property_names
            ]
        )
        self.type_embedding = nn.Embedding(len(self.property_names), self.hidden_size)
        self.missing_embedding = nn.Embedding(2, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def num_properties(self) -> int:
        return len(self.property_names)

    def forward(self, batch: PropertyBatch) -> tuple[Tensor, Tensor]:
        z, observed = self.normalizer(batch)
        batch_size = z.shape[0]
        tokens = []
        for index, mlp in enumerate(self.value_mlps):
            value_token = mlp(z[:, index:index + 1]).unsqueeze(1)
            type_ids = torch.full(
                (batch_size, 1), index, dtype=torch.long, device=z.device
            )
            missing_ids = (~observed[:, index:index + 1]).long()
            token = (
                value_token
                + self.type_embedding(type_ids)
                + self.missing_embedding(missing_ids)
            )
            tokens.append(token)
        encoded = self.dropout(torch.cat(tokens, dim=1))
        return encoded, observed

    def pooled(self, batch: PropertyBatch) -> Tensor:
        tokens, _ = self(batch)
        # Missing values already use z=0 plus a learned missing embedding.
        # Pooling every token preserves that missingness signal.  Excluding
        # unobserved tokens here would silently make missing_embedding inert
        # in the A3 branch.
        return tokens.mean(dim=1)


class AdaLNZeroResidualAdapter(nn.Module):
    """A3 block adapter with exact A0 behavior at initialization.

    The pretrained Transformer block still runs normally.  This module is added
    *after* the block and contributes a gated conditional residual.  The final
    conditioning projection is zero-initialized, so the gate starts at zero and
    the adapter initially returns the unmodified A0 hidden states.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int | None = None):
        super().__init__()
        hidden_size = int(hidden_size)
        bottleneck_size = int(bottleneck_size or max(32, hidden_size // 4))
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.condition_to_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size),
        )
        # shift, scale, and gate are all zero at initialization.
        nn.init.zeros_(self.condition_to_modulation[-1].weight)
        nn.init.zeros_(self.condition_to_modulation[-1].bias)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.SiLU(),
            nn.Linear(bottleneck_size, hidden_size),
        )

    def forward(self, hidden_states: Tensor, condition: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or condition.ndim != 2:
            raise ValueError("hidden_states must be [B,L,H] and condition must be [B,H]")
        if hidden_states.shape[0] != condition.shape[0] or hidden_states.shape[2] != condition.shape[1]:
            raise ValueError("hidden states and condition are shape-incompatible")
        shift, scale, gate = self.condition_to_modulation(condition).chunk(3, dim=-1)
        normalized = self.norm(hidden_states)
        modulated = normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        residual = self.adapter(modulated)
        return hidden_states + torch.tanh(gate).unsqueeze(1) * residual


class AdaLNZeroAdapterStack(nn.Module):
    """One independent A3 adapter per pretrained Transformer block."""

    def __init__(self, num_layers: int, hidden_size: int, bottleneck_size: int | None = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaLNZeroResidualAdapter(hidden_size, bottleneck_size)
                for _ in range(int(num_layers))
            ]
        )

    def forward_layer(self, layer_index: int, hidden_states: Tensor, condition: Tensor) -> Tensor:
        return self.layers[layer_index](hidden_states, condition)


