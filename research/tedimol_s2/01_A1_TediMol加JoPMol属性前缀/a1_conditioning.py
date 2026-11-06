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


