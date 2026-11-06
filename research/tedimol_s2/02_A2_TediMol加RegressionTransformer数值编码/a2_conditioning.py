"""Method-specific property fusion modules. Learning notes live outside code."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from property_common import PropertyBatch, PropertyNormalizer

class RTNumericalTokenEncoder(nn.Module):
    """A2 transfer of RT digit/place tokens, not the complete XLNet RT.

    Each z-scored property uses a sign and seven decimal-place tokens:
    orders 2,1,0,-1,-2,-3,-4. Thus K=3 adds 24 condition tokens.
    The molecule vocabulary stays at 323 IDs; condition IDs are separate.
    Values are quantized to four decimals and are not input-differentiable.
    Frozen NE follows official get_float_encoding, not the paper's indexing:
    [v,-v,v/3,-v/3,...] / (vmax/10), v=digit*10**order.
    Signs are separate learned symbols, as in RT tokenization.
    """

    orders = (2, 1, 0, -1, -2, -3, -4)
    precision = 4
    numerical_vmax = 10.0

    def __init__(self, property_names, hidden_size, means, stds):
        super().__init__()
        self.property_names = tuple(property_names)
        self.hidden_size = int(hidden_size)
        if self.hidden_size % 2:
            raise ValueError("RT float encoding requires an even hidden size")
        if len(set(self.property_names)) != len(self.property_names) or len(means) != len(self.property_names):
            raise ValueError("property names must be unique and match statistics")
        self.normalizer = PropertyNormalizer(means, stds)
        # IDs 0/1 are +/-; IDs 2..71 are digit tokens with explicit place.
        self.symbol_embedding = nn.Embedding(2 + 10 * len(self.orders), self.hidden_size)
        self.type_embedding = nn.Embedding(len(self.property_names), self.hidden_size)
        self.missing_embedding = nn.Embedding(2, self.hidden_size)
        ne = torch.zeros(2 + 10 * len(self.orders), self.hidden_size)
        for slot, order in enumerate(self.orders):
            for digit in range(10):
                value = digit * 10.0 ** order
                for index in range(0, self.hidden_size, 2):
                    ne[2 + slot * 10 + digit, index] = value / (index + 1)
                    ne[2 + slot * 10 + digit, index + 1] = -value / (index + 1)
        ne /= self.numerical_vmax / 10.0
        self.register_buffer("fixed_numerical_encoding", ne)

    @property
    def num_properties(self):
        return len(self.property_names)

    @property
    def tokens_per_property(self):
        return 1 + len(self.orders)

    def tokenize_z(self, z):
        if not torch.isfinite(z).all():
            raise ValueError("z must be finite")
        # double protects decimal-place extraction from float32 integer error.
        units = torch.round(z.double() * 10 ** self.precision)
        if torch.any(units.abs() >= 10 ** len(self.orders)):
            raise ValueError("A2 value outside signed fixed format +/-999.9999")
        sign = (units < 0).long().unsqueeze(-1)
        unsigned = units.abs().long()
        tokens = [sign]
        for slot, order in enumerate(self.orders):
            digit = (unsigned // 10 ** (order + self.precision)) % 10
            tokens.append((2 + slot * 10 + digit).unsqueeze(-1))
        return torch.cat(tokens, dim=-1)

    def decode_z(self, token_ids):
        if token_ids.shape[-1] != self.tokens_per_property:
            raise ValueError("bad A2 token length")
        if torch.any((token_ids[..., 0] < 0) | (token_ids[..., 0] > 1)):
            raise ValueError("bad sign token")
        result = torch.zeros(token_ids.shape[:-1], device=token_ids.device, dtype=torch.float64)
        for slot, order in enumerate(self.orders):
            digit = token_ids[..., slot + 1] - (2 + slot * 10)
            if torch.any((digit < 0) | (digit > 9)):
                raise ValueError("digit token has wrong decimal place")
            result += digit.double() * 10.0 ** order
        return torch.where(token_ids[..., 0].bool(), -result, result)

    def forward(self, batch):
        z, observed = self.normalizer(batch)
        ids = self.tokenize_z(z)
        symbols = self.symbol_embedding(ids)
        numerical = self.fixed_numerical_encoding[ids].to(symbols)
        type_ids = torch.arange(self.num_properties, device=z.device)
        types = self.type_embedding(type_ids)[None, :, None, :]
        missing = self.missing_embedding((~observed).long()).unsqueeze(2)
        tokens = symbols + numerical + types + missing
        return tokens.flatten(1, 2), observed

    def metadata(self):
        return {
            "kind": "RT-inspired digit/place prefix; not complete RT",
            "orders": list(self.orders), "precision": self.precision,
            "tokens_per_property": self.tokens_per_property,
            "prefix_length": self.num_properties * self.tokens_per_property,
            "numerical_vmax": self.numerical_vmax,
            "sign": "separate learned sign token; zero has positive sign",
            "molecular_vocabulary_changed": False,
            "quantized_input_not_differentiable": True,
        }


