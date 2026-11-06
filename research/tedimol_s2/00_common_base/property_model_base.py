"""Shared first-stage operations; method-specific fusion lives in sibling folders."""
from __future__ import annotations
import torch
from improved_diffusion.nn import timestep_embedding
from improved_diffusion.transformer_model2 import TransformerNetModel2 as BaseTediMol
from property_common import PropertyBatch


class PropertyTediMolBase(BaseTediMol):
    MODE = None
    ENCODER = None

    def __init__(self, *args, property_mode, property_names, property_means,
                 property_stds, property_bottleneck=None, **kwargs):
        super().__init__(*args, **kwargs)
        if property_mode != self.MODE:
            raise ValueError(f"This method implements {self.MODE}, not {property_mode}")
        self.property_mode = property_mode
        self.property_names = tuple(property_names)
        hidden_size = int(self.LayerNorm.normalized_shape[0])
        self.property_encoder = self.ENCODER(self.property_names, hidden_size,
                                             property_means, property_stds)
        self.property_adapters = self.make_adapters(hidden_size, property_bottleneck)

    def make_adapters(self, hidden_size, bottleneck):
        return None

    def property_metadata(self):
        result = {"mode": self.MODE, "property_names": list(self.property_names),
            "means": self.property_encoder.normalizer.means.detach().cpu().tolist(),
            "stds": self.property_encoder.normalizer.stds.detach().cpu().tolist(),
            "missing_policy": "z=0 plus learned observed/missing embedding"}
        if hasattr(self.property_encoder, "metadata"):
            result["numerical_token_encoding"] = self.property_encoder.metadata()
        return result

    def fuse_input(self, molecular_hidden, batch):
        raise NotImplementedError

    def conditioned_layers(self, hidden, desc_state, desc_mask, batch, prefix):
        raise NotImplementedError

    def forward(self, x, timesteps, desc_state, desc_mask, chemical_metric=None,
                property_observed=None, y=None):
        assert (y is not None) == (self.num_classes is not None)
        if chemical_metric is None or chemical_metric.ndim != 2 or chemical_metric.shape != (x.shape[0], len(self.property_names)):
            raise ValueError("chemical_metric must be [B,K] with the configured property order")
        if property_observed is None:
            property_observed = torch.ones_like(chemical_metric, dtype=torch.bool)
        batch = PropertyBatch(chemical_metric, property_observed)
        emb_t = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.mask:
            desc_state = torch.where(timesteps.reshape(-1, 1, 1) < 200, 0.0, desc_state)
            desc_mask = torch.where(timesteps.reshape(-1, 1) < 200, 1.0, desc_mask)
        molecular = self.input_up_proj(x)
        hidden, prefix = self.fuse_input(molecular, batch)
        length = hidden.shape[1]
        if length > self.position_ids.shape[1]:
            raise ValueError("Property prefix exceeds max_position_embeddings")
        positions = self.position_ids[:, :length]
        hidden = hidden + self.position_embeddings(positions) + emb_t.unsqueeze(1)
        hidden = self.dropout(self.LayerNorm(hidden))
        desc_state = self.dropout(self.LayerNorm(self.desc_down_proj(desc_state)))
        heads = self.input_transformers.layer[0].crossattention.self.num_attention_heads
        desc_mask = desc_mask[:, None, None, :].expand(-1, heads, -1, -1)
        hidden = self.conditioned_layers(hidden, desc_state, desc_mask, batch, prefix)
        if hidden.shape[1] != x.shape[1]:
            raise RuntimeError("Fusion changed molecular output length")
        return self.output_down_proj(hidden).type(x.dtype)
