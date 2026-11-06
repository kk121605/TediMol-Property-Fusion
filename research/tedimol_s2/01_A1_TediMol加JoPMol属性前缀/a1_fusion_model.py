"""A1: scalar property prefix transfer, not the full multimodal JoPMol."""
import torch
from property_model_base import PropertyTediMolBase
from a1_conditioning import PropertyTokenEncoder


class TediMolJoPMolPrefix(PropertyTediMolBase):
    MODE = "a1_prefix"
    ENCODER = PropertyTokenEncoder

    def fuse_input(self, molecular_hidden, batch):
        tokens, observed = self.property_encoder(batch)
        return torch.cat([tokens, molecular_hidden], dim=1), tokens.shape[1]

    def conditioned_layers(self, hidden, desc_state, desc_mask, batch, prefix):
        output = self.input_transformers(hidden, encoder_hidden_states=desc_state,
                                         encoder_attention_mask=desc_mask)
        return output.last_hidden_state[:, prefix:, :]
