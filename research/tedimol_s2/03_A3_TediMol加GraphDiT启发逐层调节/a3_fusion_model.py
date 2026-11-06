"""A3: per-layer zero-gated residual adapters inspired by Graph DiT AdaLN."""
from property_model_base import PropertyTediMolBase
from a3_conditioning import PropertyTokenEncoder, AdaLNZeroAdapterStack


class TediMolGraphDiTAdapter(PropertyTediMolBase):
    MODE = "a3_adaln_zero"
    ENCODER = PropertyTokenEncoder

    def make_adapters(self, hidden_size, bottleneck):
        return AdaLNZeroAdapterStack(len(self.input_transformers.layer), hidden_size, bottleneck)

    def fuse_input(self, molecular_hidden, batch):
        return molecular_hidden, 0

    def conditioned_layers(self, hidden, desc_state, desc_mask, batch, prefix):
        condition = self.property_encoder.pooled(batch)
        for index, layer in enumerate(self.input_transformers.layer):
            hidden = layer(hidden, attention_mask=None, head_mask=None,
                encoder_hidden_states=desc_state, encoder_attention_mask=desc_mask,
                past_key_value=None, output_attentions=False)[0]
            hidden = self.property_adapters.forward_layer(index, hidden, condition)
        return hidden
