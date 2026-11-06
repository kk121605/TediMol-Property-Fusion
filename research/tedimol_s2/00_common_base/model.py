"""Small constructor adapter around the delivered TransformerNetModel2."""
from transformers import BertConfig
from improved_diffusion.transformer_model2 import TransformerNetModel2 as SeniorModel

DEFAULT_MODEL = dict(in_channels=32, model_channels=128, hidden_size=1024,
    num_attention_heads=16, num_hidden_layers=12, intermediate_size=3072,
    dropout=0.1, max_position_embeddings=512)


def TransformerNetModel2(vocab_size, **kwargs):
    settings=DEFAULT_MODEL|kwargs
    config=BertConfig(intermediate_size=settings.pop('intermediate_size'),
        max_position_embeddings=settings.pop('max_position_embeddings'))
    return SeniorModel(vocab_size=vocab_size, config=config, training_mode='e2e', **settings)


class LossModelAdapter:
    """Supply the .module interface expected by original GaussianDiffusion.

    No DDP communication or numerical transformation is needed for a single GPU.
    """
    def __init__(self,model):
        self.module=model

    def __call__(self,*args,**kwargs):
        return self.module(*args,**kwargs)
