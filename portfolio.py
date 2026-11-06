"""Small offline constructors around the recovered research implementations.

This module does not load research data, encoder weights or diffusion checkpoints.
The underlying models and losses are the supplied code at reduced dimensions.
"""

from pathlib import Path
import sys

import torch

COMMON = Path(__file__).resolve().parent / "research/tedimol_s2/00_common_base"
sys.path.insert(0, str(COMMON))

from model import TransformerNetModel2
from property_runtime import build_property_model, initialize_from_a0

MODES = ("a0c", "a1_prefix", "a2_rt_numeric", "a3_adaln_zero")
NAMES = ("logP", "SA", "TPSA")
TINY_CONFIG = dict(
    in_channels=8,
    model_channels=16,
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=1,
    intermediate_size=64,
    max_position_embeddings=64,
    dropout=0.0,
)
# Synthetic fixture statistics, deliberately unrelated to the research dataset.
DEMO_MEANS = (2.0, 4.0, 60.0)
DEMO_STDS = (1.5, 1.0, 30.0)


def build_tiny(mode, parent=None):
    """Build a real reduced-dimension branch with a synthetic property protocol.

    If parent is provided, use its state to initialise shared parameters exactly.
    This in-memory fixture is not a historical checkpoint loader.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown branch: {mode}")
    if mode == "a0c":
        model = TransformerNetModel2(323, **TINY_CONFIG)
        if parent is not None:
            model.load_state_dict(parent.state_dict(), strict=True)
    else:
        model = build_property_model(
            323,
            TINY_CONFIG,
            property_mode=mode,
            property_names=NAMES,
            means=DEMO_MEANS,
            stds=DEMO_STDS,
            property_bottleneck=16,
        )
        if parent is not None:
            initialize_from_a0(
                model, {"model": parent.state_dict()}, init_weights="raw"
            )
    return model


def forward(model, mode, x, times, text, text_mask, properties, observed=None):
    """Match the original A0c/property branch call signatures."""
    if mode not in MODES:
        raise ValueError(f"Unknown branch: {mode}")
    kwargs = (
        {}
        if mode == "a0c"
        else {"chemical_metric": properties, "property_observed": observed}
    )
    return model(x, times, text, text_mask, **kwargs)


def main():
    """Run a synthetic denoising update for each recovered branch."""
    import json
    from comparison_runtime import Bridge, update
    from improved_diffusion.resample import UniformSampler
    from runtime import make_diffusion, seed_all

    torch.set_num_threads(1)
    seed_all(42)
    base = build_tiny("a0c")
    molecule_ids = torch.randint(0, 323, (2, 12))
    text = torch.randn(2, 10, 768)
    mask = torch.ones(2, 10)
    values = torch.tensor([[1.2, 3.0, 40.0], [2.4, 5.0, 80.0]])
    batch = (
        molecule_ids,
        text,
        mask,
        values,
        torch.ones_like(values, dtype=torch.bool),
    )
    config = dict(
        lr=1e-5,
        lr_horizon_steps=300000,
        microbatch=1,
        seed=19991009,
        grad_clip=2.4,
        ema_rate=0.9999,
    )
    result = {}
    for mode in MODES:
        seed_all(42)
        model = build_tiny(mode, parent=base).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
        ema = {name: p.detach().clone() for name, p in model.named_parameters()}
        diffusion = make_diffusion()
        record = update(
            model,
            Bridge(model, mode),
            optimizer,
            ema,
            diffusion,
            UniformSampler(diffusion),
            batch,
            0,
            config,
            "cpu",
        )
        if not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError(f"Nonfinite gradients in {mode}")
        result[mode] = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            loss=round(record["loss"], 6),
            finite_gradients=True,
        )
    print(
        json.dumps(
            {
                "scope": "real reduced-dimension code; synthetic inputs; no quality claim",
                "branches": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
