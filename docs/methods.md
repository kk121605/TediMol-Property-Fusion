# Four-branch method comparison

## Shared task and interface

The underlying TediMol pipeline encodes a text description with BioLinkBERT and denoises continuous embeddings of a SELFIES molecular sequence. The supplied backbone uses BERT encoder blocks as a diffusion denoiser, with text cross-attention. The first stage provides the initial text-only model; the second stage adds numerical conditions to selected branches.

The fixed property order is `logP, SA, TPSA`. SA here is the synthetic accessibility score used in the supplied property protocol. TPSA is topological polar surface area. Larger values are not inherently preferable for all three properties; evaluation must compare outputs with specified targets rather than interpreting any change as an improvement.

The original SELFIES vocabulary has 323 IDs, including special IDs. Property tokens use separate embedding modules and do not modify that molecular vocabulary. The historical pipeline treats the molecular output as a fixed 258-position grid, including special/padding positions. Its tokenizer and decoding conventions are retained.

```python
import torch
from portfolio import build_tiny, forward

model = build_tiny("a1_prefix").eval()
prediction = forward(
    model, "a1_prefix",
    torch.randn(2, 12, 8),          # noisy molecular embeddings
    torch.tensor([100., 900.]),    # diffusion timesteps
    torch.randn(2, 10, 768),       # synthetic text states
    torch.ones(2, 10),             # recovered text-mask convention
    torch.tensor([[1., 3., 40.], [2., 5., 80.]])
)
```

This example is a software fixture. `build_tiny` uses illustrative normalisation statistics, not the research data statistics. The historical full model has 1,024 hidden units and 12 blocks; the CPU fixture has 32 hidden units and one block.

## A0c text-only control

A0c continues the first-stage text-conditioned model without numerical input. Its purpose is to distinguish gains from additional training from gains attributable to property conditioning. The comparison wrapper supplies properties only to A1/A2/A3. A0c's successful full-test snapshot is therefore not evidence that numerical targets can control generated molecules.

## A1 scalar-vector prefix

Each raw property is normalised with frozen train-only statistics, then mapped by its own scalar MLP. Property-type and observed/missing embeddings are added. Three resulting vectors are prepended before the molecular positions enter the shared denoiser. The first three output positions are removed before molecular output projection.

This transfers the supplied JoPMol-style property-prefix idea. Molecular absolute positions shift by three positions relative to A0c, so a matched-position ablation is needed to isolate the conditioning mechanism from positional effects.

## A2 signed digit/place prefix

A2 also uses train-only z scores, but represents each value with one sign symbol plus seven digit/place symbols at orders `2, 1, 0, -1, -2, -3, -4`. Three properties add 24 prefix tokens. Values are rounded to four decimal places; the representable range is ±999.9999. Tokenisation uses discrete digit extraction and is not differentiable with respect to the numeric input.

The module combines learned symbol, property-type and missing-value embeddings with fixed numerical encodings. For a digit `d` at decimal order `p`, the fixed vector follows the official Regression Transformer implementation: alternating `v, -v, v/3, -v/3, ...`, where `v = d × 10^p`, scaled by `numerical_vmax / 10`. This is an encoding transfer, not a reproduction of Regression Transformer's XLNet architecture, alternating regression/generation tasks or complete training objective.

A2 shifts molecular positions by 24 and has a longer attention sequence. Equal optimizer-step budgets do not imply equal compute or memory cost.

## A3 per-block zero-gated residual adapter

A3 pools the three scalar-vector condition tokens, retaining the learned missingness signal. After each unchanged pretrained block, an independent adapter computes condition-dependent shift, scale and gate values. It applies shift/scale to normalised block states, passes the result through a bottleneck MLP, and adds a `tanh(gate)`-scaled residual.

The final modulation projection is zero-initialised. Consequently, A3 initially equals A0c exactly when shared parameters, inputs and evaluation mode match. At that point, the zero gate also means that some condition-path gradients are zero; gradient flow develops after the gating projection updates. This is an intended initialisation property, not proof of failed conditioning.

A3 is inspired by Graph DiT's adaptive conditioning, but it leaves the existing TediMol block intact and adds a sequence adapter **after** it. It does not diffuse molecular graphs or replace the backbone with the original Graph DiT model.

## Missing values, masks and comparison limits

Unobserved numerical values become `z=0` plus a learned missingness embedding. Observed NaN/Inf values are rejected. Normalisation buffers are stored in the model and are not fitted on test data.

The recovered text-mask path passes a broadcast 0/1 tensor directly to low-level `BertEncoder`. In the pinned Transformers eager implementation, that argument is an **additive attention bias**, rather than a conventional boolean padding mask. Padding therefore is not guaranteed to be excluded. This inherited behaviour is preserved for source/checkpoint traceability, not endorsed as correct padding handling. Tests use all-valid synthetic text; a corrected mask protocol would require explicit versioning and re-evaluation of the ongoing experiment.

The routes differ in parameter count, sequence length and absolute positions. The initial 68-row scan uses seen training records, properties are correlated, and text may already identify a particular molecule. Neither small-panel response nor a nonzero input derivative is sufficient to demonstrate useful target control on unseen molecules. A2's discrete input encoding also makes input derivatives an unsuitable like-for-like diagnostic against the continuous branches.
