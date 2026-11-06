"""Shared property batch, train-only normalization and strict key audit."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class PropertyBatch:
    """Raw properties and an observed-value mask, both shaped [B, K]."""

    values: Tensor
    observed: Tensor


class PropertyNormalizer(nn.Module):
    """Frozen train-only z-score statistics stored inside the checkpoint."""

    def __init__(self, means: Sequence[float], stds: Sequence[float]):
        super().__init__()
        means_t = torch.as_tensor(means, dtype=torch.float32)
        stds_t = torch.as_tensor(stds, dtype=torch.float32)
        if means_t.ndim != 1 or stds_t.ndim != 1 or means_t.shape != stds_t.shape:
            raise ValueError("means and stds must be one-dimensional and have the same shape")
        if not torch.isfinite(means_t).all() or not torch.isfinite(stds_t).all():
            raise ValueError("normalization statistics must be finite")
        if torch.any(stds_t <= 0):
            raise ValueError("all property standard deviations must be positive")
        self.register_buffer("means", means_t)
        self.register_buffer("stds", stds_t)

    @property
    def num_properties(self) -> int:
        return int(self.means.numel())

    def forward(self, batch: PropertyBatch) -> tuple[Tensor, Tensor]:
        values = batch.values
        observed = batch.observed.bool()
        if values.ndim != 2 or values.shape[1] != self.num_properties:
            raise ValueError(
                f"expected property values [B, {self.num_properties}], got {tuple(values.shape)}"
            )
        if observed.shape != values.shape:
            raise ValueError("observed mask must have the same shape as property values")
        if not torch.isfinite(values[observed]).all():
            raise ValueError("observed property values must be finite")
        safe_values = torch.where(observed, values, self.means.to(values))
        z = (safe_values - self.means.to(values)) / self.stds.to(values)
        # Missing values are represented by a neutral z value plus a learned mask embedding.
        z = torch.where(observed, z, torch.zeros_like(z))
        return z, observed


def audit_checkpoint_keys(
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    allowed_new_prefixes: Sequence[str] = ("property_encoder.", "property_prefix.", "property_adapters."),
) -> dict:
    """Fail if a base-model key is missing or an old checkpoint key is unexpected."""

    missing = sorted(missing_keys)
    unexpected = sorted(unexpected_keys)
    disallowed_missing = [
        key for key in missing if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    report = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "allowed_new_prefixes": list(allowed_new_prefixes),
        "disallowed_missing_keys": disallowed_missing,
        "pass": not disallowed_missing and not unexpected,
    }
    if not report["pass"]:
        raise RuntimeError(f"Checkpoint compatibility failed: {report}")
    return report


