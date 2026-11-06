"""Runtime and provenance helpers for property-conditioned TediMol runs.

This module is deployed next to the first-stage ``common.py`` and ``model.py``.
It deliberately accepts an A0 checkpoint only after checking the vocabulary,
data, baseline source files, model shape, and allowed new property keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import numpy as np
from transformers import BertConfig

from common import ROOT, MoleculeTokenizer, prepared_data, sha256, source_hashes
from model import DEFAULT_MODEL
from property_common import audit_checkpoint_keys
from property_data import load_property_protocol
from method_registry import get_model_class, fusion_sources


A0_FORMAT = "tedimol-senior-v2-vocab323"
OVERLAY_ROOT = Path(__file__).resolve().parent
DEFAULT_PROPERTY_DIR = OVERLAY_ROOT / "data" / "property_v2_senior"
PROPERTY_SOURCE_FILES = (
    "property_common.py",
    "property_model_base.py",
    "method_registry.py",
    "property_metrics.py",
    "property_data.py",
    "property_runtime.py",
    "comparison_runtime.py",
    "train_compare.py",
    "sample_compare.py",
)


def _legacy_numpy_rng_reconstruct(*args):
    # NumPy 1.x RNG pickle refers to numpy.core; NumPy 2.x renamed it _core.
    # This explicit alias permits only the known NumPy array reconstruction.
    from numpy.core.multiarray import _reconstruct
    return _reconstruct(*args)


_legacy_numpy_rng_reconstruct.__module__ = "numpy.core.multiarray"
_legacy_numpy_rng_reconstruct.__name__ = "_reconstruct"


def _torch_load(path: Path, *, mmap: bool = True):
    kwargs = dict(map_location="cpu", weights_only=True)
    if mmap:
        kwargs["mmap"] = True
    try:
        # Original A0 records torch.__version__ as TorchVersion, a str subclass.
        # Explicitly allow that one known class; do not fall back to unsafe pickle.
        from torch.torch_version import TorchVersion
        from numpy.core.multiarray import _reconstruct
        allowed = [TorchVersion, _legacy_numpy_rng_reconstruct, _reconstruct,
                   np.ndarray, np.dtype, type(np.dtype("uint32"))]
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, **kwargs)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        kwargs.pop("mmap", None)
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, **kwargs)


def property_source_hashes() -> dict[str, str]:
    result = {}
    for name in PROPERTY_SOURCE_FILES:
        path = OVERLAY_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing deployed property source: {path}")
        result[name] = sha256(path)
    for path in fusion_sources():
        result[path.relative_to(OVERLAY_ROOT.parent).as_posix()] = sha256(path)
    return result


def audit_baseline_sources(checkpoint: dict) -> dict:
    expected = checkpoint["provenance"]["code_sha256"]
    current = source_hashes()
    missing = sorted(name for name in expected if name not in current)
    changed = sorted(
        name for name, digest in expected.items()
        if current.get(name) is not None and current[name] != digest
    )
    report = {
        "expected_files": len(expected),
        "current_files": len(current),
        "missing_baseline_files": missing,
        "changed_baseline_files": changed,
        "extra_files_allowed": sorted(set(current) - set(expected)),
        "pass": not missing and not changed,
    }
    if not report["pass"]:
        raise RuntimeError(f"A0 source audit failed: {report}")
    return report


def property_protocol_receipt(
    property_dir: Path,
    property_names: Sequence[str],
) -> tuple[dict, list[float], list[float], dict]:
    property_dir = property_dir.resolve()
    schema, means, stds = load_property_protocol(property_dir, tuple(property_names))
    manifest_path = property_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_manifest = prepared_data()
    if manifest["inputs"]["baseline_train"]["sha256"] != baseline_manifest["splits"]["train_ChemBI"]["sha256"]:
        raise ValueError("Property table was not derived from the active train_ChemBI.csv")
    if manifest["inputs"]["baseline_test"]["sha256"] != baseline_manifest["splits"]["test_ChemBI"]["sha256"]:
        raise ValueError("Property table was not derived from the active test_ChemBI.csv")
    if schema.get("protocol_id") != "senior-chembi-logp-sa-tpsa-20261008":
        raise ValueError("Unexpected or superseded property protocol")
    required = (
        "property_schema.json",
        "train_stats.json",
        "train_properties.csv",
        "test_properties.csv",
    )
    hashes = {}
    for name in required:
        path = property_dir / name
        digest = sha256(path)
        if manifest["outputs"].get(name) != digest:
            raise ValueError(f"Property artifact hash mismatch: {name}")
        hashes[name] = digest
    receipt = {
        "property_dir_name": property_dir.name,
        "schema_version": schema["schema_version"],
        "protocol_id": schema["protocol_id"],
        "authoritative_source": schema["authoritative_source"],
        "property_names": list(property_names),
        "normalization": schema["normalization"],
        "artifact_sha256": hashes,
        "manifest_sha256": sha256(manifest_path),
        "baseline_train_csv_sha256": manifest["inputs"]["baseline_train"]["sha256"],
        "baseline_test_csv_sha256": manifest["inputs"]["baseline_test"]["sha256"],
        "senior_train_property_csv_sha256": manifest["inputs"]["senior_train_property"]["sha256"],
        "senior_test_property_csv_sha256": manifest["inputs"]["senior_test_property"]["sha256"],
    }
    return schema, means, stds, receipt


def build_property_model(
    vocab_size: int,
    base_model_config: dict,
    *,
    property_mode: str,
    property_names: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    property_bottleneck: int | None = None,
) -> torch.nn.Module:
    settings = dict(DEFAULT_MODEL)
    settings.update(base_model_config)
    config = BertConfig(
        intermediate_size=settings.pop("intermediate_size"),
        max_position_embeddings=settings.pop("max_position_embeddings"),
    )
    return get_model_class(property_mode)(
        vocab_size=vocab_size,
        config=config,
        training_mode="e2e",
        property_mode=property_mode,
        property_names=tuple(property_names),
        property_means=means,
        property_stds=stds,
        property_bottleneck=property_bottleneck,
        **settings,
    )


def validate_a0_checkpoint(checkpoint: dict, tokenizer: MoleculeTokenizer) -> dict:
    if checkpoint.get("format") != A0_FORMAT:
        raise ValueError(f"Expected {A0_FORMAT}, got {checkpoint.get('format')!r}")
    if checkpoint["provenance"]["vocab_sha256"] != sha256(tokenizer.path):
        raise ValueError("A0 checkpoint vocabulary hash mismatch")
    manifest = prepared_data()
    if checkpoint["provenance"]["train_csv_sha256"] != manifest["splits"]["train_ChemBI"]["sha256"]:
        raise ValueError("A0 checkpoint training-data hash mismatch")
    if checkpoint["config"]["max_length"] != tokenizer.max_length:
        raise ValueError("A0 checkpoint sequence-length mismatch")
    if checkpoint["config"].get("tiny") or checkpoint["config"].get("cache_limit"):
        raise ValueError("Property experiments require the full-data, non-tiny A0 checkpoint")
    if checkpoint["step"] != checkpoint["config"]["steps"]:
        raise ValueError("A0 checkpoint did not reach its configured target step")
    return audit_baseline_sources(checkpoint)


def initialize_from_a0(
    model: torch.nn.Module,
    checkpoint: dict,
    *,
    init_weights: str,
) -> dict:
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    key_report = audit_checkpoint_keys(
        incompatible.missing_keys,
        incompatible.unexpected_keys,
        allowed_new_prefixes=("property_encoder.", "property_adapters."),
    )
    if init_weights not in {"ema", "raw"}:
        raise ValueError("init_weights must be 'ema' or 'raw'")
    if init_weights == "ema":
        named = dict(model.named_parameters())
        missing_ema = sorted(name for name in checkpoint["ema"] if name not in named)
        if missing_ema:
            raise RuntimeError(f"EMA contains unknown A0 parameters: {missing_ema}")
        with torch.no_grad():
            for name, value in checkpoint["ema"].items():
                named[name].copy_(value)
    new_parameter_count = sum(
        p.numel() for name, p in model.named_parameters() if name.startswith("property_")
    )
    model_parameter_count = sum(v.numel() for v in model.parameters())
    return {
        "checkpoint_keys": key_report,
        "initial_weights": init_weights,
        "base_parameter_count": model_parameter_count - new_parameter_count,
        "model_parameter_count": model_parameter_count,
        "new_parameter_count": new_parameter_count,
    }


def load_a0_into_new_property_model(
    checkpoint_path: Path,
    *,
    property_dir: Path,
    property_mode: str,
    property_names: Sequence[str],
    property_bottleneck: int | None,
    init_weights: str,
    device: str,
) -> tuple[torch.nn.Module, dict, dict, dict]:
    checkpoint_path = checkpoint_path.resolve()
    tokenizer = MoleculeTokenizer()
    checkpoint = _torch_load(checkpoint_path)
    baseline_source_report = validate_a0_checkpoint(checkpoint, tokenizer)
    _, means, stds, property_receipt = property_protocol_receipt(
        property_dir, property_names
    )
    model = build_property_model(
        len(tokenizer),
        checkpoint["config"]["model"],
        property_mode=property_mode,
        property_names=property_names,
        means=means,
        stds=stds,
        property_bottleneck=property_bottleneck,
    )
    init_report = initialize_from_a0(model, checkpoint, init_weights=init_weights)
    model.to(device)
    report = {
        "result": "CHECKPOINT_COMPATIBILITY_PASS",
        "a0_checkpoint": str(checkpoint_path),
        "a0_checkpoint_sha256": sha256(checkpoint_path),
        "a0_step": checkpoint["step"],
        "a0_format": checkpoint["format"],
        "property_mode": property_mode,
        "property_names": list(property_names),
        "property_bottleneck": property_bottleneck,
        "baseline_source_audit": baseline_source_report,
        **init_report,
    }
    return model, checkpoint, property_receipt, report


__all__ = [
    "A0_FORMAT",
    "DEFAULT_PROPERTY_DIR",
    "build_property_model",
    "property_protocol_receipt",
    "property_source_hashes",
    "load_a0_into_new_property_model",
]
