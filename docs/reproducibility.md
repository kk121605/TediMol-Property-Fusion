# Reproduction guide

## Supported CPU check

Use Python 3.11 or 3.12, PyTorch 2.5.1 CPU, and `requirements-cpu.txt`. Transformers is pinned to 4.45.2 because the recovered code calls low-level BERT interfaces directly. The CPU demo reads source files and small schema/vocabulary assets only. It does not fit training statistics or load an encoder, text cache or full checkpoint.

```bash
python portfolio.py
python -m unittest discover -s tests -v
python tools/verify_sources.py
python tools/evaluate_predictions.py
```

The first command builds real reduced-dimension versions of all four branches and uses the original diffusion-loss update. It prints parameter counts and finite-gradient checks. Differences between tiny-model losses are not comparative scientific results.

The tests use an in-memory text-only parent to verify shared-parameter initialisation, A3 identity, prefix lengths, branch forward/backward, real 20-step sampling, and uninterrupted versus restored AdamW/EMA updates. They check encoding and missing-value semantics, not numerical target control on real molecules.

The original study used CUDA PyTorch 2.5.1 and a larger environment. Its exact historical dependency list is retained at `research/tedimol_s2/00_common_base/requirements.txt`; it includes packages unnecessary for the supported CPU check. Do not interpret the shorter CPU requirements as an exact GPU-runtime reproduction.

## Prediction evidence

`tools/evaluate_predictions.py` verifies the included evidence file hashes and recomputes validity, canonical exact match, three fingerprint similarities, uniqueness and novelty. It uses the same 2,795 IDs/references in both snapshots. RDKit is pinned to 2024.3.5 to match the reported canonicalisation/fingerprint semantics.

The novelty membership file was produced from the original 22,169 valid canonical training structures. It contains sorted SHA256 digests of canonical isomeric SMILES. The original training CSV hash is retained in `snapshot.json`; the CSV itself is not distributed. A recipient with authorised data can independently recreate this membership set. The membership set alone cannot validate the dataset, its training use, or split independence.

## Full research training and sampling

The preserved entrypoints are `research/tedimol_s2/00_common_base/train_s2.py` and `sample_s2.py`, with common update logic in `comparison_runtime.py`. Their first CLI help can be inspected from that directory. They require external research assets and strict provenance:

1. Authorised headerless ChemBI CSVs with identifier, SMILES, SELFIES and description, in the original row order. The original training size is 22,169; the test size is 2,795.
2. Aligned `train_properties.csv` and `test_properties.csv`, matching the supplied `logP, SA, TPSA` schema and manifests. The frozen normalisation statistics are fitted on train only.
3. The original 323-ID SELFIES vocabulary, already included byte-for-byte, and matching maximum-length/decoding conventions.
4. Local BioLinkBERT assets and matching text-state caches/masks. The encoder weights and caches are external.
5. The full first-stage checkpoint and its original source/data/cache/runtime receipt, or the exact S2 checkpoint/source snapshot when resuming. Model keys, RAW/EMA, optimizer, sampler and source identities must agree.
6. A CUDA environment matching the intended protocol, sufficient GPU memory, a dedicated output directory, and an explicit evaluation plan.

No one-command full study is claimed in this release. The source/data/cache/runtime guards intentionally reject changed historical assets. Do not bypass them by editing checkpoint provenance. The provided tiny factory does not execute those full-data guards and must not be used as a substitute for validating a historical checkpoint.

## Continuation archive

The archived 1.2M-step runner and tests retain their historical filesystem assumptions. Some archived tests refer to original local reports and are not part of the supported test command. The separate seed32 recovery module preserves Python/PyTorch seed values and wraps only NumPy's seed modulo 2^32; its historical launcher also requires a deployment receipt.

The original cluster launcher required fixes for directory layout, source-glob receipts, lock placement and seed overflow. Those changes are not all present as one verified local deployment. Treat the archive as implementation history to inspect, not as a ready-to-submit Slurm job. No cluster job is launched by any supported CPU command or CI workflow.
