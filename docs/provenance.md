# Provenance and publication scope

## Personal work and underlying sources

Yikun Luo identifies the stage-2 branch code as his research work within the group. The project extends the group-provided TediMol backbone and supplied JoPMol-style property-prefix reference. This repository records that distinction rather than claiming original ownership of TediMol, JoPMol, Regression Transformer, Graph DiT, BioLinkBERT or the inherited diffusion utilities.

The contribution represented here is the stage-2 integration and comparison of scalar prefixes, signed digit/place numerical prefixes, and per-block zero-gated adapters; shared training/sampling/provenance machinery; continuation/diagnostic tooling; and evidence checks. Publication packaging, documentation and portable checks were prepared with coding assistance. The repository is a participant's evolving implementation, not an official release of the underlying research-group frameworks.

## Recovered source selection

The main source was copied from the supplied `12_三架构融合代码与学习` directory. The stable S2 training/sampling scripts came from `13_S2_10k统一筛选`. The full source paths and byte hashes are recorded in [`source_manifest.json`](source_manifest.json). Original project files were not modified.

All 15 source entries in the recorded S2 property-source receipt matched the recovered implementation. The 21 recorded baseline source entries also matched at the time of publication preparation. These matches establish byte correspondence with local receipts, not individual line authorship or a verified upstream Git history.

The three conditioning and three fusion modules remain in physically separate method directories. A0c follows the shared text-only constructor. The later six method-specific train/sample entrypoint wrappers were absent from the original trained source receipt. They are retained only under `archive/branch_entrypoints`, outside the live source glob, so they do not alter that recorded receipt.

Selected long-run continuation, resume-contract and NumPy seed-recovery algorithms are retained under `archive/continuation`. Older A30 deployment instructions contain a stop-use notice and later fixes were applied on the cluster. This repository does not present the archived local runner as a complete, current deployment package. Personal host paths in inherited code/comments describe historical assumptions; live cluster configuration and account files are omitted.

## Publication additions

| File or area | Added for this release |
| --- | --- |
| `portfolio.py` | Tiny constructors and demo around the actual recovered model and diffusion update |
| `tests/` | Portable branch, encoding, masking-of-missing-values, continuation and metric tests |
| `tools/verify_sources.py` | Recovered-file hash verification |
| `tools/evaluate_predictions.py` | Path-independent recomputation of both test snapshots |
| `evidence/a0c_fulltest/` | Selected original prediction CSV bytes, source hashes, aggregate metrics and derived training membership hashes |
| `evidence/s2_10k/` | Selected historical training-status/configuration summary |
| README, docs, CI | English presentation, source/evidence boundaries and Linux CPU checks |

The factory's illustrative property statistics are fixtures, not new estimates of the real data. The real research schema/statistics remain under `research/.../data/property_v2_senior` for provenance. No original algorithm was silently rewritten to fit the demo, and the original low-level text-mask convention is explicitly documented.

Original data CSVs, text-feature caches, model/encoder weights, large archives, private reports, paper PDFs and host-specific deployment configuration are excluded. Some test-set reference structures and generated SELFIES appear in the authorised prediction evidence. The train membership hashes are derived data and may reveal membership to someone who already has a candidate structure.

## References

- TediMol manuscript supplied by the group: *Bridging Language and Chemistry: Conditional Diffusion for Text-Guided Molecular Generation*. The local copy lists anonymous authors and says code will be released upon acceptance. The contributor authorised public release of the present scope; this repository does not infer acceptance or an official publication venue from local filenames.
- JoPMol: Yuan et al., *Gene Expression–Informed Jointly Controlled Generative Modeling for Precision Molecular Design*, DOI `10.1145/3767308.3835669`.
- Regression Transformer: official IBM repository snapshot `2140db39f52acbcd34bd79a64324ab5976361113`. All 70 digit/place encodings were compared with its `get_float_encoding` in the earlier local engineering work; publication tests also check the formula.
- Graph DiT: official snapshot `510d9c058e2238dd0ee8fe61243dacb38e67af06`. The local snapshot had no root licence. The project implements an inspired sequence adapter and does not redistribute the complete graph model.
- TGM-DLM and OpenAI improved-diffusion: retained code/method lineage, acknowledged in the README and third-party notices.
