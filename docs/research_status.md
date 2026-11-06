# Research status and evidence

This project is ongoing. The status below is tied to the local artifacts supplied for this release, not to a live query of the training cluster. No cloud training or new full-model generation was started while preparing this repository.

## What has evidence

The local S2 receipts record that all four branches reached 10,000 optimizer updates. Their status summaries and configurations are in [`evidence/s2_10k/status_summary.json`](../evidence/s2_10k/status_summary.json). The small-panel experiments are engineering diagnostics using seen training records. They do not establish stable property control, held-out generalisation or a winning branch. The protocol intends 1.2M second-stage updates per branch, but that target must not be confused with completed training for every branch.

For A0c, the newer result package records completion of 1.2M second-stage updates and 2,795 test predictions. Its archive SHA256 is `8b6ef7ed969966a6cb7b76bc00c9e89496a097c6bd2fddfd5d98dfd0be0a7357`. All 11 listed artifact hashes were verified before extracting the public prediction snapshot. The matching stage-1 predictions are also included. The standalone verifier independently recomputes both sets of core metrics, totaling 5,590 predictions.

The current local evidence does not provide an equivalent full-test comparison among A1, A2 and A3. Their later remote progress was not inspected for this publication. A0c is a text-only continued-training control, not the proposed property-conditioned system.

## Metric definitions

| Metric | Denominator and definition |
| --- | --- |
| Validity | Valid nonempty RDKit molecule / all predictions |
| Exact match | Canonical isomeric SMILES equality with the paired reference / all predictions |
| MACCS, RDK, Morgan | Mean Tanimoto similarity over valid generated/reference pairs |
| Uniqueness | Unique canonical valid structures / valid predictions |
| Novelty | Unique canonical valid structures absent from the training reference set / unique canonical valid structures |

Morgan uses radius 2 and 2,048 bits. No salt removal, tautomer merging or other standardisation is applied beyond RDKit canonical isomeric SMILES. The training membership representation contains SHA256 hashes of those strings. It is an exact membership mechanism, not a privacy guarantee or an independent validation of the dataset's origin.

The included snapshots reproduce 400 → 443 exact matches. New validity is 2,792/2,795, compared with 2,789/2,795 previously. Similarity increases, but uniqueness and novelty decrease. Similarity means use different valid-pair counts, so this is not a paired statistical test on a fixed valid subset.

## Limits on interpretation

The result is a single training/decoding snapshot with sampling seed 121. It supports a descriptive comparison under recorded settings; no significance, robustness or stable superiority is claimed. FCD and Text2Mol were not run. The original checkpoint tensors are not published here, and only their recorded identities were checked. Recomputing predictions' metrics is distinct from rerunning generation from model weights.

The test set is the supplied ChemBI test split. The release verifies matching test IDs/references across the two prediction files, not molecule-disjoint train/test construction. Novelty statistics do not establish split independence. A dedicated structure/scaffold overlap audit and a frozen evaluation protocol are needed for stronger generalisation claims.

The inherited low-level BERT mask convention is documented in [methods](methods.md). It remains unchanged in this snapshot. Any correction must be recorded as a protocol change rather than silently compared with old checkpoints.

No docking, target binding, biological efficacy, toxicity, ADMET or wet-lab evidence is provided. The CPU demo uses random token IDs/text features and cannot produce meaningful chemistry. The full-model weights, BioLinkBERT assets and research data are not bundled.

## Next research milestones

Complete and document comparable property-branch runs, verify their available conditioning and source versions, and evaluate held-out target adherence alongside validity, uniqueness, novelty and text fidelity. Keep RAW/EMA results distinct, report per-property errors and directional response, and use multiple seeds before making a stability claim. Separate prefix-position, parameter-budget and training-duration effects. These are planned evaluations, not completed results.
