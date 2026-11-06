"""Recompute the published A0c snapshot from prediction CSVs and membership hashes."""

import csv
import hashlib
import json
import math
from pathlib import Path

import selfies as sf
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/a0c_fulltest"


def molecule(text):
    if not text:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(text)
    return mol if mol is not None and mol.GetNumAtoms() else None


def recompute(predictions, training_canonical_hashes):
    """Mirror the recorded canonical/paired-valid metric definitions.

    SHA256 hashes are only an exact training-set membership representation;
    they do not validate the source or independence of the underlying dataset.
    """
    valid, pairs, exact = [], [], 0
    ids = set()
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for index, row in enumerate(predictions):
        if int(row["index"]) != index or row["id"] in ids:
            raise ValueError(
                "Prediction rows must have sequential indices and unique IDs"
            )
        ids.add(row["id"])
        target = molecule(row["reference_smiles"])
        if target is None:
            raise ValueError("Invalid reference molecule")
        try:
            generated = sf.decoder(row["generated_selfies"])
        except sf.DecoderError:
            generated = ""
        mol = molecule(generated)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        valid.append(canonical)
        exact += canonical == Chem.MolToSmiles(target, isomericSmiles=True)
        pairs.append(
            [
                DataStructs.TanimotoSimilarity(
                    MACCSkeys.GenMACCSKeys(mol), MACCSkeys.GenMACCSKeys(target)
                ),
                DataStructs.TanimotoSimilarity(
                    Chem.RDKFingerprint(mol), Chem.RDKFingerprint(target)
                ),
                DataStructs.TanimotoSimilarity(
                    fingerprint.GetFingerprint(mol), fingerprint.GetFingerprint(target)
                ),
            ]
        )
    unique = set(valid)
    novel = sum(
        hashlib.sha256(s.encode("utf-8")).hexdigest() not in training_canonical_hashes
        for s in unique
    )
    n = len(predictions)
    if n == 0:
        raise ValueError("No prediction rows")
    return {
        "n_generated": n,
        "n_valid": len(valid),
        "n_valid_reference_pairs": len(pairs),
        "n_exact": exact,
        "n_unique": len(unique),
        "n_novel_unique": novel,
        "validity": len(valid) / n,
        "exact_canonical_isomeric_smiles": exact / n,
        "maccs_paired_valid_mean": sum(p[0] for p in pairs) / len(pairs)
        if pairs
        else None,
        "rdk_paired_valid_mean": sum(p[1] for p in pairs) / len(pairs)
        if pairs
        else None,
        "morgan_r2_2048_paired_valid_mean": sum(p[2] for p in pairs) / len(pairs)
        if pairs
        else None,
        "uniqueness_valid_canonical": len(unique) / len(valid) if valid else None,
        "novelty_unique_valid_canonical": novel / len(unique) if unique else None,
    }


def verify_snapshot():
    record = json.loads((EVIDENCE / "snapshot.json").read_text(encoding="utf-8"))
    for name, digest in record["files_sha256"].items():
        if hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Evidence file hash mismatch: {name}")
    training = set(
        json.loads((EVIDENCE / "training_canonical_sha256.json").read_text())
    )
    results, identities = {}, []
    for key, filename in [
        ("stage1_300k", "stage1_predictions.csv"),
        ("a0c_stage2_1200k", "a0c_predictions.csv"),
    ]:
        with (EVIDENCE / filename).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        identities.append([(r["id"], r["reference_smiles"]) for r in rows])
        metrics = recompute(rows, training)
        for name, expected in record[key].items():
            actual = metrics[name]
            if actual is None or not math.isclose(
                actual, expected, abs_tol=1e-12, rel_tol=0
            ):
                raise ValueError(
                    f"Metric mismatch: {key}/{name}: {actual} versus {expected}"
                )
        results[key] = metrics
    if identities[0] != identities[1] or len(identities[0]) != 2795:
        raise ValueError("Prediction snapshots do not use the same 2,795 references")
    return results


if __name__ == "__main__":
    print(
        json.dumps({"evidence_verified": True, "results": verify_snapshot()}, indent=2)
    )
