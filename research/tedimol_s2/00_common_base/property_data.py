"""Load and align frozen property CSVs with the baseline ChemBI row order."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_property_protocol(property_dir: Path, requested_names: tuple[str, ...]):
    schema = json.loads((property_dir / "property_schema.json").read_text(encoding="utf-8"))
    stats = json.loads((property_dir / "train_stats.json").read_text(encoding="utf-8"))
    schema_order = tuple(schema["selected_order"])
    if any(name not in schema_order for name in requested_names):
        raise ValueError(
            f"Requested properties {requested_names} are not a subset of frozen schema {schema_order}"
        )
    means = [stats["descriptors"][name]["mean"] for name in requested_names]
    stds = [stats["descriptors"][name]["std_population"] for name in requested_names]
    return schema, means, stds


def load_aligned_properties(
    csv_path: Path,
    expected_ids: list[str],
    property_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    table = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", *property_names}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} lacks required fields {sorted(required)}")
        for row in reader:
            mol_id = row["id"]
            if mol_id in table:
                raise ValueError(f"Duplicate property row id {mol_id}")
            table[mol_id] = [float(row[name]) for name in property_names]
    if list(table) != expected_ids:
        first_mismatch = next(
            (
                (i, expected_ids[i], list(table)[i])
                for i in range(min(len(expected_ids), len(table)))
                if expected_ids[i] != list(table)[i]
            ),
            None,
        )
        raise ValueError(
            f"Property ID order does not exactly match ChemBI row order; first mismatch={first_mismatch}"
        )
    values = np.asarray([table[mol_id] for mol_id in expected_ids], dtype=np.float32)
    observed = np.isfinite(values)
    if not observed.all():
        raise ValueError("v1 protocol is fail-closed: found missing/non-finite property values")
    return values, observed

