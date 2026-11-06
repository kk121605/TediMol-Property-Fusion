"""Verify published recovered-file hashes against the local publication manifest."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "docs/source_manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["files"]:
        path = root / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Recovered source changed: {entry['path']}")
    print(
        json.dumps(
            {
                "recovered_files_verified": len(manifest["files"]),
                "recorded_S2_property_sources_matched": sum(
                    manifest["s2_recorded_property_source_match"].values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
