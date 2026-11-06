"""Verify the uploaded release using only Python's standard library."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'bundle_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        target = root / name
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'Upload/package integrity failed: {name}')
    print(f"BUNDLE PASS: {len(manifest['files'])} files, {manifest['release']}")


if __name__ == '__main__':
    main()
