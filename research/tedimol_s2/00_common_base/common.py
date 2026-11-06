"""Project-relative assets and the senior's original vocabulary/tokenizer semantics."""
import csv
import hashlib
import json
import importlib.metadata as metadata
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
VOCAB_SHA256 = '0f8157884f95680de571601d2f17cfae1207a25a2cb48a8683ed05e67dfcb819'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    rows, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as stream:
        for n, row in enumerate(csv.reader(stream), 1):
            if len(row) != 4 or not all(v.strip() for v in row):
                raise ValueError(f'Expected four nonempty columns: {path}:{n}')
            cid, smiles, selfies, text = (v.strip() for v in row)
            if cid in seen:
                raise ValueError(f'Duplicate ID {cid}')
            seen.add(cid)
            rows.append(dict(id=cid, smiles=smiles, selfies=selfies, text=text))
    if not rows:
        raise ValueError('Empty split')
    return rows


class MoleculeTokenizer:
    """Wrap the delivered tokenizer without reordering, deduplicating or filtering."""
    def __init__(self, max_length=256):
        from mytokenizers import regexTokenizer
        self.path = ROOT/'data/selfies_vocab_CHEBI.txt'
        if sha256(self.path) != VOCAB_SHA256:
            raise ValueError('Use the original 320-line vocabulary, byte-for-byte')
        self.original = regexTokenizer(path=self.path, max_len=max_length)
        self.max_length = max_length
        self.token_to_id = self.original.toktoid
        self.tokens = [self.original.idtotok[i] for i in range(len(self.original))]

    def __len__(self):
        return len(self.original)

    def encode(self, value):
        if ''.join(self.original.rg.findall(value)) != value:
            raise ValueError('Tokenizer did not consume full molecule string')
        return self.original.encode_one(value)[0].tolist()

    def decode(self, ids):
        # Match the old sample/evaluate chain: remove ALL specials, not first-EOS stopping.
        return ''.join(self.tokens[int(i)] for i in ids if int(i) not in (0,1,2))


def split_selfies(value):
    parts = re.findall(r'\[[^\]]+\]|\.', value)
    if ''.join(parts) != value:
        raise ValueError('Invalid SELFIES token stream')
    return parts


def source_hashes():
    paths = list(ROOT.glob('*.py')) + list((ROOT/'improved_diffusion').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}


def runtime_versions():
    names=['torch']+[line.split('==')[0] for line in (ROOT/'requirements.txt').read_text().splitlines()
                     if line.strip() and not line.startswith('#')]
    return {name:metadata.version(name) for name in names}


def prepared_data():
    manifest = read_json(ROOT/'data/manifest.json')
    if sha256(ROOT/'data/selfies_vocab_CHEBI.txt') != VOCAB_SHA256:
        raise ValueError('Original vocabulary changed')
    for split, info in manifest['splits'].items():
        if sha256(ROOT/'data'/(split+'.csv')) != info['sha256']:
            raise ValueError('Data changed since preparation')
    return manifest


def cache_path(split, limit=0):
    return ROOT/'data'/(split+'_desc_states_256'+(f'.limit{limit}' if limit else '')+'.pt')


class FeatureColumn:
    def __init__(self, cache, ids, field):
        self.cache, self.ids, self.field = cache, ids, field

    def __getitem__(self, selection):
        import torch
        indices = list(range(*selection.indices(len(self.ids)))) if isinstance(selection, slice) else list(selection)
        return torch.cat([self.cache[self.ids[i]][self.field] for i in indices], dim=0)


def open_cache(split, limit=0):
    import torch
    rows = read_rows(ROOT/'data'/(split+'.csv'))
    if limit:
        rows = rows[:limit]
    path = cache_path(split, limit)
    meta = read_json(path.with_suffix('.metadata.json'))
    ids = [r['id'] for r in rows]
    if not meta['complete'] or ids != meta['ids'] or meta['csv_sha256'] != sha256(ROOT/'data'/(split+'.csv')):
        raise ValueError('Cache incomplete or CSV/ID mismatch')
    if path.stat().st_size != meta['cache_bytes'] or sha256(path) != meta['cache_sha256']:
        raise ValueError('Text feature file changed or incomplete')
    if meta['preprocess_sha256'] != sha256(ROOT/'process_text.py'):
        raise ValueError('Preprocessor code changed; use its matching cache')
    cache = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if list(cache) != ids:
        raise ValueError('Text feature ID order mismatch')
    for cid in ids:
        if cache[cid]['states'].shape != (1,256,768) or cache[cid]['states'].dtype != torch.float32 or cache[cid]['mask'].shape != (1,256):
            raise ValueError(f'Invalid text feature shape/type: {cid}')
    return rows, FeatureColumn(cache,ids,'states'), FeatureColumn(cache,ids,'mask'), meta
