"""Audit original data/vocabulary. Never build a replacement vocabulary or drop rows."""
from collections import Counter
from common import ROOT, VOCAB_SHA256, MoleculeTokenizer, read_rows, split_selfies, sha256, write_json


def main():
    tokenizer = MoleculeTokenizer()
    report = dict(vocab_sha256=VOCAB_SHA256, vocab_size=len(tokenizer), training_rows_used=22169,
        protocol='senior_TediMol_phase1_original_vocab323_legacy_masks',
        length_policy='Original tokenizer: truncate at 256 and force last token EOS; no rows removed', splits={})
    all_ids = []
    for split in ('train_ChemBI','test_ChemBI'):
        path = ROOT/'data'/(split+'.csv')
        rows = read_rows(path)
        oov = Counter()
        long_ids = []
        for row in rows:
            parts = split_selfies(row['selfies'])
            oov.update(p for p in parts if p not in tokenizer.token_to_id)
            if len(parts)+2 > 256:
                long_ids.append(row['id'])
            assert len(tokenizer.encode(row['selfies'])) == 256
        if oov:
            raise ValueError(f'Original vocabulary missing tokens: {oov}')
        all_ids.append({r['id'] for r in rows})
        report['splits'][split] = dict(rows=len(rows), sha256=sha256(path), oov_tokens=dict(oov), overlength_ids=long_ids)
    assert not all_ids[0]&all_ids[1], 'Train/test ID overlap'
    report['training_rows_used'] = report['splits']['train_ChemBI']['rows']
    write_json(ROOT/'data/manifest.json', report)
    print('DATA PASS:', {s:x['rows'] for s,x in report['splits'].items()}, 'vocab_size=',len(tokenizer))


if __name__ == '__main__':
    main()
