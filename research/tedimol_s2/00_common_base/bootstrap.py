"""Unpack only the supplied BioLinkBERT files; never contact a model hub."""
import zipfile
from common import ROOT, sha256, write_json


def main():
    archive = ROOT / 'models' / 'BioLinkBERT-base.zip'
    target = ROOT / 'models'
    allowed = {'config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'vocab.txt'}
    with zipfile.ZipFile(archive) as z:
        for name in sorted(allowed):
            member = 'BioLinkBERT-base/' + name
            dest = target / member
            if dest.exists() and dest.stat().st_size == z.getinfo(member).file_size:
                # CRC test below still checks existing content against the archive.
                import zlib
                crc = 0
                with dest.open('rb') as f:
                    for chunk in iter(lambda: f.read(8*1024*1024), b''):
                        crc = zlib.crc32(chunk, crc)
                if crc == z.getinfo(member).CRC:
                    continue
                raise ValueError(f'Existing encoder file differs: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, dest.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
    hashes = {name: sha256(target / 'BioLinkBERT-base' / name) for name in sorted(allowed)}
    write_json(target / 'encoder_manifest.json', dict(archive_sha256=sha256(archive), files=hashes))
    print('BioLinkBERT ready; 6 files verified, local-only loading enabled by scripts.')


if __name__ == '__main__':
    main()
