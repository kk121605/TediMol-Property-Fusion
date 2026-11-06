"""Cloud environment/data preflight; writes a receipt before GPU spending."""
import argparse
import importlib.metadata as metadata
import platform
import shutil
import sys
import torch
import psutil
from common import ROOT, prepared_data, read_json, sha256, write_json
from model import TransformerNetModel2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cpu-check',action='store_true',help='Local code validation; not a GPU pass')
    args=p.parse_args()
    if not (3,10)<=sys.version_info[:2]<=(3,12):
        raise RuntimeError('Use Python 3.10-3.12 for this pinned environment')
    if torch.__version__.split('+')[0]!='2.5.1':
        raise RuntimeError('Expected torch 2.5.1 (CPU for local QA, CUDA for AutoDL)')
    versions={'torch':torch.__version__}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name,version=line.split('==')
        versions[name]=metadata.version(name)
        if versions[name]!=version:
            raise RuntimeError(f'{name}: expected {version}, got {versions[name]}')
    manifest=prepared_data()
    encoder=read_json(ROOT/'models/encoder_manifest.json')
    for name,expected in encoder['files'].items():
        if sha256(ROOT/'models/BioLinkBERT-base'/name)!=expected:
            raise ValueError(f'Encoder file changed: {name}')
    with torch.device('meta'):
        model=TransformerNetModel2(manifest['vocab_size'])
    required_cache=sum(v['rows'] for v in manifest['splits'].values())*256*768*4
    free=shutil.disk_usage(ROOT).free
    cuda=torch.cuda.is_available()
    if not args.cpu_check and not cuda:
        raise RuntimeError('GPU not available; do not start training')
    existing=sum(p.stat().st_size for p in (ROOT/'data').glob('*_desc_states_256.pt'))
    if not args.cpu_check and free<max(0,required_cache-existing)+12*2**30:
        raise RuntimeError('Need text-cache size + at least 12 GiB free disk for checkpoint/temporary files')
    missing_rows=max([v['rows'] for s,v in manifest['splits'].items()
        if not (ROOT/'data'/(s+'_desc_states_256.pt')).exists()]+[0])
    required_ram=missing_rows*256*(768*4+8)+4*2**30 if missing_rows else 0
    if not args.cpu_check and psutil.virtual_memory().available < required_ram:
        raise RuntimeError(f'Need {required_ram/2**30:.1f} GiB AVAILABLE host RAM for uncached text encoding; select >=32 GiB RAM instance')
    receipt=dict(platform=platform.platform(),python=sys.version,versions=versions,cuda_available=cuda,
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30 if cuda else 0,
        host_ram_gib=psutil.virtual_memory().total/2**30,available_host_ram_gib=psutil.virtual_memory().available/2**30,
        required_encoding_ram_gib=required_ram/2**30,free_disk_gib=free/2**30,
        full_float32_text_cache_gib=required_cache/2**30,
        model_parameters=sum(p.numel() for p in model.parameters()),
        result='CPU_QA_ONLY' if args.cpu_check else 'GPU_PREFLIGHT_PASS')
    write_json(ROOT/'receipts/preflight.json',receipt)
    print(receipt)


if __name__=='__main__':
    main()
