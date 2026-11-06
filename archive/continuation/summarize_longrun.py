"""Summarize complete sampling panels; do not gate training on property response."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''): h.update(part)
    return h.hexdigest()


def mean(values):
    return sum(values)/len(values) if values else None


def summarize(path):
    with path.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=68: raise ValueError(f'Incomplete sampling panel: {path}')
    valid=[r for r in rows if r['valid']=='1']
    refs=[r for r in rows if r['regime']=='reference']
    if len(refs)!=32: raise ValueError('Expected 32 reference samples')
    ref_valid=[r for r in refs if r['valid']=='1']
    groups=defaultdict(list)
    for r in rows:
        if r['regime']=='sweep': groups[(r['source_index'],r['sweep_property'])].append(r)
    if len(groups)!=12: raise ValueError('Expected 12 sweep groups')
    sweeps=[]
    for (idx,prop),group in sorted(groups.items()):
        group.sort(key=lambda r:float(r['z']))
        if [float(r['z']) for r in group]!=[-1.,0.,1.]: raise ValueError('Wrong sweep targets')
        all_valid=all(r['valid']=='1' for r in group)
        ys=[float(r['generated_'+prop]) for r in group] if all_valid else []
        if ys and not all(math.isfinite(v) for v in ys): raise ValueError('Nonfinite valid property')
        sweeps.append(dict(source_index=int(idx),property=prop,all_valid=all_valid,
            distinct_smiles=len({r['generated_smiles'] for r in group if r['valid']=='1'}),
            generated_target_property=ys,
            target_property=[float(r['target_'+prop]) for r in group],
            numeric_response=all_valid and max(ys)-min(ys)>1e-6,
            strictly_increasing=all_valid and ys[0]+1e-6<ys[1] and ys[1]+1e-6<ys[2]))
    return dict(count=len(rows),valid=len(valid),reference_count=len(refs),reference_valid=len(ref_valid),
        reference_exact=sum(int(r['exact_reference']) for r in refs),
        reference_morgan_valid_mean=mean([float(r['morgan_similarity']) for r in ref_valid]),
        reference_mae_valid={n:mean([float(r['abs_error_'+n]) for r in ref_valid]) for n in ('logP','SA','TPSA')},
        sweep_numeric_response_groups=sum(s['numeric_response'] for s in sweeps),
        sweep_strictly_increasing_groups=sum(s['strictly_increasing'] for s in sweeps),groups=sweeps)


def package(root,destination):
    # Includes logs, CSV, configs, source identity and receipts; excludes model tensors/caches.
    allowed={'.json','.jsonl','.csv','.log','.py','.sh','.md'}
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in allowed]
    manifest=root/'evidence_manifest.json'
    files=[p for p in files if p!=manifest]
    destination.mkdir(parents=True,exist_ok=True)
    output=destination/f'TediMol_S2_longrun_evidence_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000:06d}.tar.gz'
    # The current launcher log is still being written by tee. Freeze a copy first
    # so archived bytes and manifest hashes always describe the same snapshot.
    with tempfile.TemporaryDirectory(prefix='tedimol_evidence_',dir=destination) as tmp:
        frozen=Path(tmp)
        for p in files:
            target=frozen/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        frozen_files=[p for p in sorted(frozen.rglob('*')) if p.is_file()]
        manifest.write_text(json.dumps(dict(files=[dict(path=p.relative_to(frozen).as_posix(),bytes=p.stat().st_size,
            sha256=sha(p)) for p in frozen_files],excludes='All model checkpoint tensors; their identities are recorded separately'),
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        shutil.copyfile(manifest,frozen/manifest.name)
        with tarfile.open(output,'w:gz',compresslevel=1) as tar:
            for p in sorted(frozen.rglob('*')):
                if p.is_file():tar.add(p,arcname=f'{root.name}/{p.relative_to(frozen).as_posix()}',recursive=False)
    checksum=output.with_name(output.name+'.sha256')
    checksum.write_text(f'{sha(output)}  {output.name}\n',encoding='ascii')
    print('LONGRUN_EVIDENCE_PACKAGE_PASS',output,flush=True)
    print('CHECKSUM_FILE='+str(checksum),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--package-dir',type=Path,required=True); a=p.parse_args()
    report=dict(scope='seen-train learning curve, not held-out evidence',panels={})
    paths=list(a.root.glob('*/evaluation/step*/*.csv'))+list(a.root.glob('reference_s2_10k/*/*.csv'))
    for path in sorted(paths):
        report['panels'][path.relative_to(a.root).as_posix()]=summarize(path)
    (a.root/'learning_curve_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('LONGRUN_SUMMARY_PASS',len(report['panels']),flush=True)
    package(a.root,a.package_dir)


if __name__=='__main__': main()
