"""Continue original S2 checkpoints in a separate directory; no optimizer/EMA reset."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

import torch
from common import MoleculeTokenizer, open_cache, runtime_versions, sha256, write_json
from comparison_runtime import Bridge, FORMAT, load, update, verify_provenance, verify_train_cache
from improved_diffusion.resample import UniformSampler
from property_data import load_aligned_properties
from property_runtime import DEFAULT_PROPERTY_DIR, _torch_load
from runtime import make_diffusion, seed_all
from train_s2 import next_batch, s2_sources
from longrun_contract import MODES, MILESTONES, PROTOCOL, TARGET, promote, validate

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ('longrun_contract.py','train_longrun.py','sample_longrun.py',
                'summarize_longrun.py','run_longrun.sh')


def runner_sources():
    return {name: sha256(HERE/name) for name in SOURCE_FILES}


def durable_save(value, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(value,f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)
    if os.name == 'posix':
        directory = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)


def preserve_incomplete_initialization(run):
    if run.exists() and any(run.iterdir()):
        allowed = {'config.json','provenance.json','initialization_report.json','status.json','latest.pt.tmp'}
        allowed |= {name+'.tmp' for name in tuple(allowed) if name.endswith('.json')}
        if any(not p.is_file() or p.name not in allowed for p in run.iterdir()):
            raise RuntimeError('Nonempty output without checkpoint contains unknown files; inspect before continuing')
        preserved = run.with_name(run.name+f'_incomplete_initialization_{time.time_ns()}')
        run.rename(preserved)
        print('LONGRUN_INCOMPLETE_INITIALIZATION_PRESERVED',preserved,flush=True)


def restore_training_state(model, checkpoint, device):
    c = checkpoint['config']
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=0)
    # Detach CPU Adam step tensors from an mmap-backed latest.pt before replacing it.
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    parameters = dict(model.named_parameters())
    if set(parameters) != set(checkpoint['ema']):
        raise ValueError('EMA parameter names do not match model')
    ema = {name: value.detach().clone().to(device) for name,value in checkpoint['ema'].items()}
    state = checkpoint['sampler']
    return optimizer, ema, state['permutation'].clone(), int(state['cursor']), int(state['epoch'])


def check_identity(checkpoint, mode, sources):
    validate(checkpoint, mode, sources)
    verify_provenance(checkpoint['provenance'])
    if checkpoint['provenance'].get('s2_sources') != s2_sources():
        raise ValueError('Original S2 code changed')
    if checkpoint['provenance'].get('runtime_versions') != runtime_versions():
        raise ValueError('Runtime differs from original S2; reactivate the original environment')
    steps = [int(state['step']) for state in checkpoint['optimizer'].get('state',{}).values() if 'step' in state]
    if not steps or max(steps) != checkpoint['step'] or min(steps) < 1:
        raise ValueError('Optimizer update count is inconsistent with checkpoint step')


def preflight(origins, run_root):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    sources = runner_sources()
    expected = json.loads((HERE/'s2_expected_checkpoints.json').read_text(encoding='utf-8'))
    report = dict(protocol=PROTOCOL, gpu=torch.cuda.get_device_name(0), sources=sources,
                  runtime_versions=runtime_versions(), branches={})
    for mode in MODES:
        origin = origins/mode/'latest.pt'
        digest = sha256(origin)
        if digest != expected[mode]['sha256'] or origin.stat().st_size != expected[mode]['bytes']:
            raise ValueError(f'{mode}: original 10k checkpoint differs from cloud S2 manifest')
        cp = _torch_load(origin); check_identity(cp,mode,sources)
        entry = dict(origin_sha256=digest,origin_bytes=origin.stat().st_size,origin_step=cp['step'])
        del cp
        latest = run_root/mode/'latest.pt'
        if latest.exists():
            cp = _torch_load(latest); check_identity(cp,mode,sources)
            if cp['config']['protocol'] != PROTOCOL or cp['provenance']['longrun']['origin_sha256'] != digest:
                raise ValueError('Existing long-run origin mismatch')
            entry['resume_step'] = cp['step']; del cp
        else:
            entry['resume_step'] = 10000
        report['branches'][mode] = entry
        print('LONGRUN_BRANCH_READY',mode,entry,flush=True)
    free = shutil.disk_usage(run_root.parent).free/2**30
    missing_bytes = sum(e['origin_bytes'] for m,e in report['branches'].items() if not (run_root/m/'latest.pt').exists())
    required = max(8.0, missing_bytes/2**30 + 8.0)
    if free < required:
        raise RuntimeError(f'Insufficient disk: free={free:.1f}GiB required={required:.1f}GiB')
    report.update(free_disk_gib=free,minimum_free_gib=required)
    run_root.mkdir(parents=True,exist_ok=True)
    write_json(run_root/'preflight_receipt.json',report)
    print('LONGRUN_PREFLIGHT_PASS',flush=True)


def train(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for long-run training')
    if args.until not in MILESTONES: raise ValueError('Unknown milestone')
    args.run = args.run.resolve(); args.origin = args.origin.resolve()
    if args.run == args.origin.parent or args.run/'latest.pt' == args.origin:
        raise ValueError('Long-run output must not overwrite S2')
    seed_all(19991009)
    sources = runner_sources(); path = args.run/'latest.pt'
    if path.exists():
        model, cp = load(path,'cuda','RAW')
        check_identity(cp,args.mode,sources)
        if cp['config']['protocol'] != PROTOCOL: raise ValueError('Output is not a long-run checkpoint')
        if cp['provenance']['longrun']['origin_sha256'] != sha256(args.origin):
            raise ValueError('S2 origin changed')
        config, provenance = cp['config'],cp['provenance']
    else:
        model, cp = load(args.origin,'cuda','RAW')
        check_identity(cp,args.mode,sources)
        config, provenance = promote(cp,args.mode,sha256(args.origin),sources,args.origin)
        preserve_incomplete_initialization(args.run)
    start = int(cp['step'])
    if start >= args.until:
        print(f'LONGRUN_MILESTONE_ALREADY_REACHED mode={args.mode} step={start} milestone={args.until}',flush=True)
        return
    rows, states, masks, meta = open_cache('train_ChemBI',0)
    if len(rows) != 22169: raise ValueError('Unexpected training size')
    verify_train_cache(provenance['train_text_cache_sha256'],meta['cache_sha256'])
    tokenizer = MoleculeTokenizer()
    ids = torch.tensor([tokenizer.encode(r['selfies']) for r in rows])
    values, observed = load_aligned_properties(DEFAULT_PROPERTY_DIR/'train_properties.csv',[r['id'] for r in rows],('logP','SA','TPSA'))
    values, observed = torch.from_numpy(values),torch.from_numpy(observed)
    optimizer, ema, permutation, cursor, epoch = restore_training_state(model,cp,'cuda')
    if permutation.numel() != len(rows) or not torch.equal(permutation.sort().values,torch.arange(len(rows))):
        raise ValueError('Sampler permutation corrupt')
    if not 0 <= cursor <= len(rows) or epoch < 0: raise ValueError('Invalid sampler cursor/epoch')
    report = cp['initialization_report']; del cp; gc.collect()
    args.run.mkdir(parents=True,exist_ok=True)
    write_json(args.run/'config.json',config); write_json(args.run/'provenance.json',provenance)
    write_json(args.run/'initialization_report.json',report)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    diffusion = make_diffusion(); sampler = UniformSampler(diffusion); bridge = Bridge(model,args.mode)
    model.train(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    attempt = f'{start}_{time.time_ns()}'
    print(f'LONGRUN_START mode={args.mode}; updates={start}->{args.until}; total_target={TARGET}; optimizer_and_ema_restored=True',flush=True)
    def save(step):
        # Atomic replacement needs space for an entire second checkpoint.
        size = (path if path.exists() else args.origin).stat().st_size
        if shutil.disk_usage(args.run).free < size*1.25 + 2**30:
            raise RuntimeError('Not enough disk for atomic checkpoint; latest safe checkpoint retained')
        snapshot = dict(format=FORMAT,step=step,config=config,provenance=provenance,
            initialization_report=report,model=model.state_dict(),optimizer=optimizer.state_dict(),ema=ema,
            sampler=dict(permutation=permutation,cursor=cursor,epoch=epoch),
            rng_note='Original S2 per-step/micro seeds and deterministic epoch permutations preserved')
        durable_save(snapshot,path)
        # Keep one durable hard-link per requested 100k milestone. Recovery
        # writes to latest.pt may occur more often, but retained artifacts are
        # exactly the user-requested milestone checkpoints.
        if step in MILESTONES:
            retained = args.run/'checkpoints'/f'step{step}.pt'
            retained.parent.mkdir(parents=True,exist_ok=True)
            if retained.exists():
                if sha256(retained) != sha256(path):
                    raise RuntimeError(f'Retained milestone differs: {retained}')
            else:
                os.link(path, retained)
            print(f'LONGRUN_RETAINED_CHECKPOINT mode={args.mode} step={step} path={retained}',flush=True)
        write_json(args.run/'status.json',dict(completed_steps=step,target_steps=TARGET,
            requested_milestone=args.until,completed=step==TARGET))
        print(f'LONGRUN_CHECKPOINT mode={args.mode} step={step}',flush=True)
    # Save the promoted 10k state before the first expensive update.
    if not path.exists(): save(start)
    with (args.run/f'train_{attempt}.jsonl').open('a',encoding='utf-8') as log:
        for step in range(start,args.until):
            permutation,cursor,epoch,chosen = next_batch(permutation,cursor,epoch,16,len(rows),0)
            batch = (ids[chosen],states[chosen],masks[chosen],values[chosen],observed[chosen])
            record = update(model,bridge,optimizer,ema,diffusion,sampler,batch,step,config,'cuda')
            record.update(step=step+1,epoch=epoch,attempt=attempt,
                seconds_per_step=(time.perf_counter()-started)/(step+1-start),
                max_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if step==start or (step+1)%100==0: print(json.dumps(record),flush=True)
            if (step+1)%500==0 or step+1==args.until or stop: save(step+1)
            if stop:
                print('LONGRUN_PAUSED_AFTER_SAFE_CHECKPOINT',flush=True)
                raise SystemExit(3)
    print(f'LONGRUN_TRAINING_PASS mode={args.mode} step={args.until}',flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--origins',type=Path); p.add_argument('--run-root',type=Path)
    p.add_argument('--mode',choices=MODES); p.add_argument('--origin',type=Path)
    p.add_argument('--run',type=Path); p.add_argument('--until',type=int)
    a = p.parse_args()
    if a.preflight:
        if a.origins is None or a.run_root is None: p.error('--origins and --run-root required')
        preflight(a.origins,a.run_root)
    else:
        if any(x is None for x in (a.mode,a.origin,a.run,a.until)): p.error('--mode --origin --run --until required')
        train(a)


if __name__ == '__main__': main()
