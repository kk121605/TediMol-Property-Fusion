"""S2 resumable 10k training for A0c/A1/A2/A3; separate from historical S1."""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time
from pathlib import Path

import torch

from common import MoleculeTokenizer, open_cache, runtime_versions, sha256, write_json
from improved_diffusion.resample import UniformSampler
from property_data import load_aligned_properties
from property_runtime import DEFAULT_PROPERTY_DIR, _torch_load
from runtime import atomic_torch_save, make_diffusion, seed_all
from comparison_runtime import FORMAT, MODES, NAMES, Bridge, initialize, load, update

HERE = Path(__file__).resolve().parent
S2_PROTOCOL = 'tedimol-s2-10k-20261012-v1'


def s2_sources():
    return {n: sha256(HERE / n) for n in ('train_s2.py', 'sample_s2.py', 'run_s2_10k.sh')}


def next_batch(permutation, cursor, epoch, count, total, seed):
    if cursor + count > total:
        epoch += 1
        permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed + epoch))
        cursor = 0
    chosen = permutation[cursor:cursor + count].tolist()
    if len(chosen) != count:
        raise RuntimeError('sampler could not produce a complete batch')
    return permutation, cursor + count, epoch, chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--acknowledge-new-protocol', action='store_true')
    args = p.parse_args()
    if not args.acknowledge_new_protocol:
        p.error('S2 is a new protocol; pass --acknowledge-new-protocol explicitly')
    if args.steps < 1 or not torch.cuda.is_available():
        raise RuntimeError('S2 requires positive steps and CUDA')
    args.run = args.run.resolve()
    seed_all(19991009)
    rows, states, masks, cache_meta = open_cache('train_ChemBI', 0)
    if len(rows) != 22169:
        raise RuntimeError(f'unexpected train rows: {len(rows)}')
    tokenizer = MoleculeTokenizer()
    ids = torch.tensor([tokenizer.encode(row['selfies']) for row in rows])
    values, observed = load_aligned_properties(DEFAULT_PROPERTY_DIR / 'train_properties.csv',
        [row['id'] for row in rows], NAMES)
    values, observed = torch.from_numpy(values), torch.from_numpy(observed)
    checkpoint_path = args.run / 'latest.pt'
    if checkpoint_path.resolve() == args.parent.resolve():
        raise ValueError('S2 must not overwrite its parent')
    if checkpoint_path.is_file():
        model, checkpoint = load(checkpoint_path, 'cuda', 'RAW')
        config, provenance = checkpoint['config'], checkpoint['provenance']
        if config['mode'] != args.mode or config['budget_steps'] != args.steps:
            raise RuntimeError('existing S2 checkpoint does not match mode/steps')
        if config.get('protocol') != S2_PROTOCOL or provenance.get('s2_sources') != s2_sources():
            raise RuntimeError('S2 protocol/source changed; restore its trained snapshot')
        if provenance['a0_checkpoint_sha256'] != sha256(args.parent):
            raise RuntimeError('S2 parent checkpoint changed')
        if provenance['train_text_cache_sha256'] != cache_meta['cache_sha256']:
            raise RuntimeError('S2 text cache changed')
        if provenance['runtime_versions'] != runtime_versions():
            raise RuntimeError('S2 runtime versions changed')
        start = int(checkpoint['step'])
        if not 1 <= start <= args.steps:
            raise RuntimeError('invalid checkpoint step')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0)
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema = {name: value.to('cuda') for name, value in checkpoint['ema'].items()}
        permutation = checkpoint['sampler']['permutation']
        cursor = int(checkpoint['sampler']['cursor'])
        epoch = int(checkpoint['sampler']['epoch'])
        report = checkpoint['initialization_report']
        del checkpoint
    else:
        if args.run.exists() and any(args.run.iterdir()):
            preserved = args.run.with_name(args.run.name + '_preserved_' + str(time.time_ns()))
            args.run.rename(preserved)
            print('S2_INCOMPLETE_RUN_PRESERVED:', preserved, flush=True)
        model, config, provenance, report = initialize(args.parent, args.mode, 'cuda',
            expected_cache_sha256=cache_meta['cache_sha256'])
        config['budget_steps'] = args.steps
        config['protocol'] = S2_PROTOCOL
        config['sampler_policy'] = 'drop final incomplete batch; next epoch permutation uses seed=epoch'
        config['evaluation'] = 'single-seed seen-A0-train engineering panel, not validation/test'
        provenance['s2_sources'] = s2_sources()
        provenance.update(train_text_cache_sha256=cache_meta['cache_sha256'], runtime_versions=runtime_versions())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0)
        ema = {name: p.detach().clone() for name, p in model.named_parameters()}
        permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))
        cursor, epoch, start = 0, 0, 0
    args.run.mkdir(parents=True, exist_ok=True)
    write_json(args.run / 'config.json', config)
    write_json(args.run / 'provenance.json', provenance)
    write_json(args.run / 'initialization_report.json', report)
    write_json(args.run / 'status.json', {'completed_steps': start, 'budget_steps': args.steps, 'completed': start == args.steps})
    if start == args.steps:
        print('S2_TRAINING_ALREADY_COMPLETE:', args.mode, flush=True)
        return 0
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    diffusion = make_diffusion()
    sampler = UniformSampler(diffusion)
    bridge = Bridge(model, args.mode)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f'S2_START mode={args.mode}; parameters={sum(p.numel() for p in model.parameters()):,}; updates={start}->{args.steps}', flush=True)
    log_path = args.run / 'train.jsonl'
    if start and log_path.exists():
        log_path.rename(args.run / f'train_attempt_until_resume_step{start}_{time.time_ns()}.jsonl')
    with log_path.open('a', encoding='utf-8') as log:
        for step in range(start, args.steps):
            permutation, cursor, epoch, chosen = next_batch(permutation, cursor, epoch, 16, len(rows), 0)
            batch = (ids[chosen], states[chosen], masks[chosen], values[chosen], observed[chosen])
            record = update(model, bridge, optimizer, ema, diffusion, sampler, batch, step, config, 'cuda')
            record.update(step=step + 1, epoch=epoch,
                seconds_per_step=(time.perf_counter() - started) / (step + 1 - start),
                max_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            if step == start or (step + 1) % 100 == 0:
                print(json.dumps(record), flush=True)
            if (step + 1) % 500 == 0 or step == args.steps - 1 or stop:
                snapshot = {'format': FORMAT, 'step': step + 1, 'config': config, 'provenance': provenance,
                    'initialization_report': report, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'ema': ema, 'sampler': {'permutation': permutation, 'cursor': cursor, 'epoch': epoch},
                    'rng_note': 'Stateless per-step/per-micro seeds; deterministic epoch permutations'}
                atomic_torch_save(snapshot, checkpoint_path)
                write_json(args.run / 'status.json', {'completed_steps': step + 1, 'budget_steps': args.steps, 'completed': step + 1 == args.steps})
                print(f'S2_CHECKPOINT mode={args.mode} step={step + 1}', flush=True)
            if stop:
                print('S2_PAUSED_AFTER_SAFE_CHECKPOINT', flush=True)
                return 3
    print('S2_TRAINING_PASS:', args.mode, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

