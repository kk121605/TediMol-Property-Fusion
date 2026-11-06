"""Future 1000-update engineering entry; never automatically start this."""
from __future__ import annotations

import argparse
import gc
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT, MoleculeTokenizer, open_cache, runtime_versions, sha256, write_json
from improved_diffusion.resample import UniformSampler
from property_data import load_aligned_properties
from property_runtime import DEFAULT_PROPERTY_DIR, _torch_load
from runtime import atomic_torch_save, make_diffusion, seed_all
from comparison_runtime import FORMAT, MODES, NAMES, Bridge, initialize, load, source_receipt, update


def main(expected_mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--acknowledge-new-protocol', action='store_true',
        help='Explicitly authorize this NEW protocol; not a continuation of old S1')
    args = parser.parse_args()
    if expected_mode is not None and args.mode != expected_mode:
        parser.error(f'This method entry requires --mode {expected_mode}')
    if not args.acknowledge_new_protocol:
        parser.error('Training is deferred: explicitly acknowledge the new protocol before running')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.run = args.run.resolve()
    seed_all(19991009)
    rows, states, masks, cache_meta = open_cache('train_ChemBI', 0)
    assert len(rows) == 22169
    tokenizer = MoleculeTokenizer()
    ids = torch.tensor([tokenizer.encode(row['selfies']) for row in rows])
    values, observed = load_aligned_properties(DEFAULT_PROPERTY_DIR / 'train_properties.csv',
        [row['id'] for row in rows], NAMES)
    values, observed = torch.from_numpy(values), torch.from_numpy(observed)
    checkpoint_path = args.run / 'latest.pt'
    if args.parent.resolve() == checkpoint_path.resolve():
        raise ValueError('Run checkpoint must not overwrite the parent model')
    if checkpoint_path.is_file():
        model, checkpoint = load(checkpoint_path, 'cuda', 'RAW')
        config, provenance = checkpoint['config'], checkpoint['provenance']
        assert config['mode'] == args.mode and config['budget_steps'] == 1000
        assert provenance['a0_checkpoint_sha256'] == sha256(args.parent)
        assert provenance['train_text_cache_sha256'] == cache_meta['cache_sha256']
        assert provenance['runtime_versions'] == runtime_versions()
        start = checkpoint['step']
        assert 1 <= start <= 1000
        report = checkpoint['initialization_report']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0)
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema = {name: value.to('cuda') for name, value in checkpoint['ema'].items()}
        permutation, cursor = checkpoint['sampler']['permutation'], checkpoint['sampler']['cursor']
        del checkpoint
    else:
        if args.run.exists() and any(args.run.iterdir()):
            preserved = args.run.with_name(args.run.name + '_interrupted_before_checkpoint_' + str(time.time_ns()))
            args.run.rename(preserved)
            print('INCOMPLETE_COMPARISON_RUN_PRESERVED:', preserved, flush=True)
        model, config, provenance, report = initialize(args.parent, args.mode, 'cuda',
            expected_cache_sha256=cache_meta['cache_sha256'])
        provenance.update(train_text_cache_sha256=cache_meta['cache_sha256'], runtime_versions=runtime_versions())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0)
        ema = {name: p.detach().clone() for name, p in model.named_parameters()}
        permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))
        cursor, start = 0, 0
    gc.collect()
    assert config['training_rows'] == len(rows)
    args.run.mkdir(parents=True, exist_ok=True)
    log_path = args.run / 'train.jsonl'
    if start and log_path.exists() and start < 1000:
        log_path.rename(args.run / f'train_attempt_until_resume_step{start}_{time.time_ns()}.jsonl')
    write_json(args.run / 'config.json', config)
    write_json(args.run / 'provenance.json', provenance)
    write_json(args.run / 'initialization_report.json', report)
    write_json(args.run / 'status.json', {'completed_steps': start, 'budget_steps': 1000, 'completed': start == 1000})
    if start == 1000:
        print('COMPARISON_TRAINING_ALREADY_COMPLETE:', args.mode, flush=True)
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
    print(f'COMPARISON_START mode={args.mode}; parameters={sum(p.numel() for p in model.parameters()):,}; updates={start}->1000', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        for step in range(start, 1000):
            chosen = permutation[cursor:cursor + 16].tolist()
            assert len(chosen) == 16  # 1000*16 is smaller than the full train split.
            cursor += 16
            batch = (ids[chosen], states[chosen], masks[chosen], values[chosen], observed[chosen])
            record = update(model, bridge, optimizer, ema, diffusion, sampler, batch, step, config, 'cuda')
            record.update(step=step+1, seconds_per_step=(time.perf_counter()-started)/(step+1-start),
                          max_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            if step == start or (step+1) % 100 == 0:
                print(json.dumps(record), flush=True)
            if (step+1) % 250 == 0 or step == 999 or stop:
                snapshot = {'format': FORMAT, 'step': step+1, 'config': config, 'provenance': provenance,
                    'initialization_report': report, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'ema': ema, 'sampler': {'permutation': permutation, 'cursor': cursor},
                    'rng_note': 'Stateless per-step/per-micro seeds; no dropout-RNG state needed for update continuation'}
                atomic_torch_save(snapshot, checkpoint_path)
                write_json(args.run / 'status.json', {'completed_steps': step+1, 'budget_steps': 1000, 'completed': step == 999})
                print(f'COMPARISON_CHECKPOINT step={step+1}', flush=True)
            if stop:
                print('COMPARISON_PAUSED_AFTER_SAFE_CHECKPOINT', flush=True)
                return 3
    print('COMPARISON_TRAINING_PASS:', args.mode, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
