"""Single-GPU handoff baseline with microbatches, EMA, resumable optimizer/RNG.

The original GaussianDiffusion loss is retained. No correction or property branch.
"""
import argparse
import json
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MoleculeTokenizer, prepared_data, open_cache, write_json, runtime_versions
from model import DEFAULT_MODEL, TransformerNetModel2, LossModelAdapter
from improved_diffusion.resample import UniformSampler
from runtime import make_diffusion, seed_all, atomic_torch_save, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=ROOT / 'runs' / 'chembi_seed19991009')
    p.add_argument('--steps', type=int, default=300000, help='Fixed LR schedule endpoint')
    p.add_argument('--stop-after', type=int, default=0, help='Stop at this global step without changing LR schedule')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=19991009)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--resume', type=Path)
    p.add_argument('--cache-limit', type=int, default=0)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--tiny', action='store_true', help='Validation only; never paper evidence')
    args = p.parse_args()
    if min(args.steps,args.batch_size,args.microbatch,args.save_every,args.log_every) < 1 or args.lr <= 0 or args.cache_limit < 0 or args.stop_after < 0:
        p.error('Invalid nonpositive training argument')
    if args.stop_after > args.steps:
        p.error('stop-after cannot exceed steps')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.device == 'cpu' and not args.tiny:
        raise ValueError('Full model training belongs on AutoDL; CPU permitted only with --tiny')
    seed_all(args.seed)
    manifest = prepared_data()
    tokenizer = MoleculeTokenizer()
    rows, states, masks, cache_meta = open_cache('train_ChemBI', args.cache_limit)
    indices = list(range(len(rows)))
    if not indices:
        raise ValueError('No training samples')
    ids = np.asarray([tokenizer.encode(rows[i]['selfies']) for i in indices], dtype=np.int64)
    model_config = dict(DEFAULT_MODEL)
    if args.tiny:
        model_config.update(hidden_size=32, num_attention_heads=4, num_hidden_layers=1, intermediate_size=64, model_channels=16)
    config = dict(model=model_config, max_length=256, steps=args.steps, lr=args.lr, seed=args.seed,
        batch_size=args.batch_size, microbatch=args.microbatch, ema_rate=0.9999, weight_decay=0.0, grad_clip=2.4,
        precision='float32', torch_version=torch.__version__, runtime_versions=runtime_versions(), device_type=args.device,
        cache_limit=args.cache_limit, training_rows=len(ids), tiny=args.tiny,
        data_order='original DistributedSampler single-rank seed=0, epoch never advanced; last partial batch retained',
        text_mask='original encoder unmasked, cross-attention raw 0/1 additive bias',
        protocol='senior TediMol phase1, original vocab323 and original truncation, no row exclusions')
    receipt = provenance(manifest, cache_meta)
    model = TransformerNetModel2(len(tokenizer), **model_config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    ema = {k: v.detach().clone() for k,v in model.named_parameters()}
    # Original get_dataloader recreates DistributedSampler without set_epoch:
    # its default seed=0 and epoch=0 repeat the same permutation every epoch.
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(len(ids), generator=generator)
    cursor = 0
    epoch = 0
    start_step = 0
    args.run = args.run.resolve()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['format'] != 'tedimol-senior-v2-vocab323' or checkpoint['config'] != config or checkpoint['provenance'] != receipt:
            raise ValueError('Resume mismatch in config, code, vocabulary, data or text cache')
        if args.resume.resolve().parent != args.run:
            raise ValueError('Resume in the original --run directory')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        ema = {k:v.to(args.device) for k,v in checkpoint['ema'].items()}
        start_step = checkpoint['step']
        permutation = checkpoint['sampler']['permutation']
        cursor, epoch = checkpoint['sampler']['cursor'], checkpoint['sampler']['epoch']
        generator.set_state(checkpoint['sampler']['generator'])
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    elif args.run.exists() and any(args.run.iterdir()):
        raise ValueError('Run directory is not empty. Use --resume or a new --run')
    args.run.mkdir(parents=True, exist_ok=True)
    # Preserve the previous log as a separate attempt. A crash can leave log
    # steps newer than latest.pt; they must not look like duplicate resumed steps.
    if args.resume and (args.run / 'train.jsonl').exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = args.run / f'train_before_resume_{stamp}.jsonl'
        (args.run / 'train.jsonl').rename(archive)
        with (args.run / 'resume_events.jsonl').open('a', encoding='utf-8') as events:
            events.write(json.dumps(dict(resume_from_step=start_step, archived_log=archive.name,
                note='Only steps <= resume_from_step belong to the checkpoint; later old lines are abandoned'))+'\n')
    write_json(args.run / 'config.json', config)
    write_json(args.run / 'provenance.json', receipt)
    diffusion = make_diffusion()
    schedule_sampler = UniformSampler(diffusion)
    loss_model = LossModelAdapter(model)
    print(f'PARAMETERS={sum(v.numel() for v in model.parameters()):,}; samples={len(ids)}; microbatch={args.microbatch}', flush=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save(step):
        snapshot = dict(format='tedimol-senior-v2-vocab323', step=step, config=config, provenance=receipt,
            model=model.state_dict(), optimizer=opt.state_dict(), ema=ema,
            sampler=dict(permutation=permutation, cursor=cursor, epoch=epoch, generator=generator.get_state()),
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else []))
        atomic_torch_save(snapshot, args.run / 'latest.pt')
        write_json(args.run / 'status.json', dict(completed_steps=step, target_steps=args.steps,
            protocol=config['protocol'], completed=step==args.steps, tiny=args.tiny))
        print(f'CHECKPOINT step={step}: {args.run / "latest.pt"}', flush=True)

    end_step = args.stop_after or args.steps
    if end_step <= start_step:
        raise ValueError('Requested endpoint is already reached; increase --stop-after or omit it')
    started = time.perf_counter()
    model.train()
    with (args.run / 'train.jsonl').open('a', encoding='utf-8') as log:
        for step in range(start_step, end_step):
            if cursor == len(ids):
                epoch += 1
                cursor = 0
            take = min(args.batch_size, len(ids)-cursor)
            chosen = permutation[cursor:cursor+take].tolist()
            cursor += take
            batch_ids = torch.from_numpy(ids[chosen])
            source_indices = [indices[j] for j in chosen]
            text = states[source_indices]
            mask = masks[source_indices]
            lr = args.lr * (1-step/args.steps)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for offset in range(0, take, args.microbatch):
                stop = min(offset+args.microbatch, take)
                mol = batch_ids[offset:stop].to(args.device)
                micro = (mol, text[offset:stop].to(args.device), mask[offset:stop].to(args.device), mol)
                ts, weights = schedule_sampler.sample(stop-offset,args.device)
                terms = diffusion.training_losses(loss_model, micro, ts)
                loss = (terms['loss']*weights).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}; previous checkpoint remains intact')
                weight = (stop-offset)/take
                (loss*weight).backward()
                total += float(loss.detach())*weight
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.4, error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    ema[name].mul_(0.9999).add_(param, alpha=0.0001)
            record = dict(step=step+1, epoch=epoch, loss=total, lr=lr, grad_norm=float(grad),
                seconds_per_step=(time.perf_counter()-started)/(step+1-start_step),
                max_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else 0)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            log.flush()
            if step == start_step or (step+1) % args.log_every == 0:
                print(json.dumps(record), flush=True)
            if (step+1) % args.save_every == 0 or step+1 == end_step or stop_requested:
                save(step+1)
            if stop_requested:
                print('Stopped safely at an optimizer boundary.', flush=True)
                break


if __name__ == '__main__':
    main()
