"""Generate from descriptions only. Reference molecules never enter the model."""
import argparse
import csv
import os
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MoleculeTokenizer, open_cache, prepared_data, sha256, write_json
from runtime import load_checkpoint, make_diffusion, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['test_ChemBI','train_ChemBI'], default='test_ChemBI')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--cache-limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--sample-steps', type=int, default=200)
    p.add_argument('--sample-length', type=int, default=258, help='Delivered sampling script uses 258, training uses 256')
    p.add_argument('--seed', type=int, default=121)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--raw-weights', action='store_true', help='Default uses EMA like the handoff')
    args = p.parse_args()
    if args.batch_size < 1 or args.limit < 0 or args.cache_limit < 0:
        p.error('Invalid sample limits')
    if args.sample_length < 1 or args.sample_length > 512:
        p.error('sample-length must be in [1,512]')
    if args.output.exists() or args.output.with_suffix('.metadata.json').exists():
        raise FileExistsError('Use a new output name; no overwrite of existing samples')
    manifest = prepared_data()
    tokenizer = MoleculeTokenizer()
    rows, states, masks, meta = open_cache(args.split, args.cache_limit)
    model, checkpoint = load_checkpoint(args.checkpoint, tokenizer, args.device, not args.raw_weights)
    if checkpoint['provenance']['train_csv_sha256'] != manifest['splits']['train_ChemBI']['sha256']:
        raise ValueError('Training data hash mismatch')
    train_text = checkpoint['provenance']['text_cache_identity']
    for key in ('encoder_hashes','length','dtype','attention_mask_passed','transformers_version'):
        if meta[key] != train_text[key]:
            raise ValueError(f'Train/test text feature protocol mismatch: {key}')
    seed_all(args.seed)
    model.eval()
    diffusion = make_diffusion(args.sample_steps)
    count = min(args.limit or len(rows), len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix+'.partial')
    if partial.exists():
        raise FileExistsError('Previous partial generation exists; use a new output filename')
    with partial.open('x', encoding='utf-8', newline='') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=['index','id','generated_selfies','reference_smiles','eos_found'])
        writer.writeheader()
        for start in range(0,count,args.batch_size):
            end = min(start+args.batch_size,count)
            desc = (states[start:end].to(args.device), masks[start:end].to(args.device))
            samples = diffusion.p_sample_loop(model,(end-start,args.sample_length,model.in_channels),
                device=args.device, clip_denoised=False, top_p=1.0, progress=False, desc=desc)
            predicted = model.get_logits(samples).argmax(-1).cpu().tolist()
            for i, tokens in enumerate(predicted,start):
                writer.writerow(dict(index=i,id=rows[i]['id'],generated_selfies=tokenizer.decode(tokens),
                    reference_smiles=rows[i]['smiles'], eos_found=2 in tokens))
            f.flush()
            print(f'Sampled {end}/{count}',flush=True)
    os.replace(partial,args.output)
    write_json(args.output.with_suffix('.metadata.json'),dict(checkpoint_sha256=sha256(args.checkpoint),
        train_csv_sha256=checkpoint['provenance']['train_csv_sha256'],
        vocab_sha256=checkpoint['provenance']['vocab_sha256'],
        code_sha256=checkpoint['provenance']['code_sha256'],
        text_cache_sha256=meta['cache_sha256'],
        checkpoint_step=checkpoint['step'], tiny=checkpoint['config']['tiny'], split=args.split,
        training_rows=checkpoint['config']['training_rows'],
        training_cache_limit=checkpoint['config']['cache_limit'],
        training_target_steps=checkpoint['config']['steps'],
        training_complete=checkpoint['step']==checkpoint['config']['steps'],
        qa_training=checkpoint['config']['tiny'] or checkpoint['config']['cache_limit']>0,
        csv_sha256=sha256(ROOT/'data'/(args.split+'.csv')), count=count, total_split_rows=len(rows),
        subset=count<len(rows) or args.cache_limit>0, cache_limit=args.cache_limit,
        seed=args.seed, batch_size=args.batch_size, sample_steps=args.sample_steps,
        timestep_map=diffusion.timestep_map, weights='raw' if args.raw_weights else 'EMA',
        sample_length=args.sample_length,
        decoding='original tied linear logits argmax; remove all PAD/SOS/EOS without early EOS stopping',
        sampling_noise='handoff truncated standard normal, abs(noise)<=1.0',
        output_sha256=sha256(args.output), reference_not_model_input=True))
    print(f'SAMPLE PASS: {args.output}')


if __name__ == '__main__':
    main()
