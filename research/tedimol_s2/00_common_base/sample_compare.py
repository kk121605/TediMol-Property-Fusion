"""Future engineering panel; supports A0c and all three property candidates."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from common import open_cache, MoleculeTokenizer, sha256, write_json
from property_data import load_aligned_properties
from property_runtime import DEFAULT_PROPERTY_DIR
from runtime import make_diffusion, seed_all
from comparison_runtime import NAMES, load, source_receipt, verify_train_cache
from property_metrics import compute_generated_properties

SWEEP_INDICES = (0, 100, 1000, 10000)
FIELDS = ['regime', 'source_index', 'id', 'sweep_property', 'z', 'generated_selfies',
          'generated_smiles', 'valid', 'exact_reference', 'morgan_similarity']
for name in NAMES:
    FIELDS += [f'target_{name}', f'generated_{name}', f'abs_error_{name}']


def preserve(path):
    if path.exists():
        dest = path.with_name(path.name + f'.preserved_{time.time_ns()}')
        path.rename(dest)
        print('INCOMPLETE_OR_MISMATCHED_SAMPLE_PRESERVED:', dest, flush=True)


def evaluate_selfies(value):
    # Failed/empty structures are reportable generated outcomes, not a reason
    # to lose an entire comparison after training all modes.
    try:
        smiles, valid, computed = compute_generated_properties(value, list(NAMES))
        molecule = Chem.MolFromSmiles(smiles) if valid else None
        valid = bool(molecule is not None and molecule.GetNumAtoms() > 0 and
                     all(math.isfinite(computed[name]) for name in NAMES))
        return smiles, valid, computed if valid else {}
    except (TypeError, ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return '', False, {}


def main(expected_mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=('RAW', 'EMA'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Full engineering panel requires CUDA; use test_three_methods_cpu.py locally')
    output = args.output
    metadata_path = output.with_suffix('.metadata.json')
    partial = output.with_suffix('.csv.partial')
    checkpoint_hash = sha256(args.checkpoint)
    model, checkpoint = load(args.checkpoint, 'cuda', args.weights)
    if expected_mode is not None and checkpoint['config']['mode'] != expected_mode:
        raise ValueError(f'Expected {expected_mode} checkpoint')
    rows, states, masks, cache_meta = open_cache('train_ChemBI', 0)
    verify_train_cache(checkpoint['provenance']['train_text_cache_sha256'], cache_meta['cache_sha256'])
    identity = {'checkpoint_sha256': checkpoint_hash, 'weights': args.weights,
        'source_sha256': source_receipt(), 'sample_seed': 121, 'sample_steps': 200,
        'sample_length': 258, 'reference_indices': list(range(32)), 'sweep_indices': list(SWEEP_INDICES),
        'sweep_z': [-1, 0, 1], 'text_cache_sha256': cache_meta['cache_sha256'],
        'split': 'train_ChemBI', 'seen_during_training': True, 'not_held_out_evaluation': True}
    if output.is_file() and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous.get('identity') == identity and previous.get('output_sha256') == sha256(output):
            with output.open(encoding='utf-8', newline='') as stream:
                completed_rows = list(csv.DictReader(stream))
            if len(completed_rows) == 68:
                preserve(partial)
                print('COMPARISON_SAMPLE_ALREADY_COMPLETE:', output, flush=True)
                return
    preserve(output)
    preserve(metadata_path)
    preserve(partial)
    assert checkpoint['step'] == 1000
    mode = checkpoint['config']['mode']
    values, observed = load_aligned_properties(DEFAULT_PROPERTY_DIR / 'train_properties.csv',
        [row['id'] for row in rows], NAMES)
    means = np.asarray(checkpoint['config']['property_means'], dtype=np.float32)
    stds = np.asarray(checkpoint['config']['property_stds'], dtype=np.float32)
    tokenizer = MoleculeTokenizer()
    model.eval()
    diffusion = make_diffusion(200)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with partial.open('x', encoding='utf-8', newline='') as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        def generate(indices, targets, regime, property_name='', z='', noise=None, seed=121):
            seed_all(seed)
            desc = (states[indices].to('cuda'), masks[indices].to('cuda'))
            kwargs = {} if mode == 'a0c' else {
                'chemical_metric': torch.from_numpy(targets).to('cuda'),
                'property_observed': torch.ones(len(indices), 3, dtype=torch.bool, device='cuda')}
            sample = diffusion.p_sample_loop(model, (len(indices), 258, model.in_channels), noise=noise,
                device='cuda', clip_denoised=False, top_p=1.0, progress=False, desc=desc, model_kwargs=kwargs)
            tokens = model.get_logits(sample).argmax(-1).cpu().tolist()
            for local, (index, sequence) in enumerate(zip(indices, tokens)):
                selfies = tokenizer.decode(sequence)
                smiles, valid, computed = evaluate_selfies(selfies)
                reference = Chem.MolFromSmiles(rows[index]['smiles'])
                generated = Chem.MolFromSmiles(smiles) if valid else None
                exact = bool(valid and Chem.MolToSmiles(reference, isomericSmiles=True) == smiles)
                similarity = DataStructs.TanimotoSimilarity(generator.GetFingerprint(reference),
                    generator.GetFingerprint(generated)) if valid else math.nan
                row = {'regime': regime, 'source_index': index, 'id': rows[index]['id'],
                    'sweep_property': property_name, 'z': z, 'generated_selfies': selfies,
                    'generated_smiles': smiles, 'valid': int(valid),
                    'exact_reference': int(exact), 'morgan_similarity': similarity}
                for k, name in enumerate(NAMES):
                    row[f'target_{name}'] = float(targets[local, k])
                    value = computed.get(name, math.nan)
                    row[f'generated_{name}'] = value
                    row[f'abs_error_{name}'] = abs(value-float(targets[local, k]))
                writer.writerow(row)
            stream.flush()
        for start in range(0, 32, 8):
            indices = list(range(start, start+8))
            generate(indices, values[indices], 'reference', seed=121+start)
            print(f'COMPARISON_REFERENCE {start+8}/32', flush=True)
        for index in SWEEP_INDICES:
            seed = 121+index
            seed_all(seed)
            noise = torch.randn(1, 258, model.in_channels, device='cuda')
            for k, name in enumerate(NAMES):
                for z in (-1, 0, 1):
                    target = values[index:index+1].copy()
                    target[0, k] = means[k]+z*stds[k]
                    generate([index], target, 'sweep', name, z, noise=noise, seed=seed)
                print(f'COMPARISON_SWEEP index={index}; property={name}', flush=True)
    os.replace(partial, output)
    write_json(metadata_path, {'identity': identity, 'mode': mode, 'checkpoint_step': 1000,
        'output_sha256': sha256(output), 'elapsed_seconds': time.perf_counter()-started,
        'max_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
        'reference_smiles_direct_model_input': False,
        'reference_properties_used_as_condition': mode != 'a0c',
        'sweep_is_stress_test': 'other attributes fixed at reference; joint target may be OOD/conflicting',
        'sampling_noise': 'reference: normal initial noise; sweep: explicit normal; all reverse-step noise truncated by legacy top_p=1.0',
        'morgan_similarity_is_structure_proxy_not_text_consistency': True,
        'not_a_scientific_superiority_claim': True})
    print('COMPARISON_SAMPLE_PASS:', mode, args.weights, output, flush=True)
    del model, checkpoint
    gc.collect()


if __name__ == '__main__':
    main()
