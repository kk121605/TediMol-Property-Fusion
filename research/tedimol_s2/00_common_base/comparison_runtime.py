"""NEW local three-candidate protocol, not the historical S1 implementation."""
from __future__ import annotations

from pathlib import Path
import torch

from common import MoleculeTokenizer, sha256, source_hashes
from model import TransformerNetModel2
from property_runtime import (
    DEFAULT_PROPERTY_DIR, _torch_load, build_property_model, initialize_from_a0,
    property_protocol_receipt, validate_a0_checkpoint, property_source_hashes,
)

HERE = Path(__file__).resolve().parent
FORMAT = 'tedimol-three-candidates-engineering-20261010-v1'
MODES = ('a0c', 'a1_prefix', 'a2_rt_numeric', 'a3_adaln_zero')
NAMES = ('logP', 'SA', 'TPSA')
FILES = ('comparison_runtime.py', 'train_compare.py', 'sample_compare.py')


def source_receipt():
    return {name: sha256(HERE / name) for name in FILES}


def construct(mode, model_config, means, stds, bottleneck=256):
    if mode not in MODES:
        raise ValueError(mode)
    tokenizer = MoleculeTokenizer()
    if mode == 'a0c':
        return TransformerNetModel2(len(tokenizer), **model_config)
    return build_property_model(len(tokenizer), model_config,
        property_mode=mode, property_names=NAMES, means=means, stds=stds, property_bottleneck=bottleneck)


def initialize(parent_path, mode, device, expected_cache_sha256=None):
    parent = _torch_load(parent_path)
    audit = validate_a0_checkpoint(parent, MoleculeTokenizer())
    if expected_cache_sha256 is not None:
        verify_train_cache(parent['provenance']['text_cache_identity']['cache_sha256'], expected_cache_sha256)
    _, means, stds, receipt = property_protocol_receipt(DEFAULT_PROPERTY_DIR, NAMES)
    model = construct(mode, parent['config']['model'], means, stds)
    if mode == 'a0c':
        model.load_state_dict(parent['model'], strict=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(parent['ema'][name])
        load_report = {'initial_weights': 'ema', 'new_parameter_count': 0,
            'model_parameter_count': sum(p.numel() for p in model.parameters())}
    else:
        load_report = initialize_from_a0(model, parent, init_weights='ema')
    config = {
        'protocol': FORMAT, 'mode': mode, 'model': parent['config']['model'],
        'property_names': list(NAMES), 'property_means': means, 'property_stds': stds,
        'property_bottleneck': 256, 'initial_weights': 'A0 final EMA; fresh optimizer',
        'budget_steps': 1000, 'lr_horizon_steps': 300000, 'lr': 1e-5,
        'batch_size': 16, 'microbatch': 2, 'seed': 19991009, 'ema_rate': 0.9999,
        'grad_clip': 2.4, 'weight_decay': 0.0, 'precision': 'float32',
        'training_rows': 22169, 'data_permutation_seed': 0,
        'rng': 'seed+step*8192+micro_offset before each micro; matched pre-forward noise, dropout shapes differ',
        'evaluation': 'fixed seen-train engineering panel; not held-out validation or test',
        'comparison_boundary': 'unequal parameter counts, A1/A2 prefix lengths and positional shifts; engineering only',
        'property_encoder_metadata': model.property_metadata() if mode != 'a0c' else None,
    }
    provenance = {'a0_checkpoint_sha256': sha256(parent_path), 'a0_step': parent['step'],
        'baseline_code_sha256': parent['provenance']['code_sha256'],
        'three_candidate_property_sources': property_source_hashes(), 'comparison_sources': source_receipt(),
        'property_data': receipt, 'baseline_source_audit': audit}
    model.to(device)
    del parent
    return model, config, provenance, load_report


def verify_provenance(provenance):
    current = source_hashes()
    baseline = provenance['baseline_code_sha256']
    changed = [name for name, digest in baseline.items() if current.get(name) != digest]
    if changed:
        raise ValueError(f'Baseline backbone/diffusion source changed: {changed}')
    if provenance['comparison_sources'] != source_receipt():
        raise ValueError('Three-candidate source changed; use the trained source snapshot')
    if provenance['three_candidate_property_sources'] != property_source_hashes():
        raise ValueError('Three-candidate property implementation changed')
    _, _, _, receipt = property_protocol_receipt(DEFAULT_PROPERTY_DIR, NAMES)
    if provenance['property_data'] != receipt:
        raise ValueError('Property protocol changed')


def verify_train_cache(expected, actual):
    if expected != actual:
        raise ValueError('Training text cache differs from the recorded parent/checkpoint cache')


def load(path, device, weights='RAW'):
    checkpoint = _torch_load(path)
    if checkpoint['format'] != FORMAT:
        raise ValueError('Wrong checkpoint format')
    verify_provenance(checkpoint['provenance'])
    config = checkpoint['config']
    model = construct(config['mode'], config['model'], config['property_means'], config['property_stds'],
                      config.get('property_bottleneck', 256))
    model.load_state_dict(checkpoint['model'], strict=True)
    if weights == 'EMA':
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(checkpoint['ema'][name])
    elif weights != 'RAW':
        raise ValueError(weights)
    return model.to(device), checkpoint


class Bridge:
    def __init__(self, model, mode):
        self.module, self.mode = model, mode
        self.values = self.observed = None

    def __call__(self, *args, **kwargs):
        if self.mode != 'a0c':
            if self.values is None:
                raise RuntimeError('Missing microbatch property condition')
            kwargs.update(chemical_metric=self.values, property_observed=self.observed)
        return self.module(*args, **kwargs)


def update(model, bridge, optimizer, ema, diffusion, sampler, batch, step, config, device):
    """One optimizer update, shared by the three candidates plus A0c."""
    from runtime import seed_all
    molecules, states, masks, values, observed = batch
    take = len(molecules)
    lr = config['lr'] * (1 - step / config['lr_horizon_steps'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for offset in range(0, take, config['microbatch']):
        end = min(offset + config['microbatch'], take)
        seed_all(config['seed'] + step * 8192 + offset)
        mol = molecules[offset:end].to(device)
        micro = (mol, states[offset:end].to(device), masks[offset:end].to(device), mol)
        times, weights = sampler.sample(end - offset, device)
        bridge.values = values[offset:end].to(device)
        bridge.observed = observed[offset:end].to(device)
        try:
            loss = (diffusion.training_losses(bridge, micro, times)['loss'] * weights).mean()
        finally:
            bridge.values = bridge.observed = None
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
        weight = (end - offset) / take
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    groups = {}
    for prefix in ('property_encoder.', 'property_adapters.'):
        groups[prefix] = sum(float(p.grad.detach().float().square().sum())
            for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)**0.5
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            ema[name].mul_(config['ema_rate']).add_(parameter, alpha=1-config['ema_rate'])
    return {'loss': total, 'lr': lr, 'grad_norm': float(grad), 'property_group_gradient_l2': groups}
