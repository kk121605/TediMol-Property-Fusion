"""Immutable training contract for extending S2 without resetting learned state."""
from copy import deepcopy

PROTOCOL = 'tedimol-s2-a30-1200k-20261015-v1'
OLD_PROTOCOL = 'tedimol-s2-10k-20261012-v1'
FORMAT = 'tedimol-three-candidates-engineering-20261010-v1'
MODES = ('a0c', 'a1_prefix', 'a2_rt_numeric', 'a3_adaln_zero')
TARGET = 1200000
MILESTONES = tuple(range(100000, TARGET + 1, 100000))
FIXED = dict(lr=1e-5, lr_horizon_steps=TARGET, batch_size=16, microbatch=2,
             seed=19991009, ema_rate=0.9999, grad_clip=2.4, weight_decay=0.0,
             precision='float32', training_rows=22169, data_permutation_seed=0)


def validate(checkpoint, mode, sources=None):
    required = {'format','step','config','provenance','model','optimizer','ema','sampler','initialization_report'}
    if not required.issubset(checkpoint):
        raise ValueError(f'Incomplete checkpoint: {sorted(required-set(checkpoint))}')
    c = checkpoint['config']
    if checkpoint['format'] != FORMAT or mode not in MODES or c['mode'] != mode:
        raise ValueError('Wrong model format/mode')
    if c.get('property_names') != ['logP','SA','TPSA']:
        raise ValueError('Property order changed')
    expected_fixed = dict(FIXED)
    if c.get('protocol') == OLD_PROTOCOL:
        expected_fixed['lr_horizon_steps'] = 300000
    for name, value in expected_fixed.items():
        if c.get(name) != value:
            raise ValueError(f'Training setting changed: {name}={c.get(name)!r}, expected {value!r}')
    if not {'permutation','cursor','epoch'}.issubset(checkpoint['sampler']):
        raise ValueError('Incomplete sampler state')
    if c.get('protocol') == OLD_PROTOCOL:
        if checkpoint['step'] != 10000 or c.get('budget_steps') != 10000:
            raise ValueError('Origin must be the completed S2 10k checkpoint')
    elif c.get('protocol') == PROTOCOL:
        if c.get('budget_steps') != TARGET or not 10000 <= checkpoint['step'] <= TARGET:
            raise ValueError('Invalid long-run target/step')
        receipt = checkpoint['provenance'].get('longrun', {})
        if receipt.get('origin_step') != 10000 or receipt.get('protocol') != PROTOCOL:
            raise ValueError('Missing long-run origin record')
        if sources is not None and receipt.get('sources') != sources:
            raise ValueError('Long-run source changed; restore the deployed snapshot')
    else:
        raise ValueError('Unknown training protocol')


def promote(checkpoint, mode, origin_sha256, sources, origin_path):
    validate(checkpoint, mode)
    if checkpoint['config']['protocol'] != OLD_PROTOCOL:
        raise ValueError('Only an original S2 checkpoint can be promoted')
    config, provenance = deepcopy(checkpoint['config']), deepcopy(checkpoint['provenance'])
    config.update(protocol=PROTOCOL, budget_steps=TARGET, lr_horizon_steps=TARGET,
                  evaluation='original S2 seen-train panel learning curve; not held-out validation/test')
    provenance['longrun'] = dict(protocol=PROTOCOL, origin_step=10000, origin_sha256=origin_sha256,
        origin_path=str(origin_path), sources=sources,
        continuation='RAW model + AdamW + EMA + sampler; A30 1200k LR horizon')
    return config, provenance
