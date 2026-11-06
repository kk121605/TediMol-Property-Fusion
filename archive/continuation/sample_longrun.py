"""Use the unchanged S2 sampler after validating the new continuation contract."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

from common import sha256, write_json
from property_runtime import _torch_load
import sample_s2
from longrun_contract import PROTOCOL
from train_longrun import check_identity, runner_sources


def reusable(output, checkpoint_hash, step, weights, sources):
    metadata = output.with_suffix('.metadata.json')
    if not output.is_file() or not metadata.is_file(): return False
    try:
        m = json.loads(metadata.read_text(encoding='utf-8'))
        return (m.get('checkpoint_sha256') == checkpoint_hash and m.get('checkpoint_step') == step
                and m.get('weights') == weights and m.get('longrun_sources') == sources
                and m.get('output_sha256') == sha256(output))
    except (ValueError,OSError): return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--weights',choices=('RAW','EMA'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-steps',type=int,required=True)
    a = p.parse_args(); sources = runner_sources()
    cp = _torch_load(a.checkpoint); check_identity(cp,cp['config']['mode'],sources)
    if cp['config']['protocol'] != PROTOCOL or cp['step'] != a.expected_steps:
        raise ValueError('Long-run sampling checkpoint step/protocol mismatch')
    digest = sha256(a.checkpoint); del cp; gc.collect()
    if reusable(a.output,digest,a.expected_steps,a.weights,sources):
        print('LONGRUN_SAMPLE_ALREADY_COMPLETE',a.output,flush=True); return
    # The historical sampler's only protocol restriction is its module constant.
    # Keep its original source receipt, panel, seeds, decoding and 200 diffusion steps.
    # Do not edit sample_s2.py or any trained source file on disk.
    sample_s2.S2_PROTOCOL = PROTOCOL
    sample_s2.main()
    metadata = a.output.with_suffix('.metadata.json')
    m = json.loads(metadata.read_text(encoding='utf-8'))
    if m['checkpoint_sha256'] != digest: raise ValueError('Checkpoint changed while sampling')
    m.update(protocol=PROTOCOL,longrun_sources=sources,
             evaluation='Original S2 seen-train panel; learning curve only, not held-out evidence')
    write_json(metadata,m)
    print('LONGRUN_SAMPLE_PASS',a.output,flush=True)


if __name__ == '__main__': main()
