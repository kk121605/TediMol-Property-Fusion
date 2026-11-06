"""Audited NumPy seed compatibility for the original A30 continuation runner."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys


def seed_function(source, namespace):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'seed_all')
    matches = [n for n in ast.walk(node) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == 'np.random.seed']
    if len(matches) != 1 or ast.unparse(matches[0].args[0]) != 'seed':
        raise RuntimeError('Unexpected NumPy seeding implementation; refusing compatibility patch')
    matches[0].args[0] = ast.BinOp(left=ast.Name(id='seed', ctx=ast.Load()),
                                 op=ast.Mod(), right=ast.Constant(value=2**32))
    module = ast.Module(body=[node], type_ignores=[])
    local = {}
    exec(compile(ast.fix_missing_locations(module), '<seed32-compatibility>', 'exec'), namespace, local)
    return local['seed_all']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_record(handoff, runtime_path):
    return dict(fix='numpy-seed32-v1', numpy_seed='seed % 2**32',
                python_torch_seed='unchanged', first_affected_update=521849,
                runtime_sha256=digest(runtime_path), launcher_sha256=digest(Path(__file__).resolve()),
                train_longrun_sha256=digest(handoff / 'train_longrun.py'))


def install(namespace, runtime, record):
    runtime.seed_all = seed_function(Path(runtime.__file__).read_text(encoding='utf-8'), vars(runtime))
    original_check = namespace['check_identity']
    original_save = namespace['durable_save']

    def checked(checkpoint, mode, sources):
        original_check(checkpoint, mode, sources)
        prior = checkpoint['provenance'].get('runtime_compatibility')
        if prior is not None and prior != record:
            raise ValueError('Runtime compatibility record mismatch')
        if prior is None and checkpoint['step'] >= record['first_affected_update']:
            raise ValueError('Post-boundary checkpoint lacks runtime compatibility record')

    def saved(checkpoint, path):
        updated = dict(checkpoint)
        updated['provenance'] = copy.deepcopy(checkpoint['provenance'])
        prior = updated['provenance'].get('runtime_compatibility')
        if prior is not None and prior != record:
            raise ValueError('Refusing to overwrite a different compatibility record')
        updated['provenance']['runtime_compatibility'] = record
        return original_save(updated, path)

    namespace.update(seed_all=runtime.seed_all, check_identity=checked, durable_save=saved)


def selftest(source):
    import random
    import numpy as np
    import torch
    old_node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'seed_all')
    scope = dict(random=random, np=np, torch=torch)
    exec(compile(ast.Module(body=[old_node], type_ignores=[]), '<original>', 'exec'), scope)
    old = scope['seed_all']
    fixed = seed_function(source, dict(random=random, np=np, torch=torch))
    try:
        old(19991009 + 521848 * 8192)
    except ValueError:
        print('ORIGINAL_SEED_OVERFLOW_REPRODUCED', flush=True)
    else:
        raise AssertionError('Expected original overflow was not reproduced')
    for seed in (0, 19991009, 4294961647, 2**32 - 1):
        old(seed)
        expected = random.random(), np.random.random(4).tolist(), torch.rand(4).tolist()
        fixed(seed)
        assert expected == (random.random(), np.random.random(4).tolist(), torch.rand(4).tolist())
    for step in (521848, 600000, 1048576, 1199999):
        for offset in range(0, 16, 2):
            seed = 19991009 + step * 8192 + offset
            random.seed(seed); torch.manual_seed(seed)
            expected = random.random(), torch.rand(4).tolist()
            fixed(seed)
            assert expected[0] == random.random()
            assert expected[1] == torch.rand(4).tolist()
            assert np.random.random(4).tolist() == np.random.RandomState(seed % 2**32).random(4).tolist()
            if torch.cuda.is_available():
                fixed(seed); expected_cuda = torch.rand(4, device='cuda')
                torch.cuda.manual_seed_all(seed)
                assert torch.equal(expected_cuda, torch.rand(4, device='cuda'))
    print('SEED32_COMPATIBILITY_TEST_PASS', flush=True)


def main():
    handoff = Path(__file__).resolve().parent
    import runtime
    record = audit_record(handoff, Path(runtime.__file__).resolve())
    receipt = handoff / 'seed32_receipt.json'
    if json.loads(receipt.read_text(encoding='utf-8')) != record:
        raise RuntimeError('Repair receipt differs from deployed launcher/runtime/runner')
    if sys.argv[1:] == ['--seed32-selftest']:
        selftest(Path(runtime.__file__).read_text(encoding='utf-8'))
        return
    # Preserve the original runner file and all its provenance checks.
    module = runpy.run_path(str(handoff / 'train_longrun.py'), run_name='seed32_target')
    namespace = module['main'].__globals__
    install(namespace, runtime, record)
    print('SEED32_RUNTIME_COMPATIBILITY', json.dumps(record, sort_keys=True), flush=True)
    namespace['main']()


if __name__ == '__main__':
    main()
