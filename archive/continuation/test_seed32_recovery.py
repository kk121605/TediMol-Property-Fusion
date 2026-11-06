import ast
from pathlib import Path
import random
import unittest
import copy
import tempfile
import types
import numpy as np
import torch
from seed32_recovery import seed_function, install

SOURCE = (Path(__file__).resolve().parent.parent / '12_三架构融合代码与学习' /
          '00_共用第一阶段底座与训练工具' / 'runtime.py').read_text(encoding='utf-8')


def original_seed(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'seed_all')
    scope = dict(random=random, np=np, torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original>', 'exec'), scope)
    return scope['seed_all']


def draws():
    return random.random(), np.random.random(8).tolist(), torch.rand(8).tolist()


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.fixed = seed_function(SOURCE, dict(random=random, np=np, torch=torch))

    def test_original_reproduces_reported_failure(self):
        old = original_seed(SOURCE)
        old(19991009 + 521847 * 8192 + 14)
        with self.assertRaises(ValueError):
            old(19991009 + 521848 * 8192)

    def test_all_streams_unchanged_below_boundary(self):
        old = original_seed(SOURCE)
        for seed in (0, 19991009, 19991009 + 521847 * 8192 + 14, 2**32 - 1):
            old(seed)
            expected = draws()
            self.fixed(seed)
            self.assertEqual(expected, draws())

    def test_crosses_boundary_through_target_preserving_python_and_torch(self):
        for step in (521848, 600000, 1048576, 1199999):
            for offset in range(0, 16, 2):
                seed = 19991009 + step * 8192 + offset
                random.seed(seed)
                torch.manual_seed(seed)
                expected = (random.random(), torch.rand(8).tolist())
                expected_numpy = np.random.RandomState(seed % 2**32).random(8).tolist()
                self.fixed(seed)
                actual = draws()
                self.assertEqual(expected, (actual[0], actual[2]))
                self.assertEqual(expected_numpy, actual[1])

    def test_unexpected_implementation_rejected(self):
        with self.assertRaises(RuntimeError):
            seed_function(SOURCE.replace('np.random.seed(seed)', 'np.random.seed(seed + 1)'), {})

    def test_guard_preserves_original_failures_and_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runtime.py'
            path.write_text(SOURCE, encoding='utf-8')
            runtime = types.ModuleType('runtime')
            runtime.__dict__.update(__file__=str(path), random=random, np=np, torch=torch)
            def check(cp, mode, sources):
                if cp.get('invalid'):
                    raise ValueError('original provenance failure')
            namespace = dict(check_identity=check, durable_save=torch.save)
            record = dict(first_affected_update=521849, fix='test')
            install(namespace, runtime, record)
            cp = dict(step=521500, provenance=dict(original='retained'))
            namespace['check_identity'](cp, 'a0c', {})
            with self.assertRaisesRegex(ValueError, 'original provenance'):
                namespace['check_identity'](dict(cp, invalid=True), 'a0c', {})
            namespace['durable_save'](cp, Path(tmp) / 'latest.pt')
            saved = torch.load(Path(tmp) / 'latest.pt', weights_only=True)
            self.assertEqual(cp['provenance'], dict(original='retained'))
            self.assertEqual(saved['provenance']['runtime_compatibility'], record)
            namespace['check_identity'](dict(saved, step=600000), 'a0c', {})
            with self.assertRaisesRegex(ValueError, 'lacks runtime compatibility'):
                namespace['check_identity'](dict(cp, step=600000), 'a0c', {})

    def test_real_adam_resume_across_seed_boundary(self):
        def update(model, opt, step):
            self.fixed(19991009 + step * 8192)
            opt.zero_grad()
            x = torch.randn(8, 3)
            loss = model(x).square().mean() + 0.0 * np.random.random()
            loss.backward(); opt.step()
            return loss.detach().clone()
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        update(model, opt, 521847)
        cp = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict()))
        expected = update(model, opt, 521848)
        resumed = torch.nn.Linear(3, 2)
        resumed.load_state_dict(cp['model'])
        opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
        opt2.load_state_dict(cp['optimizer'])
        self.assertTrue(torch.equal(expected, update(resumed, opt2, 521848)))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]))


if __name__ == '__main__':
    unittest.main()
