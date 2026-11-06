import importlib.util
import random
import unittest

import numpy as np
import torch

from portfolio import COMMON


class SeedCompatibilityTests(unittest.TestCase):
    def setUp(self):
        path = COMMON.parents[2] / "archive/continuation/seed32_recovery.py"
        spec = importlib.util.spec_from_file_location("archived_seed32", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.source = (COMMON / "runtime.py").read_text(encoding="utf-8")
        self.fixed = module.seed_function(
            self.source, dict(random=random, np=np, torch=torch)
        )

    def test_seed_overflow_fix_preserves_python_and_torch_streams(self):
        from runtime import seed_all

        seed = 19991009 + 521848 * 8192
        with self.assertRaises(ValueError):
            seed_all(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        expected = random.random(), torch.rand(4).tolist()
        self.fixed(seed)
        self.assertEqual(expected, (random.random(), torch.rand(4).tolist()))
        self.assertEqual(
            np.random.random(4).tolist(),
            np.random.RandomState(seed % 2**32).random(4).tolist(),
        )

    def test_recovery_rejects_unexpected_seed_implementation(self):
        with self.assertRaises(RuntimeError):
            self.module.seed_function(
                self.source.replace("np.random.seed(seed)", "np.random.seed(seed + 1)"),
                {},
            )


if __name__ == "__main__":
    unittest.main()
