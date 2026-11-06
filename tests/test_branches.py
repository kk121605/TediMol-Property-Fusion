import copy
import unittest

import torch

try:
    import portfolio
except ImportError:
    portfolio = None


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(portfolio, "The publication factory is missing")
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.base = portfolio.build_tiny("a0c")
        self.x = torch.randn(2, 12, 8)
        self.text = torch.randn(2, 10, 768)
        self.mask = torch.ones(2, 10)
        self.times = torch.tensor([100.0, 900.0])
        self.values = torch.tensor([[1.2, 3.0, 40.0], [2.4, 5.0, 80.0]])

    def test_all_four_real_models_run_and_a3_starts_as_a0(self):
        expected = self.base.eval()(self.x, self.times, self.text, self.mask)
        for mode in portfolio.MODES:
            with self.subTest(mode=mode):
                model = portfolio.build_tiny(mode, parent=self.base).eval()
                result = portfolio.forward(
                    model, mode, self.x, self.times, self.text, self.mask, self.values
                )
                self.assertEqual(result.shape, (2, 12, 8))
                self.assertTrue(torch.isfinite(result).all())
                if mode in ("a0c", "a3_adaln_zero"):
                    torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_prefix_lengths_are_three_and_twenty_four(self):
        for mode, prefix in [
            ("a1_prefix", 3),
            ("a2_rt_numeric", 24),
            ("a3_adaln_zero", 0),
        ]:
            model = portfolio.build_tiny(mode)
            lengths = []
            hook = model.input_transformers.layer[0].register_forward_pre_hook(
                lambda module, args: lengths.append(args[0].shape[1])
            )
            portfolio.forward(
                model, mode, self.x, self.times, self.text, self.mask, self.values
            )
            hook.remove()
            self.assertEqual(lengths, [12 + prefix])

    def test_adamw_ema_resume_matches_uninterrupted_real_diffusion(self):
        from comparison_runtime import Bridge, update
        from improved_diffusion.resample import UniformSampler
        from runtime import make_diffusion

        config = {
            "lr": 1e-5,
            "lr_horizon_steps": 300000,
            "microbatch": 1,
            "seed": 19991009,
            "grad_clip": 2.4,
            "ema_rate": 0.9999,
        }
        batch = (
            torch.randint(0, 323, (2, 12)),
            self.text,
            self.mask,
            self.values,
            torch.ones_like(self.values, dtype=torch.bool),
        )
        for mode in portfolio.MODES:
            with self.subTest(mode=mode):
                model = portfolio.build_tiny(mode, parent=self.base).train()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=1e-5, weight_decay=0
                )
                ema = {name: p.detach().clone() for name, p in model.named_parameters()}
                diffusion = make_diffusion()
                sampler = UniformSampler(diffusion)
                first = update(
                    model,
                    Bridge(model, mode),
                    optimizer,
                    ema,
                    diffusion,
                    sampler,
                    batch,
                    0,
                    config,
                    "cpu",
                )
                self.assertTrue(torch.isfinite(torch.tensor(first["loss"])))
                if mode != "a0c":
                    self.assertTrue(
                        any(v > 0 for v in first["property_group_gradient_l2"].values())
                    )
                saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), ema))
                update(
                    model,
                    Bridge(model, mode),
                    optimizer,
                    ema,
                    diffusion,
                    sampler,
                    batch,
                    1,
                    config,
                    "cpu",
                )
                resumed = portfolio.build_tiny(mode).train()
                resumed.load_state_dict(saved[0], strict=True)
                opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-5, weight_decay=0)
                opt2.load_state_dict(saved[1])
                update(
                    resumed,
                    Bridge(resumed, mode),
                    opt2,
                    saved[2],
                    diffusion,
                    sampler,
                    batch,
                    1,
                    config,
                    "cpu",
                )
                for name, value in model.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, resumed.state_dict()[name]), name
                    )
                for name, value in ema.items():
                    self.assertTrue(torch.equal(value, saved[2][name]), name)

    def test_real_twenty_step_sampler_is_finite_for_every_branch(self):
        from runtime import make_diffusion, seed_all

        for mode in portfolio.MODES:
            with self.subTest(mode=mode):
                seed_all(121)
                model = portfolio.build_tiny(mode).eval()
                kwargs = {} if mode == "a0c" else {"chemical_metric": self.values}
                with torch.inference_mode():
                    sample = make_diffusion(20).p_sample_loop(
                        model,
                        self.x.shape,
                        device="cpu",
                        clip_denoised=False,
                        top_p=1.0,
                        progress=False,
                        desc=(self.text, self.mask),
                        model_kwargs=kwargs,
                    )
                self.assertEqual(sample.shape, self.x.shape)
                self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__":
    unittest.main()
