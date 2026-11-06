import unittest

import torch

import portfolio
from method_registry import get_model_class
from property_common import PropertyBatch, PropertyNormalizer, audit_checkpoint_keys


class ConditioningTests(unittest.TestCase):
    def test_training_statistics_are_buffers_and_missing_values_are_neutral(self):
        normalizer = PropertyNormalizer([2, 4, 60], [1.5, 1, 30])
        self.assertEqual(len(list(normalizer.parameters())), 0)
        values = torch.tensor([[3.5, float("nan"), 30]])
        z, observed = normalizer(
            PropertyBatch(values, torch.tensor([[True, False, True]]))
        )
        torch.testing.assert_close(z, torch.tensor([[1.0, 0, -1]]))
        self.assertFalse(observed[0, 1])
        with self.assertRaises(ValueError):
            normalizer(PropertyBatch(values, torch.ones_like(values, dtype=torch.bool)))

    def test_a2_signed_digit_roundtrip_and_all_seventy_fixed_encodings(self):
        encoder = get_model_class("a2_rt_numeric").ENCODER(
            portfolio.NAMES, 32, portfolio.DEMO_MEANS, portfolio.DEMO_STDS
        )
        values = torch.tensor(
            [[-1.2345, 0, 12.3456], [999.9999, -999.9999, 0.00004]], dtype=torch.float64
        )
        tokens = encoder.tokenize_z(values)
        torch.testing.assert_close(
            encoder.decode_z(tokens),
            torch.round(values * 10000) / 10000,
            atol=1e-12,
            rtol=0,
        )
        for slot, order in enumerate(encoder.orders):
            for digit in range(10):
                reference = torch.tensor(
                    [
                        (-1 if i % 2 else 1) * digit * 10.0**order / (i - i % 2 + 1)
                        for i in range(32)
                    ]
                )
                torch.testing.assert_close(
                    encoder.fixed_numerical_encoding[2 + slot * 10 + digit],
                    reference,
                    atol=0,
                    rtol=0,
                )
        with self.assertRaises(ValueError):
            encoder.tokenize_z(torch.tensor([[1000.0]]))

    def test_missing_values_do_not_leak_through_all_property_encoders(self):
        for mode in portfolio.MODES[1:]:
            encoder = portfolio.build_tiny(mode).property_encoder.eval()
            observed = torch.tensor([[True, False, True]])
            a = PropertyBatch(torch.tensor([[2.0, float("nan"), 60.0]]), observed)
            b = PropertyBatch(torch.tensor([[2.0, 9000, 60.0]]), observed)
            torch.testing.assert_close(encoder(a)[0], encoder(b)[0], atol=0, rtol=0)

    def test_a3_missingness_embedding_remains_in_pooled_condition(self):
        encoder = portfolio.build_tiny("a3_adaln_zero").property_encoder
        batch = PropertyBatch(
            torch.tensor([[2.0, 4, 60]]), torch.zeros(1, 3, dtype=torch.bool)
        )
        encoder.pooled(batch).sum().backward()
        self.assertGreater(
            encoder.missing_embedding.weight.grad[1].abs().sum().item(), 0
        )

    def test_checkpoint_audit_rejects_missing_backbone_and_unexpected_keys(self):
        self.assertTrue(
            audit_checkpoint_keys(["property_encoder.new.weight"], [])["pass"]
        )
        with self.assertRaises(RuntimeError):
            audit_checkpoint_keys(["input_up_proj.0.weight"], [])
        with self.assertRaises(RuntimeError):
            audit_checkpoint_keys([], ["unknown.weight"])


if __name__ == "__main__":
    unittest.main()
