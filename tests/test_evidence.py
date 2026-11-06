import unittest
import hashlib

try:
    from tools.evaluate_predictions import recompute
except ImportError:
    recompute = None


class EvidenceTests(unittest.TestCase):
    def test_metric_denominators_and_training_membership(self):
        self.assertIsNotNone(recompute, "Portable prediction verifier is missing")
        predictions = [
            {
                "index": "0",
                "id": "a",
                "generated_selfies": "[C][C][O]",
                "reference_smiles": "OCC",
            },
            {
                "index": "1",
                "id": "b",
                "generated_selfies": "",
                "reference_smiles": "CCO",
            },
            {
                "index": "2",
                "id": "c",
                "generated_selfies": "[C][C][C]",
                "reference_smiles": "CCO",
            },
        ]
        training = {hashlib.sha256(b"CCO").hexdigest()}
        metrics = recompute(predictions, training)
        self.assertEqual(metrics["n_generated"], 3)
        self.assertEqual(metrics["n_valid"], 2)
        self.assertEqual(metrics["n_exact"], 1)
        self.assertEqual(metrics["validity"], 2 / 3)
        self.assertEqual(metrics["exact_canonical_isomeric_smiles"], 1 / 3)
        self.assertEqual(metrics["uniqueness_valid_canonical"], 1)
        self.assertEqual(metrics["novelty_unique_valid_canonical"], 0.5)

    def test_no_valid_outputs_have_explicit_undefined_similarity(self):
        self.assertIsNotNone(recompute, "Portable prediction verifier is missing")
        row = {
            "index": "0",
            "id": "a",
            "generated_selfies": "",
            "reference_smiles": "CO",
        }
        metrics = recompute([row], set())
        self.assertIsNone(metrics["maccs_paired_valid_mean"])
        self.assertIsNone(metrics["novelty_unique_valid_canonical"])


if __name__ == "__main__":
    unittest.main()
