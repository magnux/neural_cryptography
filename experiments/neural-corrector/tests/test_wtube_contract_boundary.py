"""New corpus preparation must not silently train the older 2ppc network."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_model, require_implemented_model_contract
from train import run_training


class WTubeContractBoundaryTests(unittest.TestCase):
    def test_training_and_evaluation_reject_before_io_or_model_use(self):
        data = SimpleNamespace(dataset_format="neural-qr-dataset", pixels_per_cell=4)
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)/"must-not-exist"
            with self.assertRaisesRegex(ValueError, "requires W-tube"):
                run_training(data, data, out, model_profile="multires")
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, "requires W-tube"):
                evaluate_model(None, data)

    def test_prepared_experiment_contracts_remain_supported(self):
        for format_, ppc in (("neural-qr-dataset", 2), ("optical-decoder-dataset", 4)):
            require_implemented_model_contract(SimpleNamespace(dataset_format=format_, pixels_per_cell=ppc))


if __name__ == "__main__":
    unittest.main()
