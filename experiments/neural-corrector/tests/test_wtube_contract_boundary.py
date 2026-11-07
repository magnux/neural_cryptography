"""Dataset raster and target semantics are checked before model execution."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_model, require_implemented_model_contract


class WTubeContractBoundaryTests(unittest.TestCase):
    def test_matching_profiles_and_rasters(self):
        for format_, version, ppc, profile in (("neural-qr-dataset", 2, 4, "W-tube"),
                ("neural-qr-dataset", 1, 2, "multires"),
                ("optical-decoder-dataset", 1, 4, "U-tube-2x-attn")):
            data = SimpleNamespace(dataset_format=format_, dataset_version=version, pixels_per_cell=ppc)
            require_implemented_model_contract(data)
            require_implemented_model_contract(data, profile)

    def test_wrong_profile_or_raster_rejected_before_sampling(self):
        for version, ppc, wrong_profiles in ((2, 4, ("multires", "U-tube-2x-attn")),
                                             (1, 2, ("W-tube", "U-tube"))):
            data = SimpleNamespace(dataset_format="neural-qr-dataset", dataset_version=version, pixels_per_cell=ppc)
            for profile in wrong_profiles:
                with self.assertRaisesRegex(ValueError, "W-tube requires"):
                    evaluate_model(SimpleNamespace(profile=profile), data)
        data = SimpleNamespace(dataset_format="neural-qr-dataset", dataset_version=2, pixels_per_cell=2)
        with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
            require_implemented_model_contract(data, "W-tube")
        obm = SimpleNamespace(dataset_format="optical-decoder-dataset", dataset_version=1, pixels_per_cell=4)
        with self.assertRaisesRegex(ValueError, "W-tube requires"):
            require_implemented_model_contract(obm, "W-tube")


if __name__ == "__main__":
    unittest.main()
