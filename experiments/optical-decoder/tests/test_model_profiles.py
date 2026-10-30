"""Architecture ablations preserve color evidence, checkpoint identity, and ONNX I/O."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_EXPORT = HAS_TORCH and importlib.util.find_spec("onnx") is not None
if HAS_TORCH:
    import torch
    from model import MODEL_PROFILES, ChannelLayerNorm, OpticalDecoder, parameter_count
    from checkpoint import load_model_checkpoint, validate_split_provenance, training_provenance
    torch.set_num_threads(1)


def training_metadata():
    return {"steps": 1, "seed": 7, "trainSessionIds": ["training"], "trainPayloadIds": ["train-payload"],
            "validationSessionIds": ["validation"], "validationPayloadIds": ["validation-payload"]}


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for model profile tests")
class ProfileTests(unittest.TestCase):
    def test_profiles_output_contract_gradients_and_batch_independence(self):
        torch.manual_seed(7)
        for profile in MODEL_PROFILES:
            with self.subTest(profile=profile):
                model = OpticalDecoder(profile)
                for size in (64, 128):
                    pixels = torch.randn(1, 3, 4 * size, 4 * size).clamp(-1.5, 1.5)
                    output = model(pixels)
                    self.assertEqual(tuple(output.shape), (1, 16, size, size))
                    self.assertTrue(torch.isfinite(output).all())
                model.zero_grad()
                pixels = torch.randn(1, 3, 256, 256).clamp(-1.5, 1.5)
                logits = model(pixels)
                torch.nn.functional.cross_entropy(logits, torch.randint(16, (1, 64, 64))).backward()
                self.assertGreater(float(model.inlet[0][0].weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
                if profile == "wide":
                    self.assertTrue(any(isinstance(m, ChannelLayerNorm) for m in model.modules()))
                    self.assertGreater(float(model.local_color[0].weight.grad.abs().sum()), 0)
                model.eval()
                with torch.inference_mode():
                    single = model(pixels)
                    batch = model(torch.cat((pixels, torch.randn_like(pixels)), dim=0))[:1]
                torch.testing.assert_close(single, batch, rtol=1e-4, atol=1e-5)
        self.assertEqual(parameter_count(OpticalDecoder("compact")), 121792)
        self.assertLess(parameter_count(OpticalDecoder("wide")), 600000)

    def test_local_color_readout_preserves_absolute_brightness(self):
        # Isolate the unnormalized path: feature normalization must not erase
        # every way for the model to distinguish differently exposed colors.
        model = OpticalDecoder("wide").eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.local_color[0].weight[0, 0, 0, 0] = 1
            model.local_color[2].weight[0, 0, 0, 0] = 1
        with torch.inference_mode():
            dark = model(torch.full((1, 3, 256, 256), .1))
            light = model(torch.full((1, 3, 256, 256), .6))
        torch.testing.assert_close(light[:, 0] - dark[:, 0], torch.full((1, 64, 64), .5))

    def test_same_checkpoint_format_selects_exact_profile_without_parameter_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            for profile in MODEL_PROFILES:
                original = OpticalDecoder(profile)
                value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
                         "modelState": original.state_dict(), "training": training_metadata()}
                torch.save(value, path)
                loaded, checkpoint = load_model_checkpoint(path)
                self.assertEqual(loaded.profile, profile)
                self.assertEqual(checkpoint["version"], 1)
                for name, tensor in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
                value["modelState"]["head.weight"] = torch.zeros(16, 17, 1, 1)
                torch.save(value, path)
                with self.assertRaisesRegex(ValueError, "supported decoder profile"):
                    load_model_checkpoint(path)

    def test_large_dataset_provenance_is_not_capped_at_one_mib_or_100k_identities(self):
        metadata = training_metadata()
        metadata["trainPayloadIds"] = [f"payload-{i:064x}" for i in range(100001)]
        public = training_provenance(metadata)
        validate_split_provenance(metadata)
        self.assertNotIn("trainPayloadIds", public)
        self.assertEqual(public["steps"], 1)


@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX are required for wide export tests")
class WideExportTests(unittest.TestCase):
    def test_wide_export_static_graph_and_torch_fixtures(self):
        import onnx
        from export import export_smoke, ALLOWED_OPERATORS
        from contract import fixture_input
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = export_smoke(directory, sizes=(64, 128), check_ort=False, profile="wide")
            self.assertEqual(manifest["modelProfile"], "wide")
            torch.manual_seed(7)
            model = OpticalDecoder("wide").eval()
            self.assertEqual(manifest["parameterCount"], parameter_count(model))
            for case in manifest["cases"]:
                graph = onnx.load(directory / case["model"]["file"])
                onnx.checker.check_model(graph, full_check=True)
                operators = {node.op_type for node in graph.graph.node}
                self.assertTrue(operators <= ALLOWED_OPERATORS)
                self.assertIn("ReduceMean", operators)
                self.assertIn("AveragePool", operators)
                expected = np.fromfile(directory / case["expected"]["file"], dtype="<f4").reshape(case["outputShape"])
                with torch.inference_mode():
                    actual = model(torch.from_numpy(fixture_input(case["size"], 7))).numpy()
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
