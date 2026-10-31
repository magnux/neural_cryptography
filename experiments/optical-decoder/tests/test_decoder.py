"""M1 tests: tensor contract, architecture, and actual ONNX execution parity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contract import (ATOL, PARAMETER_COUNT, fixture_input, normalize_rgb,
                      parity_metrics, shapes, write_f32)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_EXPORT = HAS_TORCH and importlib.util.find_spec("onnx") is not None
if HAS_TORCH:
    import torch
    from model import OpticalDecoder, parameter_count, initialize_frames, frames_from_state
    torch.set_num_threads(1)


class ContractTests(unittest.TestCase):
    def test_normalization_channel_order_clipping_and_precision(self):
        rgb = np.zeros((256, 256, 3), dtype=np.float64)
        rgb[0, 0] = [0, 127.5, 255]
        rgb[0, 1] = [-127.5, 382.5, 254.25]
        pixels = normalize_rgb(rgb)
        self.assertEqual(pixels.shape, (1, 3, 256, 256))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertTrue(pixels.flags.c_contiguous)
        np.testing.assert_array_equal(pixels[0, :, 0, 0], [-1, 0, 1])
        np.testing.assert_array_equal(pixels[0, :, 0, 1], [-1.5, 1.5, np.float32(254.25 / 127.5 - 1)])

    def test_bad_inputs_are_rejected(self):
        for rgb in (np.zeros((32, 32, 3)), np.zeros((256, 256, 4)),
                    np.zeros((256, 512, 3)), np.full((256, 256, 3), np.nan),
                    np.zeros((256, 256, 3), dtype=np.complex64)):
            with self.subTest(shape=rgb.shape), self.assertRaises(ValueError):
                normalize_rgb(rgb)
        for size in (32, 65, 256, True, 64.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                shapes(size)

    def test_diagnostic_inputs_are_reproducible_and_nonconstant(self):
        for size in (64, 128):
            value = fixture_input(size, 7)
            self.assertEqual(value.shape, shapes(size)[0])
            np.testing.assert_array_equal(value, fixture_input(size, 7))
            self.assertFalse(np.array_equal(value, fixture_input(size, 8)))
            self.assertGreater(float(value.std()), .3)
            self.assertGreaterEqual(float(value.min()), -1.5)
            self.assertLessEqual(float(value.max()), 1.5)

    def test_four_frame_fixture_has_distinct_rgb_slots_and_rejects_bad_counts(self):
        for size in (64, 128):
            value = fixture_input(size, 7, frames=4)
            self.assertEqual(value.shape, (1, 12, 4*size, 4*size))
            self.assertTrue(value.flags.c_contiguous)
            np.testing.assert_array_equal(value, fixture_input(size, 7, frames=4))
            for slot in range(1, 4):
                self.assertFalse(np.array_equal(value[:, :3], value[:, 3*slot:3*(slot+1)]))
        for frames in (0, 2, 3, 5, True, 4.0):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                shapes(64, frames)

    def test_binary_contract_is_little_endian_float32(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.f32"
            details = write_f32(path, np.array([1, -2.5, .125], dtype=np.float64))
            expected = struct.pack("<fff", 1, -2.5, .125)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(details, {"file": "fixture.f32", "sha256": hashlib.sha256(expected).hexdigest(), "bytes": 12})

    def test_tolerance_uses_absolute_plus_relative_and_rejects_nan(self):
        expected = np.array([0, 10], dtype=np.float32)
        self.assertTrue(parity_metrics(expected, expected + [ATOL / 2, .001])["passed"])
        self.assertFalse(parity_metrics(expected, expected + [ATOL * 2, 0])["passed"])
        with self.assertRaises(ValueError):
            parity_metrics(expected, np.array([0, np.nan]))


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for model execution")
class ModelTests(unittest.TestCase):
    def test_architecture_shape_and_input_sensitivity(self):
        torch.manual_seed(7)
        model = OpticalDecoder().eval()
        self.assertEqual(parameter_count(model), PARAMETER_COUNT)
        self.assertFalse(any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)) for m in model.modules()))
        for size in (64, 128):
            with self.subTest(size=size), torch.inference_mode():
                a = model(torch.from_numpy(fixture_input(size, 7)))
                b = model(torch.from_numpy(fixture_input(size, 8)))
                self.assertEqual(tuple(a.shape), shapes(size)[1])
                self.assertTrue(torch.isfinite(a).all())
                self.assertGreater(float(torch.max(torch.abs(a - b))), 1e-7)

    def test_four_frame_wide_stays_small_and_each_slot_affects_logits(self):
        torch.manual_seed(7)
        model = OpticalDecoder("wide", frames=4).eval()
        self.assertEqual(parameter_count(model), 488_544)
        self.assertLess(parameter_count(model), 500_000)
        self.assertEqual(frames_from_state(model.state_dict()), 4)
        for size in (64, 128):
            with torch.inference_mode():
                pixels = torch.from_numpy(fixture_input(size, 7, frames=4))
                expected = model(pixels)
                self.assertEqual(tuple(expected.shape), shapes(size, 4)[1])
                self.assertTrue(torch.isfinite(expected).all())
                for slot in range(4):
                    changed = pixels.clone()
                    changed[:, slot*3:(slot+1)*3] = 0
                    self.assertGreater(float((model(changed)-expected).abs().max()), 1e-6)

    def test_explicit_four_frame_warm_start_preserves_repeated_input_function(self):
        for profile in ("compact", "wide"):
            torch.manual_seed(31)
            original = OpticalDecoder(profile).eval()
            before = {name: value.clone() for name, value in original.state_dict().items()}
            expanded = initialize_frames(original, 4)
            self.assertEqual(expanded.frames, 4)
            self.assertFalse(expanded.training)
            with torch.inference_mode():
                pixels = torch.from_numpy(fixture_input(64, 9))
                torch.testing.assert_close(expanded(pixels.repeat(1, 4, 1, 1)), original(pixels),
                                           rtol=2e-5, atol=2e-6)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                if name not in ("inlet.0.0.weight", "local_color.0.weight"):
                    torch.testing.assert_close(expanded.state_dict()[name], before[name], rtol=0, atol=0)
            self.assertIs(initialize_frames(expanded, 4), expanded)
            with self.assertRaisesRegex(ValueError, "one-to-four"):
                initialize_frames(expanded, 1)
        for frames in (False, 0, 2, 4.0):
            with self.assertRaises(ValueError):
                OpticalDecoder("wide", frames)



@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX are required for graph export")
class ExportTests(unittest.TestCase):
    def test_fixed_graphs_use_same_weights_and_fixtures_match_torch(self):
        import onnx
        from export import export_smoke

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            manifest = export_smoke(directory, check_ort=False, profile="compact")
            self.assertEqual(manifest["status"], "untrained")
            self.assertEqual(manifest["parameterCount"], PARAMETER_COUNT)
            self.assertEqual(manifest["framesPerInput"], 1)
            self.assertEqual(manifest["pythonParityStatus"]["status"], "not_run")
            self.assertEqual(json.loads((directory / "manifest.json").read_text()), manifest)
            self.assertEqual([item["size"] for item in manifest["cases"]], [64, 128])
            weights = []
            torch.manual_seed(7)
            model = OpticalDecoder().eval()
            for case in manifest["cases"]:
                for kind in ("model", "input", "expected"):
                    info = case[kind]
                    data = (directory / info["file"]).read_bytes()
                    self.assertEqual(len(data), info["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), info["sha256"])
                graph = onnx.load(directory / case["model"]["file"])
                onnx.checker.check_model(graph, full_check=True)
                actual_input_shape = [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim]
                actual_output_shape = [dim.dim_value for dim in graph.graph.output[0].type.tensor_type.shape.dim]
                self.assertEqual(actual_input_shape, case["inputShape"])
                self.assertEqual(actual_output_shape, case["outputShape"])
                weights.append({v.name: onnx.numpy_helper.to_array(v) for v in graph.graph.initializer})
                self.assertEqual(case["pythonParity"], [])
                pixels = np.fromfile(directory / case["input"]["file"], dtype="<f4").reshape(case["inputShape"])
                expected = np.fromfile(directory / case["expected"]["file"], dtype="<f4").reshape(case["outputShape"])
                with torch.inference_mode():
                    actual = model(torch.from_numpy(pixels)).numpy()
                np.testing.assert_array_equal(expected, actual)
            self.assertEqual(set(weights[0]), set(weights[1]))
            for name in weights[0]:
                np.testing.assert_array_equal(weights[0][name], weights[1][name])

    def test_four_frame_export_has_fixed_twelve_channel_graph_and_exact_fixture(self):
        import onnx
        from export import export_smoke
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"burst"
            manifest = export_smoke(directory, frames=4, sizes=(64,), check_ort=False, profile="wide")
            self.assertEqual(manifest["framesPerInput"], 4)
            self.assertEqual(manifest["channels"], 12)
            self.assertEqual(manifest["parameterCount"], 488_544)
            case = manifest["cases"][0]
            self.assertEqual(case["inputShape"], [1, 12, 256, 256])
            graph = onnx.load(directory/case["model"]["file"])
            self.assertEqual([dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim],
                             case["inputShape"])
            torch.manual_seed(7)
            model = OpticalDecoder("wide", frames=4).eval()
            pixels = np.fromfile(directory/case["input"]["file"], dtype="<f4").reshape(case["inputShape"])
            expected = np.fromfile(directory/case["expected"]["file"], dtype="<f4").reshape(case["outputShape"])
            with torch.inference_mode():
                np.testing.assert_array_equal(model(torch.from_numpy(pixels)).numpy(), expected)

    def test_native_runtime_parity_when_available(self):
        from export import export_smoke
        # A restricted machine can expose an importable package whose native
        # CPU initialization aborts. Probe separately so the result is a visible
        # skip, never an interpreter crash or a claimed successful check.
        probe = subprocess.run([sys.executable, "-c", "import onnxruntime; print(onnxruntime.__version__)"],
                               capture_output=True, text=True, timeout=30)
        if probe.returncode:
            self.skipTest(f"Native ONNX Runtime unavailable (exit {probe.returncode}); browser parity still required")
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_smoke(Path(temporary) / "native", sizes=(64,), check_ort=True)
            self.assertEqual(manifest["pythonParityStatus"]["status"], "passed")
            self.assertTrue(all(check["passed"] for check in manifest["cases"][0]["pythonParity"]))


if __name__ == "__main__":
    unittest.main()
