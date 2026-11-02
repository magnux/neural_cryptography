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

from contract import (ATOL, MAX_FRAMES, PARAMETER_COUNT, fixture_input, normalize_rgb,
                      parity_metrics, shapes, write_f32)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_EXPORT = HAS_TORCH and importlib.util.find_spec("onnx") is not None
if HAS_TORCH:
    import torch
    from model import (ChannelLayerNorm, MODEL_PROFILES, OpticalDecoder, parameter_count,
                       initialize_frames, frames_from_state, profile_from_state)
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

    def test_multiframe_fixture_has_distinct_rgb_slots_and_rejects_bad_counts(self):
        for size, frames in ((64, 4), (128, 4), (64, 5)):
            value = fixture_input(size, 7, frames=frames)
            self.assertEqual(value.shape, (1, 3*frames, 4*size, 4*size))
            self.assertTrue(value.flags.c_contiguous)
            np.testing.assert_array_equal(value, fixture_input(size, 7, frames=frames))
            for slot in range(1, frames):
                self.assertFalse(np.array_equal(value[:, :3], value[:, 3*slot:3*(slot+1)]))
        for frames in (2, 3, 5, 32, MAX_FRAMES):
            self.assertEqual(shapes(128, frames), ((1, 3*frames, 512, 512), (1, 16, 128, 128)))
        for frames in (0, MAX_FRAMES+1, True, False, 4.0):
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

    def test_five_frame_wide_stays_small_and_each_slot_affects_logits(self):
        torch.manual_seed(7)
        model = OpticalDecoder("wide", frames=5).eval()
        self.assertEqual(parameter_count(model), 489_024)
        self.assertLess(parameter_count(model), 500_000)
        self.assertEqual(frames_from_state(model.state_dict()), 5)
        for size in (64, 128):
            with torch.inference_mode():
                pixels = torch.from_numpy(fixture_input(size, 7, frames=5))
                expected = model(pixels)
                self.assertEqual(tuple(expected.shape), shapes(size, 5)[1])
                self.assertTrue(torch.isfinite(expected).all())
                for slot in range(5):
                    changed = pixels.clone()
                    changed[:, slot*3:(slot+1)*3] = 0
                    self.assertGreater(float((model(changed)-expected).abs().max()), 1e-6)

    def test_normalized_profiles_preserve_resolution_with_no_rgb_readout(self):
        for profile, count, width in (("normalized-compact", 125600, 16),
                                      ("normalized-tube", 144432, 32)):
            with self.subTest(profile=profile):
                torch.manual_seed(7)
                model = OpticalDecoder(profile, frames=5).eval()
                self.assertEqual(parameter_count(model), count)
                self.assertEqual(model.inlet[0][0].out_channels, width)
                self.assertEqual(model.inlet[1][0].out_channels, width)
                self.assertEqual(sum(isinstance(layer, ChannelLayerNorm) for layer in model.modules()),
                                 14 if profile == "normalized-tube" else 12)
                self.assertFalse(hasattr(model, "local_color"))
                self.assertEqual(profile_from_state(model.state_dict()), profile)
                if profile == "normalized-tube":
                    hidden = [layer for layer in model.modules()
                              if isinstance(layer, torch.nn.Conv2d) and layer is not model.head]
                    self.assertEqual(len(hidden), 14)
                    self.assertEqual({layer.out_channels for layer in hidden}, {32})
                    self.assertEqual((model.d1[0][0].in_channels, model.d0[0][0].in_channels), (64, 64))
                for size in (64, 128):
                    with self.subTest(size=size), torch.inference_mode():
                        pixels = torch.from_numpy(fixture_input(size, 7, frames=5))
                        spatial = {}
                        stages = ("inlet.0", "inlet.1", "e0", "e1", "bottleneck", "d1", "d0",
                                  "outlet.0", "outlet.1", "head") if profile == "normalized-tube" else ()
                        handles = [model.get_submodule(name).register_forward_hook(
                            lambda _module, _args, output, name=name: spatial.__setitem__(name, tuple(output.shape)))
                            for name in stages]
                        try:
                            expected = model(pixels)
                        finally:
                            for handle in handles:
                                handle.remove()
                        if profile == "normalized-tube":
                            for name in ("inlet.0", "inlet.1", "e0", "d0"):
                                self.assertEqual(spatial[name], (1, 32, 4*size, 4*size))
                            for name in ("e1", "d1", "outlet.0"):
                                self.assertEqual(spatial[name], (1, 32, 2*size, 2*size))
                            for name in ("bottleneck", "outlet.1"):
                                self.assertEqual(spatial[name], (1, 32, size, size))
                            self.assertEqual(spatial["head"], (1, 16, size, size))
                        self.assertEqual(tuple(expected.shape), shapes(size, 5)[1])
                        self.assertTrue(torch.isfinite(expected).all())
                        changed = pixels.clone()
                        changed[:, -3:] = 0
                        self.assertGreater(float((model(changed)-expected).abs().max()), 1e-6)

    def test_every_profile_clips_before_learned_layers_without_mutating_captures(self):
        torch.manual_seed(19)
        pixels = torch.linspace(-1.5, 1.5, 15*32*32).reshape(1, 15, 32, 32)
        original = pixels.clone()
        for profile in MODEL_PROFILES:
            with self.subTest(profile=profile), torch.inference_mode():
                model = OpticalDecoder(profile, frames=5).eval()
                received = []
                handle = model.inlet.register_forward_pre_hook(
                    lambda _module, args: received.append(args[0].clone()))
                actual = model(pixels)
                handle.remove()
                expected = model(pixels.clamp(-1, 1))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(received[0], pixels.clamp(-1, 1), rtol=0, atol=0)
                torch.testing.assert_close(pixels, original, rtol=0, atol=0)

    def test_explicit_five_frame_warm_start_preserves_repeated_input_function(self):
        for profile in MODEL_PROFILES:
            torch.manual_seed(31)
            original = OpticalDecoder(profile).eval()
            before = {name: value.clone() for name, value in original.state_dict().items()}
            expanded = initialize_frames(original, 5)
            self.assertEqual(expanded.frames, 5)
            self.assertFalse(expanded.training)
            with torch.inference_mode():
                pixels = torch.from_numpy(fixture_input(64, 9))
                torch.testing.assert_close(expanded(pixels.repeat(1, 5, 1, 1)), original(pixels),
                                           rtol=2e-5, atol=2e-6)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                if name not in ("inlet.0.0.weight", "local_color.0.weight"):
                    torch.testing.assert_close(expanded.state_dict()[name], before[name], rtol=0, atol=0)
            self.assertIs(initialize_frames(expanded, 5), expanded)
            for frames in (1, 4, 32):
                with self.assertRaisesRegex(ValueError, "single-frame source"):
                    initialize_frames(expanded, frames)
        for frames in (False, True, 0, MAX_FRAMES+1, 4.0):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                OpticalDecoder("wide", frames)
            with self.subTest(initialization=frames), self.assertRaises(ValueError):
                initialize_frames(original, frames)

    def test_larger_frame_counts_expand_only_input_weights(self):
        for profile, baseline, per_frame in (("compact", 121_792, 216), ("wide", 487_104, 480),
                                             ("normalized-compact", 123_872, 432), ("normalized-tube", 140_976, 864)):
            original = OpticalDecoder(profile).eval()
            for frames in (2, 3, 32, MAX_FRAMES):
                with self.subTest(profile=profile, frames=frames):
                    expanded = initialize_frames(original, frames)
                    self.assertEqual(frames_from_state(expanded.state_dict()), frames)
                    self.assertEqual(parameter_count(expanded), baseline+per_frame*(frames-1))
                    self.assertEqual(expanded.inlet[0][0].in_channels, 3*frames)
                    self.assertEqual(expanded.head.in_channels, original.head.in_channels)
                    torch.testing.assert_close(expanded.inlet[0][0].weight,
                        original.inlet[0][0].weight.repeat(1, frames, 1, 1)/frames, rtol=0, atol=0)

    def test_checkpoint_frame_inference_rejects_non_rgb_or_out_of_range_widths(self):
        for width in (0, 1, 4, 191, 193, 195):
            with self.subTest(width=width), self.assertRaises(ValueError):
                frames_from_state({"inlet.0.0.weight": torch.empty((8, width, 3, 3))})
        for value in ({}, {"inlet.0.0.weight": torch.empty((8, 3, 3))}, None):
            with self.assertRaises(ValueError):
                frames_from_state(value)



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

    def test_five_frame_export_has_fixed_fifteen_channel_graph_and_exact_fixture(self):
        import onnx
        from export import export_smoke
        for profile, count in (("normalized-compact", 125600), ("normalized-tube", 144432)):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)/"burst"
                manifest = export_smoke(directory, frames=5, sizes=(64,), check_ort=False, profile=profile)
                self.assertEqual(manifest["framesPerInput"], 5)
                self.assertEqual(manifest["channels"], 15)
                self.assertEqual(manifest["parameterCount"], count)
                self.assertEqual(manifest["modelProfile"], profile)
                case = manifest["cases"][0]
                self.assertEqual(case["inputShape"], [1, 15, 256, 256])
                graph = onnx.load(directory/case["model"]["file"])
                self.assertEqual([dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim],
                                 case["inputShape"])
                torch.manual_seed(7)
                model = OpticalDecoder(profile, frames=5).eval()
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
