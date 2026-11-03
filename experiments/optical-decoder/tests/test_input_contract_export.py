"""Stored calibration evidence is preserved; the exported network clips inputs."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contract import (CAPTURE_INPUT_MIN, CAPTURE_INPUT_MAX, NETWORK_INPUT_MIN,
                      NETWORK_INPUT_MAX, normalize_rgb)

HAS_EXPORT = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("onnx") is not None


class CaptureNormalizationTests(unittest.TestCase):
    def test_rgb_formula_preserves_calibration_overshoot_until_model_boundary(self):
        rgb = np.broadcast_to(np.array([-20., 127.5, 280.]), (256, 256, 3)).copy()
        original = rgb.copy()
        captured = normalize_rgb(rgb)
        expected = np.array([-20./127.5-1, 0., 280./127.5-1], dtype=np.float32)
        np.testing.assert_array_equal(captured[0, :, 0, 0], expected)
        np.testing.assert_array_equal(rgb, original)
        self.assertLess(float(captured.min()), NETWORK_INPUT_MIN)
        self.assertGreater(float(captured.max()), NETWORK_INPUT_MAX)
        self.assertGreaterEqual(float(captured.min()), CAPTURE_INPUT_MIN)
        self.assertLessEqual(float(captured.max()), CAPTURE_INPUT_MAX)
        # The stored evidence keeps the existing bound even for extreme fitting
        # overshoot. This is separate from the stricter in-graph network clamp.
        rgb[0, 0] = [-500., 127.5, 500.]
        np.testing.assert_array_equal(normalize_rgb(rgb)[0, :, 0, 0], [-1.5, 0., 1.5])


@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX are required for export checks")
class NetworkInputExportTests(unittest.TestCase):
    def test_default_five_frame_export_clips_before_any_network_consumer(self):
        import onnx
        from onnx import numpy_helper
        from export import export_smoke

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = export_smoke(directory, sizes=(64,), check_ort=False, frames=5)
            self.assertEqual(manifest["modelProfile"], "normalized-tube")
            self.assertEqual(manifest["parameterCount"], 144432)
            self.assertEqual(manifest["normalization"],
                             {"divisor": 127.5, "offset": -1., "clip": [-1.5, 1.5]})
            self.assertEqual(manifest["networkInputClip"], [-1., 1.])
            case = manifest["cases"][0]
            pixels = np.fromfile(directory/case["input"]["file"], dtype="<f4")
            self.assertTrue(np.any(pixels < -1.) and np.any(pixels > 1.),
                            "parity fixtures must exercise real calibration overshoot")
            graph = onnx.load(directory/case["model"]["file"]).graph
            consumers = [node for node in graph.node if "pixels" in node.input]
            self.assertEqual(len(consumers), 1)
            clip = consumers[0]
            self.assertEqual(clip.op_type, "Clip")
            constants = {initializer.name: numpy_helper.to_array(initializer)
                         for initializer in graph.initializer}
            for node in graph.node:
                if node.op_type == "Constant":
                    value = next((attribute.t for attribute in node.attribute
                                  if attribute.name == "value"), None)
                    if value is not None:
                        constants[node.output[0]] = numpy_helper.to_array(value)
            self.assertEqual(float(constants[clip.input[1]]), NETWORK_INPUT_MIN)
            self.assertEqual(float(constants[clip.input[2]]), NETWORK_INPUT_MAX)


if __name__ == "__main__":
    unittest.main()
