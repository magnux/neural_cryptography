"""Static deployment bakes one low-frequency size anchor and exact input packing."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contract import MULTIRES_SIZES, fixture_input, normalize_rgb, shapes

HAS_EXPORT = all(importlib.util.find_spec(name) is not None for name in ("torch", "onnx"))


class MultiresContractTests(unittest.TestCase):
    def test_two_pixel_geometry_and_four_phase_presence_at_every_size(self):
        for size in MULTIRES_SIZES:
            with self.subTest(size=size):
                self.assertEqual(shapes(size, 8, presence=True, pixels_per_cell=2),
                                 ((1, 32, 2*size, 2*size), (1, 16, size, size)))
                value = fixture_input(size, 23, 8, presence=True,
                                      available_frames=5, pixels_per_cell=2)
                flags = value[0, 24:, 0, 0]
                self.assertEqual(int(flags.sum()), 5)
                self.assertTrue(np.all(flags.reshape(4, 2).sum(1) >= 1))
                self.assertTrue(np.all(value[0, 24:] == flags[:, None, None]))
                for index, present in enumerate(flags):
                    if not present:
                        self.assertTrue(np.all(value[0, 3*index:3*index+3] == 0))
                self.assertEqual(value.dtype, np.float32)
                rgb = np.full((2*size, 2*size, 3), [0., 127.5, 255.])
                np.testing.assert_array_equal(normalize_rgb(rgb, pixels_per_cell=2)[0, :, 0, 0],
                                              [-1., 0., 1.])
        for kwargs in ({"size": 4}, {"size": 64, "frames": 5},
                       {"size": 64, "presence": False}, {"size": 64, "pixels_per_cell": 3}):
            arguments = {"size": 64, "frames": 8, "presence": True, "pixels_per_cell": 2, **kwargs}
            with self.assertRaises(ValueError):
                shapes(**arguments)


@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX are required")
class MultiresExportTests(unittest.TestCase):
    def test_static_profiles_export_only_selected_anchor_and_expected_logits(self):
        import onnx
        import torch
        from export import export_smoke
        from model import NeuralCorrector

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = export_smoke(output, seed=19, sizes=(8, 16), profile="multires", check_ort=False)
            self.assertEqual(manifest["pixelsPerCell"], 2)
            self.assertEqual(manifest["datasetFormat"], "neural-qr-dataset")
            self.assertEqual(manifest["parameterCount"], 274_128)
            self.assertEqual(manifest["anchorDeployment"], "baked-selected-spatial-grid")
            self.assertEqual(manifest["anchor"]["representation"], "dct-low-frequency")
            self.assertEqual(manifest["anchor"]["maxModes"], 16)
            self.assertEqual(manifest["bottleneck"]["blocks"], ["conv", "attention"]*3)
            self.assertEqual(manifest["frameCounts"], [4, 5, 6, 7, 8])
            torch.manual_seed(19)
            model = NeuralCorrector("multires").eval()
            for case in manifest["cases"]:
                size = case["size"]
                graph = onnx.load(output/case["model"]["file"])
                self.assertEqual({item.name: list(item.dims) for item in graph.graph.initializer
                                  if "anchor" in item.name},
                                 {"baked_anchor": [1, 32, size, size]})
                self.assertFalse(any("basis" in item.name or "cosine" in item.name
                                     for item in graph.graph.initializer))
                self.assertNotIn("DFT", case["operators"])
                self.assertEqual(case["inputShape"], [1, 32, 2*size, 2*size])
                pixels = np.fromfile(output/case["input"]["file"], "<f4").reshape(case["inputShape"])
                expected = np.fromfile(output/case["expected"]["file"], "<f4").reshape(case["outputShape"])
                with torch.inference_mode():
                    np.testing.assert_array_equal(expected, model(torch.from_numpy(pixels)).numpy())

    def test_baking_is_export_only_and_preserves_coefficients_weights_and_rng(self):
        import torch
        from export import static_export_model
        from model import NeuralCorrector

        torch.set_num_threads(1)
        torch.manual_seed(117)
        source = NeuralCorrector("multires").eval()
        before = {name: tensor.clone() for name, tensor in source.state_dict().items()}
        random_state = torch.get_rng_state().clone()
        baked = static_export_model(source, 32)
        torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)
        self.assertIsNot(baked, source)
        self.assertTrue(hasattr(source, "anchor_coefficients"))
        self.assertFalse(hasattr(baked, "anchor_coefficients"))
        torch.testing.assert_close(baked.baked_anchor, source.spatial_anchor(32), rtol=0, atol=0)
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        pixels = torch.from_numpy(fixture_input(32, 23, 8, presence=True, pixels_per_cell=2))
        with torch.inference_mode():
            torch.testing.assert_close(baked(pixels), source(pixels), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "selected grid"):
            baked.spatial_anchor(16)

    def test_trained_checkpoint_defaults_to_trained_sizes_and_rejects_other_anchors(self):
        import torch
        from export import export_smoke
        from model import NeuralCorrector

        model = NeuralCorrector("multires").eval()
        training = {"steps": 2, "seed": 7, "modelProfile": "multires",
                    "datasetFormat": "neural-qr-dataset", "pixelsPerCell": 2,
                    "supportedSizes": list(MULTIRES_SIZES), "trainedSizes": [8],
                    "framesPerInput": 8, "inputLayout": "phase-paired-rgb-presence-planes",
                    "minFramesPerInput": 4, "frameCounts": [4, 5, 6, 7, 8],
                    "frameCountPolicy": "uniform-4-8", "outputLabelSpace": "logical",
                    "phaseMasks": [0, 15, 7, 8], "trainSessionIds": ["train-session"],
                    "trainPayloadIds": ["train-target"], "validationSessionIds": ["validation-session"],
                    "validationPayloadIds": ["validation-target"]}
        training.update(anchor=dict(model.anchor_metadata), bottleneck=dict(model.bottleneck_metadata))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/"model.pt"
            torch.save({"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
                        "modelState": model.state_dict(), "training": training}, checkpoint)
            manifest = export_smoke(root/"bundle", checkpoint=checkpoint, check_ort=False)
            self.assertEqual([row["size"] for row in manifest["cases"]], [8])
            self.assertEqual(manifest["training"]["trainedSizes"], [8])
            with self.assertRaisesRegex(ValueError, "no trained anchor"):
                export_smoke(root/"invalid", checkpoint=checkpoint, sizes=(16,), check_ort=False)
            self.assertFalse((root/"invalid").exists())


if __name__ == "__main__":
    unittest.main()
