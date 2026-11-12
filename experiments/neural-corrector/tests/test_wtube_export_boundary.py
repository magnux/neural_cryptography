"""Four-pixel static exports preserve trained anchor and packed-input boundaries."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import fixture_input, shapes
HAS_EXPORT = all(importlib.util.find_spec(name) is not None for name in ("torch", "onnx"))
HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
if HAS_EXPORT:
    import torch
    import onnx
    from export import export_smoke, static_export_model
    from model import NeuralCorrector
    from test_wtube_model import checkpoint


class WTubeContractTests(unittest.TestCase):
    def test_four_pixel_phase_profiles_pack_every_supported_geometry(self):
        for size in (16, 32, 64, 128):
            self.assertEqual(shapes(size, 8, presence=True),
                             ((1, 32, 4*size, 4*size), (1, 16, size, size)))
            for frames in (4, 8):
                value = fixture_input(size, 7, 8, presence=True, available_frames=frames)
                flags = value[0, 24:, 0, 0]
                self.assertEqual(int(flags.sum()), frames)
                self.assertTrue(np.all(flags.reshape(4, 2).sum(1) >= 1))
                self.assertTrue(np.all(value[0, 24:] == flags[:, None, None]))
                for slot, present in enumerate(flags):
                    if not present:
                        self.assertTrue(np.all(value[0, slot*3:slot*3+3] == 0))
        for size in (8, 256):
            with self.assertRaises(ValueError):
                shapes(size, 8, presence=True)


@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX required")
class WTubeExportBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)

    def test_baked_profiles_preserve_student_weights_rng_and_logits(self):
        for profile in ("W-tube", "VW"):
            with self.subTest(profile=profile):
                torch.manual_seed(11)
                source = NeuralCorrector(profile).eval()
                before = {key: value.clone() for key, value in source.state_dict().items()}
                state = torch.get_rng_state().clone()
                baked = static_export_model(source, 16)
                torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
                self.assertEqual(baked.profile, profile)
                self.assertFalse(hasattr(baked, "anchor_coefficients"))
                torch.testing.assert_close(baked.baked_anchor, source.spatial_anchor(16), rtol=0, atol=0)
                for key, value in source.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                pixels = torch.from_numpy(fixture_input(16, 7, 8, presence=True))
                with torch.inference_mode():
                    torch.testing.assert_close(source(pixels), baked(pixels), rtol=0, atol=0)

    def test_current_neural_qr_manifest_graph_and_native_parity(self):
        for profile in ("W-tube", "VW"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                manifest = export_smoke(out, profile=profile, sizes=(16,), check_ort=HAS_ORT)
                self.assertEqual(manifest["modelProfile"], profile)
                self.assertEqual(manifest["datasetFormat"], "neural-qr-dataset")
                self.assertEqual(manifest["datasetVersion"], 2)
                self.assertEqual(manifest["pixelsPerCell"], 4)
                self.assertEqual(manifest["frameCounts"], [4, 5, 6, 7, 8])
                self.assertEqual(manifest["bottleneck"]["stages"], 2 if profile == "W-tube" else 3)
                self.assertEqual(manifest["anchorDeployment"], "baked-selected-spatial-grid")
                case = manifest["cases"][0]
                self.assertEqual(case["inputShape"], [1, 32, 64, 64])
                self.assertEqual(case["outputShape"], [1, 16, 16, 16])
                graph = onnx.load(out/case["model"]["file"])
                self.assertEqual({value.name: list(value.dims) for value in graph.graph.initializer
                                  if "anchor" in value.name}, {"baked_anchor": [1, 32, 16, 16]})
                self.assertFalse(any(any(word in value.name for word in ("teacher", "projector", "basis"))
                                     for value in graph.graph.initializer))
                if HAS_ORT:
                    self.assertEqual(len(case["pythonParity"]), 2)
                    self.assertTrue(all(row["passed"] for row in case["pythonParity"]))

    def test_trained_vw_exports_only_learned_anchors_without_replacing_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = NeuralCorrector("VW").eval()
            value = checkpoint(source)
            value["training"]["modelProfile"] = "VW"
            value["training"]["settings"] = {"selfDistillation": {"enabled": True}}
            path = root/"trained.pt"
            torch.save(value, path)
            out = root/"bundle"
            out.mkdir()
            (out/"keep.txt").write_text("unrelated")
            manifest = export_smoke(out, checkpoint=path, check_ort=False)
            self.assertEqual(manifest["status"], "trained")
            self.assertEqual([row["size"] for row in manifest["cases"]], [16])
            self.assertEqual(manifest["training"]["trainedSizes"], [16])
            self.assertNotIn("settings", manifest["training"])
            self.assertEqual((out/"keep.txt").read_text(), "unrelated")
            before = (out/"manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "no trained anchor"):
                export_smoke(out, checkpoint=path, sizes=(32,), check_ort=False)
            self.assertEqual((out/"manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
