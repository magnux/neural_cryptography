"""W-tube geometry, evidence masking, trainability and strict checkpoints."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import load_model_checkpoint
from model import (NeuralCorrector, W_TUBE_SIZES, frames_from_state,
                   parameter_count, processing_parameter_count, profile_from_state)


def packed(size, counts=8):
    generator = torch.Generator().manual_seed(size+counts)
    values = torch.rand((1, 32, 4*size, 4*size), generator=generator)*2-1
    values[:, 24:] = 0
    slots = [0, 2, 4, 6, 1, 3, 5, 7][:counts]
    values[:, [24+i for i in slots]] = 1
    return values


def checkpoint(model):
    return {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
        "modelState": model.state_dict(), "training": {
            "steps": 1, "seed": 7, "modelProfile": "W-tube", "datasetFormat": "neural-qr-dataset",
            "datasetVersion": 2, "pixelsPerCell": 4, "supportedSizes": list(W_TUBE_SIZES),
            "trainedSizes": [16], "anchor": copy.deepcopy(model.anchor_metadata),
            "bottleneck": copy.deepcopy(model.bottleneck_metadata), "framesPerInput": 8,
            "inputLayout": model.input_layout, "outputLabelSpace": "logical",
            "phaseMasks": [0, 15, 7, 8], "minFramesPerInput": 4,
            "frameCounts": [4, 5, 6, 7, 8], "frameCountPolicy": "uniform-4-8",
            "trainSessionIds": ["train-session"], "trainPayloadIds": ["a"*64],
            "validationSessionIds": ["validation-session"], "validationPayloadIds": ["b"*64]}}


class WTubeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)

    def test_four_grids_run_two_full_u_stages_without_subcell_bottlenecks(self):
        model = NeuralCorrector("W-tube").eval()
        self.assertEqual(parameter_count(model), 487_760)
        self.assertEqual(processing_parameter_count(model), 454_992)
        self.assertEqual(set(model.anchor_coefficients), {"16", "32", "64", "128"})
        self.assertEqual(len(list(model.stages[0].bottleneck.pairs)), 1)
        self.assertEqual(len(model.stages[0].attention), 2)
        self.assertIsNot(model.stages[0].e0.pairs[0].conv1.weight,
                         model.stages[1].e0.pairs[0].conv1.weight)
        shapes, hooks = {}, []
        for index, stage in enumerate(model.stages):
            for name in ("e0", "e1", "bottleneck", "attention", "d1", "d0", "fusion"):
                key = (index, name)
                hooks.append(getattr(stage, name).register_forward_hook(
                    lambda module, args, output, key=key: shapes.__setitem__(key, tuple(output.shape))))
        try:
            for size in W_TUBE_SIZES:
                with self.subTest(grid=size), torch.no_grad():
                    output = model(packed(size))
                    self.assertEqual(tuple(output.shape), (1, 16, size, size))
                    self.assertTrue(torch.isfinite(output).all())
                    for index in range(2):
                        for name, scale in (("e0", 4), ("e1", 2), ("bottleneck", 1),
                                            ("attention", 1), ("d1", 2), ("d0", 4), ("fusion", 4)):
                            self.assertEqual(shapes[index, name], (1, 32, scale*size, scale*size))
        finally:
            for hook in hooks:
                hook.remove()
        for size in (8, 256):
            with self.assertRaisesRegex(ValueError, "requires packed eight-slot"):
                model(torch.empty((1, 32, 4*size, 4*size), device="meta"))
        with self.assertRaisesRegex(ValueError, "eight phase-paired slots"):
            NeuralCorrector("W-tube", frames=5)

    def test_absent_rgb_slots_are_ignored_and_live_evidence_uses_clipped_range(self):
        model = NeuralCorrector("W-tube").eval()
        pixels = packed(16, counts=4)
        altered = pixels.clone()
        for slot in (1, 3, 5, 7):
            altered[:, slot*3:slot*3+3] = 999
        with torch.no_grad():
            torch.testing.assert_close(model(pixels), model(altered), rtol=0, atol=0)
            overshoot = packed(16, counts=8)
            overshoot[:, :24] *= 1.5
            clipped = overshoot.clone()
            clipped[:, :24].clamp_(-1, 1)
            torch.testing.assert_close(model(overshoot), model(clipped), rtol=0, atol=0)

    def test_selected_anchor_expands_four_times_once_before_first_u(self):
        model = NeuralCorrector("W-tube").eval()
        pixels = packed(32)
        seen = []
        hook = model.stages[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            with torch.no_grad():
                model(pixels)
                expected = model.inlet(pixels) + torch.nn.functional.interpolate(
                    model.spatial_anchor(32), scale_factor=4, mode="bilinear", align_corners=False)
                torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        finally:
            hook.remove()
        self.assertEqual(tuple(model.spatial_anchor(32).shape), (1, 32, 32, 32))

    def test_both_u_stages_and_only_selected_anchor_receive_gradients(self):
        torch.manual_seed(7)
        model = NeuralCorrector("W-tube")
        loss = model(packed(16, counts=5)).square().mean()
        loss.backward()
        for index in range(2):
            for parameter in (model.stages[index].e0.pairs[0].conv1.weight,
                              model.stages[index].bottleneck.pairs[0].conv2.weight,
                              model.stages[index].attention[1].qkv.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(model.anchor_coefficients["16"].grad.abs().sum().item(), 0)
        for size in (32, 64, 128):
            self.assertIsNone(model.anchor_coefficients[str(size)].grad)

    def test_checkpoint_roundtrip_requires_exact_geometry_and_metadata(self):
        torch.manual_seed(8)
        model = NeuralCorrector("W-tube").eval()
        self.assertEqual(profile_from_state(model.state_dict()), "W-tube")
        self.assertEqual(frames_from_state(model.state_dict()), 8)
        original = checkpoint(model)
        original["training"]["nativeRealFraction"] = .75
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.pt"
            torch.save(original, path)
            loaded, record = load_model_checkpoint(path)
            self.assertEqual(record["training"]["datasetVersion"], 2)
            self.assertEqual(record["training"]["nativeRealFraction"], .75)
            with torch.no_grad():
                torch.testing.assert_close(model(packed(16)), loaded.eval()(packed(16)), rtol=0, atol=0)
            for key, value in (("nativeRealFraction", True), ("nativeRealFraction", 1.1),
                               ("datasetVersion", 1), ("pixelsPerCell", 2),
                               ("supportedSizes", [16, 32, 64, 128, 256]),
                               ("trainedSizes", [8]), ("modelProfile", "multires"),
                               ("bottleneck", {"blocks": ["conv", "attention"]*3, "layersPerBlock": 2})):
                with self.subTest(metadata=key):
                    damaged = {**original, "training": {**original["training"], key: value}}
                    torch.save(damaged, path)
                    with self.assertRaises(ValueError):
                        load_model_checkpoint(path)
            missing = {**original, "training": dict(original["training"])}
            del missing["training"]["datasetVersion"]
            torch.save(missing, path)
            with self.assertRaisesRegex(ValueError, "datasetVersion"):
                load_model_checkpoint(path)
            damaged = {**original, "modelState": dict(original["modelState"])}
            damaged["modelState"]["anchor_coefficients.16"] = torch.zeros(1, 32, 8, 8)
            torch.save(damaged, path)
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint tensor"):
                load_model_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
