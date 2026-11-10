"""VW adds a complete independent third U pass to the four-pixel contract."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import load_checkpoint, load_model_checkpoint
from model import (NeuralCorrector, W_TUBE_PROFILES, W_TUBE_SIZES, frames_from_state,
                   parameter_count, processing_parameter_count, profile_from_state,
                   wtube_bottleneck_metadata)
from test_wtube_model import checkpoint, packed


def vw_checkpoint(model):
    value = checkpoint(model)
    value["training"]["modelProfile"] = model.profile
    return value


class VWModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)

    def test_three_independent_full_u_paths_cover_all_supported_geometries(self):
        model = NeuralCorrector("VW").eval()
        self.assertEqual(W_TUBE_PROFILES, ("W-tube", "VW"))
        self.assertEqual(parameter_count(model), 696_400)
        self.assertEqual(processing_parameter_count(model), 663_632)
        self.assertEqual(len(model.stages), 3)
        self.assertEqual(model.frames, 8)
        self.assertEqual(model.supported_sizes, W_TUBE_SIZES)
        self.assertEqual(model.bottleneck_metadata["stages"], 3)
        self.assertEqual(wtube_bottleneck_metadata("W-tube")["stages"], 2)
        self.assertEqual(len({id(stage.e0.pairs[0].conv1.weight) for stage in model.stages}), 3)
        self.assertEqual(len({stage.attention[0].qkv.weight.data_ptr() for stage in model.stages}), 3)
        shapes, hooks = {}, []
        for index, stage in enumerate(model.stages):
            self.assertEqual(len(stage.bottleneck.pairs), 1)
            self.assertEqual([block.shift for block in stage.attention], [0, 4])
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
                    for index in range(3):
                        for name, scale in (("e0", 4), ("e1", 2), ("bottleneck", 1),
                                            ("attention", 1), ("d1", 2), ("d0", 4), ("fusion", 4)):
                            self.assertEqual(shapes[index, name], (1, 32, scale*size, scale*size))
        finally:
            for hook in hooks:
                hook.remove()
        for size in (8, 256):
            with self.assertRaisesRegex(ValueError, "requires packed eight-slot"):
                model(torch.empty(1, 32, 4*size, 4*size, device="meta"))
        with self.assertRaisesRegex(ValueError, "eight phase-paired slots"):
            NeuralCorrector("VW", frames=5)

    def test_all_three_paths_and_selected_anchor_receive_gradients(self):
        torch.manual_seed(31)
        model = NeuralCorrector("VW")
        output = model(packed(16, counts=5))
        torch.nn.functional.cross_entropy(output, torch.randint(0, 16, (1, 16, 16))).backward()
        for index, stage in enumerate(model.stages):
            for parameter in (stage.e0.pairs[0].conv1.weight, stage.e1.pairs[0].conv1.weight,
                              stage.bottleneck.pairs[0].conv2.weight, stage.attention[1].qkv.weight,
                              stage.d1.pairs[0].conv1.weight, stage.d0.pairs[0].conv2.weight):
                with self.subTest(stage=index):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(model.anchor_coefficients["16"].grad.abs().sum().item(), 0)
        for size in (32, 64, 128):
            self.assertIsNone(model.anchor_coefficients[str(size)].grad)

    def test_masking_and_anchor_injection_stay_at_shared_inlet(self):
        model = NeuralCorrector("VW").eval()
        pixels = packed(16, counts=4)
        altered = pixels.clone()
        for slot in (1, 3, 5, 7):
            altered[:, slot*3:slot*3+3] = 999
        anchor_calls = []
        original = model.spatial_anchor
        def observed_anchor(size):
            anchor_calls.append(size)
            return original(size)
        model.spatial_anchor = observed_anchor
        with torch.no_grad():
            reference = model(pixels)
            self.assertEqual(anchor_calls, [16])
            torch.testing.assert_close(reference, model(altered), rtol=0, atol=0)

    def test_strict_checkpoint_identity_and_stage_count(self):
        torch.manual_seed(44)
        model = NeuralCorrector("VW").eval()
        self.assertEqual(profile_from_state(model.state_dict()), "VW")
        self.assertEqual(frames_from_state(model.state_dict()), 8)
        original = vw_checkpoint(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"vw.pt"
            torch.save(original, path)
            loaded, value = load_model_checkpoint(path)
            self.assertEqual(loaded.profile, "VW")
            self.assertEqual(value["training"]["bottleneck"]["stages"], 3)
            with torch.no_grad():
                torch.testing.assert_close(model(packed(16)), loaded.eval()(packed(16)), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "parameters do not match"):
                load_checkpoint(path, NeuralCorrector("W-tube"))
            for damaged_state in (
                    {key: val for key, val in original["modelState"].items() if not key.startswith("stages.1.")},
                    {key: val for key, val in original["modelState"].items() if not key.startswith("stages.2.")},
                    {**original["modelState"], "stages.3.extra": torch.zeros(1)},
                    {**original["modelState"], "stages.bad.extra": torch.zeros(1)}):
                torch.save({**original, "modelState": damaged_state}, path)
                with self.assertRaises(ValueError):
                    load_model_checkpoint(path)
            for key, content in (("modelProfile", "W-tube"),
                                 ("bottleneck", wtube_bottleneck_metadata("W-tube")),
                                 ("datasetVersion", 1), ("pixelsPerCell", 2)):
                damaged = {**original, "training": {**original["training"], key: content}}
                torch.save(damaged, path)
                with self.subTest(metadata=key), self.assertRaises(ValueError):
                    load_model_checkpoint(path)
            metadata = copy.deepcopy(original["training"])
            metadata["bottleneck"]["stages"] = 3.0
            torch.save({**original, "training": metadata}, path)
            with self.assertRaises(ValueError):
                load_model_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
