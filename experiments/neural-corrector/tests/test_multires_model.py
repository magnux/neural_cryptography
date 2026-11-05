"""Multires geometry, anchor isolation, phase packing and explicit warm starts."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from model import (NeuralCorrector, MULTIRES_SIZES, WindowAttentionBlock,
                       parameter_count, processing_parameter_count, ANCHOR_METADATA, BOTTLENECK_METADATA, project_anchor)
    from checkpoint import initialize_multires, load_model_checkpoint, load_checkpoint
    torch.set_num_threads(1)


def pixels(size, count=4):
    result = torch.randn(1, 32, 2*size, 2*size).clamp(-1.5, 1.5)
    result[:, 24:] = 0
    slots = [0, 2, 4, 6] + [1, 3, 5, 7][:count-4]
    result[:, [24+slot for slot in slots]] = 1
    return result


def metadata():
    return {"steps": 1, "seed": 7, "modelProfile": "multires", "framesPerInput": 8,
            "minFramesPerInput": 4, "frameCounts": [4, 5, 6, 7, 8],
            "frameCountPolicy": "uniform-4-8", "inputLayout": "phase-paired-rgb-presence-planes",
            "outputLabelSpace": "logical", "phaseMasks": [0, 15, 7, 8],
            "pixelsPerCell": 2, "supportedSizes": list(MULTIRES_SIZES),
            "trainedSizes": [8, 16, 64], "datasetFormat": "neural-qr-dataset",
            "anchor": ANCHOR_METADATA, "bottleneck": BOTTLENECK_METADATA,
            "trainSessionIds": ["train"], "trainPayloadIds": ["train-payload"],
            "validationSessionIds": ["val"], "validationPayloadIds": ["val-payload"]}


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for multires checks")
class MultiresModelTests(unittest.TestCase):
    def test_geometry_interleaving_and_parameter_storage(self):
        model = NeuralCorrector("multires")
        self.assertEqual(model.supported_sizes, MULTIRES_SIZES)
        self.assertEqual((model.frames, model.min_frames, model.pixels_per_cell), (8, 4, 2))
        self.assertEqual(parameter_count(model), 274128)
        self.assertEqual(processing_parameter_count(model), 231120)
        self.assertFalse(hasattr(model, "e1"))
        self.assertFalse(hasattr(model, "d1"))
        self.assertEqual(len(model.bottleneck.pairs), 1)
        self.assertEqual(len(model.bottleneck_second.pairs), 1)
        self.assertEqual(len(model.bottleneck_third.pairs), 1)
        self.assertEqual([block.shift for block in (*model.attention, *model.attention_second, *model.attention_third)], [0, 4, 0, 4, 0, 4])
        self.assertEqual({name: tuple(anchor.shape) for name, anchor in model.anchor_coefficients.items()},
                         {str(size): (1, 32, min(size, 16), min(size, 16)) for size in MULTIRES_SIZES})
        stages = []
        hooks = [getattr(model, name).register_forward_hook(
            lambda _module, _args, output, name=name: stages.append((name, output.shape[-1])))
            for name in ("e0", "bottleneck", "attention", "bottleneck_second", "attention_second", "bottleneck_third", "attention_third", "d0", "outlet")]
        with torch.inference_mode():
            self.assertEqual(tuple(model(pixels(16)).shape), (1, 16, 16, 16))
        for hook in hooks:
            hook.remove()
        self.assertEqual(stages, [("e0", 32), ("bottleneck", 16), ("attention", 16),
                                  ("bottleneck_second", 16), ("attention_second", 16),
                                  ("bottleneck_third", 16), ("attention_third", 16), ("d0", 32), ("outlet", 16)])

    def test_presence_gating_and_anchor_gradient_isolation(self):
        torch.manual_seed(71)
        model = NeuralCorrector("multires")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for size in (8, 16):
            optimizer.zero_grad(set_to_none=True)
            unchanged = {key: anchor.detach().clone() for key, anchor in model.anchor_coefficients.items() if key != str(size)}
            sample = pixels(size)
            corrupted = sample.clone()
            for slot in (1, 3, 5, 7):
                corrupted[:, slot*3:slot*3+3] = 1000
            expected = model(sample)
            torch.testing.assert_close(model(corrupted), expected, rtol=0, atol=0)
            F.cross_entropy(expected, torch.randint(16, (1, size, size))).backward()
            self.assertGreater(float(model.anchor_coefficients[str(size)].grad.abs().sum()), 0)
            for key in unchanged:
                self.assertIsNone(model.anchor_coefficients[key].grad)
            for block in (*model.attention, *model.attention_second, *model.attention_third):
                self.assertGreater(float(block.qkv.weight.grad.abs().sum()), 0)
            optimizer.step()
            for key, before in unchanged.items():
                torch.testing.assert_close(model.anchor_coefficients[key], before, rtol=0, atol=0)

    def test_anchor_enlarges_exactly_two_times_and_grid_validation(self):
        model = NeuralCorrector("multires")
        sample = pixels(8, 8).clamp(-1, 1)
        seen = []
        hook = model.e0.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
        model(sample, grid_size=8)
        hook.remove()
        expected = model.inlet(sample) + F.interpolate(model.spatial_anchor(8), scale_factor=2,
                                                       mode="bilinear", align_corners=False)
        torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        for bad in (torch.zeros(1, 32, 24, 24), torch.zeros(1, 24, 16, 16), torch.zeros(1, 32, 16, 32)):
            with self.assertRaises(ValueError):
                model(bad)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            model(sample, grid_size=16)
        for frames in (4, 5, 7, 9):
            with self.assertRaises(ValueError):
                NeuralCorrector("multires", frames=frames)

    def test_g8_disables_shift_and_other_sizes_do_not_wrap(self):
        unshifted, shifted = WindowAttentionBlock(), WindowAttentionBlock(4)
        shifted.load_state_dict(unshifted.state_dict(), strict=True)
        sample = torch.randn(2, 32, 8, 8)
        torch.testing.assert_close(unshifted(sample), shifted(sample), rtol=0, atol=0)
        for size in (16, 32, 128, 256):
            mask = WindowAttentionBlock.boundary_mask(size)
            self.assertEqual(tuple(mask.shape), ((size//8)**2, 1, 64, 64))
            self.assertTrue(bool((mask.diagonal(dim1=-2, dim2=-1) == 0).all()))
            self.assertEqual(float(mask.min()), -10000.)

    def test_exact_checkpoint_and_explicit_partial_warm_start(self):
        source = NeuralCorrector("U-tube-2x-attn")
        with torch.no_grad():
            source.position_anchor.fill_(.4)
        model, report = initialize_multires(source)
        self.assertEqual(report["transferredAnchors"], [64])
        self.assertEqual(report["projected"], ["anchor_coefficients.64"])
        torch.testing.assert_close(model.anchor_coefficients["64"], project_anchor(source.position_anchor), rtol=0, atol=0)
        torch.testing.assert_close(model.spatial_anchor(64), source.position_anchor, rtol=1e-5, atol=1e-6)
        self.assertIn("anchor_coefficients.128", report["fresh"])
        self.assertIn("attention_second.0.qkv.weight", report["fresh"])
        self.assertIn("e1.pairs.0.conv1.weight", report["discarded"])
        for name in report["copied"]:
            source_name = name
            torch.testing.assert_close(model.state_dict()[name], source.state_dict()[source_name], rtol=0, atol=0)
        self.assertFalse(bool((model.anchor_coefficients["128"] == .4).all()))
        with self.assertRaisesRegex(ValueError, "source"):
            initialize_multires(model)
        value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
                 "modelState": model.state_dict(), "training": metadata()}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"model.pt"
            torch.save(value, path)
            loaded, _ = load_model_checkpoint(path)
            self.assertEqual(loaded.profile, "multires")
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_checkpoint(path, source)
            for field, wrong in (("pixelsPerCell", 4), ("supportedSizes", [64]),
                                 ("trainedSizes", []), ("datasetFormat", "optical-decoder-dataset"),
                                 ("anchor", {**ANCHOR_METADATA, "maxModes": 32}),
                                 ("bottleneck", {**BOTTLENECK_METADATA, "layersPerBlock": 4})):
                torch.save({**value, "training": {**metadata(), field: wrong}}, path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_model_checkpoint(path)
            for field in ("anchor", "bottleneck"):
                incomplete = metadata()
                del incomplete[field]
                torch.save({**value, "training": incomplete}, path)
                with self.subTest(missing=field), self.assertRaises(ValueError):
                    load_model_checkpoint(path)
            malformed = dict(value["modelState"])
            del malformed["anchor_coefficients.256"]
            torch.save({**value, "modelState": malformed}, path)
            with self.assertRaisesRegex(ValueError, "parameters"):
                load_model_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
