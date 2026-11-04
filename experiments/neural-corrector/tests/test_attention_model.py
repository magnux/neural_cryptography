"""Combined learned-anchor, masked-burst and shifted-window model behavior."""

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
    from model import NeuralCorrector, WindowAttentionBlock, parameter_count, initialize_frames
    from contract import fixture_input
    from checkpoint import load_model_checkpoint
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for attention model checks")
class AttentionModelTests(unittest.TestCase):
    def test_fixed_capacity_global_anchor_and_supported_grid(self):
        model = NeuralCorrector("U-tube-2x-attn")
        self.assertEqual(parameter_count(model), 432848)
        self.assertEqual(model.frames, 8)
        self.assertEqual(model.min_frames, 4)
        self.assertEqual(model.input_channels, 32)
        self.assertEqual(model.supported_sizes, (64,))
        self.assertEqual(model.input_layout, "phase-paired-rgb-presence-planes")
        self.assertEqual(model.output_label_space, "logical")
        self.assertEqual(model.phase_masks, (0, 15, 7, 8))
        self.assertEqual(tuple(model.position_anchor.shape), (1, 32, 64, 64))
        self.assertEqual([block.shift for block in model.attention], [0, 4])
        self.assertAlmostEqual(float(model.position_anchor.detach().std()), .02, delta=.001)
        self.assertIs(initialize_frames(model, 8), model)
        for count in (1, 5, 7, 9, 64):
            with self.assertRaises(ValueError):
                NeuralCorrector("U-tube-2x-attn", frames=count)
        with self.assertRaisesRegex(ValueError, "G64"):
            model(torch.zeros(1, 32, 512, 512))

    def test_missing_slots_are_ignored_and_real_slots_remain_sensitive(self):
        torch.manual_seed(13)
        model = NeuralCorrector("U-tube-2x-attn").eval()
        pixels = torch.from_numpy(fixture_input(64, 13, 8, presence=True, available_frames=4))
        presence = pixels[0, 24:, 0, 0]
        self.assertEqual(int(presence.sum()), 4)
        self.assertTrue(bool((presence.reshape(4, 2).sum(1) == 1).all()))
        missing = torch.nonzero(presence == 0).flatten().tolist()
        available = torch.nonzero(presence == 1).flatten().tolist()
        changed = pixels.clone()
        for slot in missing:
            changed[:, 3*slot:3*slot+3] = 1000*torch.randn_like(changed[:, 3*slot:3*slot+3])
        with torch.inference_mode():
            expected = model(pixels)
            torch.testing.assert_close(model(changed), expected, rtol=0, atol=0)
            # The whole packed input clips identically: binary flags are fixed,
            # and RGB calibration overshoot is clipped before learned layers.
            torch.testing.assert_close(model(pixels.clamp(-1, 1)), expected, rtol=0, atol=0)
            changed = pixels.clone()
            slot = available[0]
            changed[:, 3*slot:3*slot+3] *= -1
            self.assertGreater(float((model(changed)-expected).abs().max()), 1e-5)

    def test_anchor_is_added_after_full_resolution_inlet_and_receives_gradients(self):
        torch.manual_seed(21)
        model = NeuralCorrector("U-tube-2x-attn")
        pixels = torch.from_numpy(fixture_input(64, 21, 8, presence=True, available_frames=8)).clamp(-1, 1)
        observed = {}
        hook = model.e0.register_forward_pre_hook(lambda _module, args: observed.update(e0=args[0]))
        output = model(pixels)
        hook.remove()
        expected = model.inlet(pixels) + F.interpolate(model.position_anchor, size=(256, 256),
                                                       mode="bilinear", align_corners=False)
        torch.testing.assert_close(observed["e0"], expected, rtol=0, atol=0)
        F.cross_entropy(output, torch.randint(16, (1, 64, 64))).backward()
        self.assertGreater(float(model.position_anchor.grad.abs().sum()), 0)
        for block in model.attention:
            self.assertGreater(float(block.qkv.weight.grad.abs().sum()), 0)
            self.assertGreater(float(block.projection.weight.grad.abs().sum()), 0)
            self.assertGreater(float(block.ffn[0].weight.grad.abs().sum()), 0)

    def test_shifted_windows_never_wrap_evidence_across_opposite_edges(self):
        block = WindowAttentionBlock(shift=4).eval()
        with torch.no_grad():
            block.qkv.weight.zero_()
            block.qkv.bias.zero_()
            block.qkv.weight[64:] = torch.eye(32)
            block.projection.weight.copy_(torch.eye(32))
            block.projection.bias.zero_()
            for parameter in block.ffn.parameters():
                parameter.zero_()
        baseline = torch.zeros(1, 32, 64, 64)
        impulse = baseline.clone()
        impulse[0, 0, 0, 0] = 1
        with torch.inference_mode():
            difference = block(impulse)-block(baseline)
        self.assertGreater(float(difference[:, :, 0, 1].abs().sum()), 0)
        self.assertEqual(float(difference[:, :, 63, 0].abs().sum()), 0)
        self.assertEqual(float(difference[:, :, 0, 63].abs().sum()), 0)
        self.assertEqual(float(difference[:, :, 63, 63].abs().sum()), 0)

    def test_checkpoint_requires_exact_masked_frame_contract(self):
        model = NeuralCorrector("U-tube-2x-attn")
        metadata = {"steps": 1, "seed": 7, "framesPerInput": 8,
                    "modelProfile": "U-tube-2x-attn", "inputLayout": "phase-paired-rgb-presence-planes",
                    "minFramesPerInput": 4, "frameCounts": [4, 5, 6, 7, 8],
                    "frameCountPolicy": "uniform-4-8", "outputLabelSpace": "logical",
                    "phaseMasks": [0, 15, 7, 8],
                    "selectionAggregation": "frame-count-group-balanced-real",
                    "trainSessionIds": ["train"], "trainPayloadIds": ["train-payload"],
                    "validationSessionIds": ["val"], "validationPayloadIds": ["val-payload"]}
        value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
                 "modelState": model.state_dict(), "training": metadata}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"attention.pt"
            torch.save(value, path)
            restored, _ = load_model_checkpoint(path)
            self.assertEqual(restored.profile, "U-tube-2x-attn")
            for key, wrong in (("minFramesPerInput", 5), ("frameCounts", [5, 8]),
                               ("frameCountPolicy", "fixed"), ("inputLayout", "rgb-presence-planes"),
                               ("outputLabelSpace", "physical"), ("phaseMasks", [])):
                broken = {**value, "training": {**metadata, key: wrong}}
                torch.save(broken, path)
                with self.subTest(field=key), self.assertRaises(ValueError):
                    load_model_checkpoint(path)
            broken = {**value, "training": {key: val for key, val in metadata.items() if key != "frameCounts"}}
            torch.save(broken, path)
            with self.assertRaisesRegex(ValueError, "requires frameCounts"):
                load_model_checkpoint(path)

    def test_fixture_has_every_phase_at_all_capture_counts(self):
        for count in range(4, 9):
            pixels = fixture_input(64, 5, 8, presence=True, available_frames=count)
            present = pixels[0, 24:, 0, 0]
            self.assertEqual(int(present.sum()), count)
            self.assertTrue(bool((present.reshape(4, 2).sum(1) >= 1).all()))


if __name__ == "__main__":
    unittest.main()
