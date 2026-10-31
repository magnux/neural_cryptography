"""Checks for adaptation gradients, calibrated augmentation, and patience rules."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from adaptation import adaptation_loss, OnlineRealAugment, PlateauController
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for adaptation")
class AdaptationTests(unittest.TestCase):
    def test_per_frame_blend_and_masked_gradients(self):
        # First frame has four valid cells, second just one: their ordinary CE
        # must be averaged equally rather than weighted by data-cell counts.
        correct = torch.tensor([[.8, .6, .4, .2], [.5, .9, .9, .9]])
        probabilities = torch.stack([correct]+[(1-correct)/3]*3, dim=1).unsqueeze(2)
        inactive = torch.full((2, 12, 1, 4), 100.)
        logits = torch.cat([probabilities.log(), inactive], dim=1).requires_grad_()
        labels = torch.tensor([[[0, 0, 0, 0]], [[0, 255, 255, 255]]])
        mask = torch.tensor([[[True, True, True, True]], [[True, False, False, False]]])
        loss, ordinary = adaptation_loss(logits, labels, mask, 4)
        first_mean = -correct[0].log().mean()
        second_mean = -correct[1, 0].log()
        expected = (.9*first_mean+.1*(-correct[0, -1].log())+second_mean)/2
        torch.testing.assert_close(ordinary, torch.stack([first_mean, second_mean]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:, 4:] == 0))
        self.assertTrue(torch.all(logits.grad[1, :, :, 1:] == 0))
        self.assertLess(float(logits.grad[0, 0, 0, -1]), 0)
        mean_only, _ = adaptation_loss(logits, labels, mask, 4, hard_weight=0)
        torch.testing.assert_close(mean_only, ordinary.mean())
        all_hard, _ = adaptation_loss(logits, labels, mask, 4, hard_fraction=1)
        torch.testing.assert_close(all_hard, ordinary.mean())

    def test_loss_uses_ceil_hard_fraction_and_rejects_empty_frames(self):
        probabilities = torch.tensor([.8, .4, .1])
        logits = torch.stack([probabilities]+[(1-probabilities)/3]*3)[None, :, None].log()
        labels = torch.zeros((1, 1, 3), dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        loss, ordinary = adaptation_loss(logits, labels, mask, 4, hard_weight=.2, hard_fraction=.5)
        torch.testing.assert_close(loss, .8*ordinary[0]+.2*(-probabilities[1:].log().mean()))
        with self.assertRaisesRegex(ValueError, "Every frame"):
            adaptation_loss(logits, labels, torch.zeros_like(mask), 4)
        for kwargs in ({"hard_weight": -1}, {"hard_weight": math.nan},
                       {"hard_fraction": 0}, {"hard_fraction": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adaptation_loss(logits, labels, mask, 4, **kwargs)

    def test_augmentation_is_reproducible_real_only_and_does_not_mutate(self):
        pixels = torch.linspace(-1.5, 1.5, 4*3*16*16).reshape(4, 3, 16, 16)
        original = pixels.clone()
        first, repeat = OnlineRealAugment(seed=9), OnlineRealAugment(seed=9)
        rng_before = torch.random.get_rng_state().clone()
        output = first(pixels, [True, False, True, False])
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        torch.testing.assert_close(output, repeat(pixels, [True, False, True, False]), rtol=0, atol=0)
        torch.testing.assert_close(pixels, original, rtol=0, atol=0)
        torch.testing.assert_close(output[[1, 3]], pixels[[1, 3]], rtol=0, atol=0)
        self.assertFalse(torch.equal(output[[0, 2]], pixels[[0, 2]]))
        self.assertFalse(torch.equal(output, first(pixels, [True, False, True, False])))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(float(output.abs().max()), 1.5)
        self.assertEqual(output.dtype, torch.float32)

    def test_augmentation_retains_calibrated_overshoot_and_zero_identity(self):
        pixels = torch.tensor([1.3, -1.3, .2]).reshape(1, 3, 1, 1).expand(2, 3, 4, 4).clone()
        disabled = OnlineRealAugment(exposure_stops=0, white_balance_stops=0, noise_std=0)
        identity = disabled(pixels, [True, True])
        torch.testing.assert_close(identity, pixels, rtol=0, atol=0)
        self.assertNotEqual(identity.data_ptr(), pixels.data_ptr())
        mild = OnlineRealAugment(exposure_stops=.001, white_balance_stops=0, noise_std=0)
        output = mild(pixels, [True, False])
        self.assertGreater(float(output[0, 0].min()), 1)
        self.assertLess(float(output[0, 1].max()), -1)
        torch.testing.assert_close(output[1], pixels[1], rtol=0, atol=0)

    def test_white_balance_and_exposure_gain_bounds_without_spatial_changes(self):
        # A zero normalized input is mid-gray. With no noise, each frame's
        # channel gains must be constant over space and geometrically balanced.
        pixels = torch.zeros((24, 3, 8, 8))
        wb = OnlineRealAugment(seed=31, exposure_stops=0, noise_std=0)
        gains = wb(pixels, [True]*24)+1
        torch.testing.assert_close(gains, gains[:, :, :1, :1].expand_as(gains))
        torch.testing.assert_close(gains.prod(1), torch.ones((24, 8, 8)), rtol=1e-6, atol=1e-6)
        self.assertLessEqual(float(gains.log2().abs().max()), .100001)
        exposure = OnlineRealAugment(seed=31, white_balance_stops=0, noise_std=0)
        gains = exposure(pixels, [True]*24)+1
        torch.testing.assert_close(gains, gains[:, :1, :1, :1].expand_as(gains))
        stops = gains[:, 0, 0, 0].log2()
        self.assertGreater(float(stops.max()), 0)
        self.assertLess(float(stops.min()), 0)
        self.assertLessEqual(float(stops.abs().max()), .150001)

    def test_augmentation_rejects_bad_inputs_and_options(self):
        for kwargs in ({"seed": True}, {"seed": -1}, {"seed": 1.5},
                       {"noise_std": -1}, {"noise_std": math.inf},
                       {"white_balance_stops": math.nan}, {"exposure_stops": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OnlineRealAugment(**kwargs)
        augment = OnlineRealAugment()
        pixels = torch.zeros((2, 3, 8, 8))
        for mask in ([1, 0], [True], torch.ones(2), [[True], [False]]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                augment(pixels, mask)
        with self.assertRaises(ValueError):
            augment(pixels.double(), [True, True])
        with self.assertRaises(ValueError):
            augment(pixels+2, [True, True])

    def test_plateau_baseline_reductions_and_stop_counter(self):
        controller = PlateauController(1e-5)
        initial = controller.update(.1, 1.)
        self.assertTrue(initial["improved"])
        self.assertEqual(initial["badChecks"], 0)
        for check in range(1, 13):
            result = controller.update(.1, 1.01)
            self.assertEqual(result["badChecks"], check)
            self.assertEqual(result["shouldStop"], check == 12)
            self.assertEqual(result["reduced"], check in (4, 8, 12))
        self.assertEqual(controller.lr, 2e-6)
        self.assertEqual(controller.reductions, 3)
        self.assertEqual(controller.bad_checks, 12)

    def test_plateau_progress_uses_global_meaningful_anchor(self):
        controller = PlateauController(1e-5)
        controller.update(.1, 1.)
        # Small improvements accumulate relative to the meaningful best.
        self.assertFalse(controller.update(.1, .9996)["improved"])
        self.assertFalse(controller.update(.1, .9992)["improved"])
        improved = controller.update(.1, .9988)
        self.assertTrue(improved["improved"])
        self.assertEqual(improved["badChecks"], 0)
        # Lower CE cannot compensate for worse exact recovery.
        self.assertFalse(controller.update(.09, .5)["improved"])
        self.assertTrue(controller.update(.11, 2.)["improved"])
        self.assertEqual(controller.bad_checks, 0)
        low = PlateauController(1e-7, min_lr=1e-7, lr_patience=1, stop_patience=2)
        low.update(0, 1)
        result = low.update(0, 1)
        self.assertEqual(result["lr"], 1e-7)
        self.assertFalse(result["reduced"])
        self.assertTrue(low.update(0, 1)["shouldStop"])

    def test_plateau_controls_can_be_disabled_independently(self):
        no_decay = PlateauController(1e-5, lr_patience=0, stop_patience=2)
        no_decay.update(0, 1)
        no_decay.update(0, 1)
        result = no_decay.update(0, 1)
        self.assertEqual(result["lr"], 1e-5)
        self.assertTrue(result["shouldStop"])
        no_stop = PlateauController(1e-5, lr_patience=1, stop_patience=0)
        no_stop.update(0, 1)
        result = no_stop.update(0, 1)
        self.assertEqual(result["lr"], 5e-6)
        self.assertFalse(result["shouldStop"])

    def test_plateau_rejects_nonfinite_or_invalid_settings(self):
        for kwargs in ({"lr": 0}, {"lr": math.nan}, {"factor": 1}, {"factor": 0},
                       {"min_lr": -1}, {"min_lr": .1}, {"min_delta": math.inf},
                       {"lr_patience": True}, {"lr_patience": -1}, {"stop_patience": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PlateauController(**{"lr": 1e-5, **kwargs})
        controller = PlateauController(1e-5)
        for exact, ce in ((1.1, 1), (0, -1), (math.nan, 1), (0, math.inf)):
            with self.subTest(exact=exact, ce=ce), self.assertRaises(ValueError):
                controller.update(exact, ce)


if __name__ == "__main__":
    unittest.main()
