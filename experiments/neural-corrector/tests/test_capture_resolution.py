"""Finite camera sampling preserves target geometry but loses image detail."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from synthetic import (CaptureResolutionConfig, apply_capture_resolution,
                           corrupt_burst, sample_capture_resolution)
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for capture-resolution simulation")
class CaptureResolutionTests(unittest.TestCase):
    def test_minimum_capture_size_and_low_resolution_blur_before_reexpansion(self):
        config = CaptureResolutionConfig(probability=1, min_scale=.25, max_scale=.25)
        for side in (256, 512):
            with self.subTest(side=side):
                sampled = sample_capture_resolution(side, 71, .65, config)
                self.assertEqual(sampled["side"], side//4)
                image = torch.from_numpy(np.random.default_rng(8).uniform(
                    0, 1, (3, side, side)).astype(np.float32))
                original = image.clone()
                sigma = sampled["blurSigma"]
                low = F.interpolate(image[None], size=(side//4, side//4), mode="area")
                radius = max(1, int(np.ceil(3*sigma)))
                axis = torch.arange(-radius, radius+1, dtype=low.dtype)
                kernel = torch.exp(-axis.square()/(2*sigma*sigma))
                kernel /= kernel.sum()
                expected = F.conv2d(F.pad(low, (radius, radius, 0, 0), mode="replicate"),
                    kernel.reshape(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
                expected = F.conv2d(F.pad(expected, (0, 0, radius, radius), mode="replicate"),
                    kernel.reshape(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
                expected = F.interpolate(expected, size=(side, side), mode="bilinear",
                                         align_corners=False)[0]
                result = apply_capture_resolution(image, sampled)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                torch.testing.assert_close(image, original, rtol=0, atol=0)
                self.assertEqual(result.shape, image.shape)
                self.assertGreaterEqual(float(result.min()), 0)
                self.assertLessEqual(float(result.max()), 1)

    def test_downsampling_removes_unresolvable_texture_and_preserves_constant_color(self):
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        checker = ((x+y)%2).float().expand(3, -1, -1)
        reduced = apply_capture_resolution(checker, {"side": 64, "blurSigma": .35})
        torch.testing.assert_close(reduced, torch.full_like(checker, .5), rtol=0, atol=1e-7)
        color = torch.tensor([.15, .4, .8])[:, None, None].expand(3, 256, 256)
        torch.testing.assert_close(apply_capture_resolution(color, {"side": 64, "blurSigma": .35}),
                                   color, rtol=0, atol=2e-7)
        self.assertIs(apply_capture_resolution(color, {"side": 256, "blurSigma": 0}), color)

    def test_resolution_draws_are_reproducible_bounded_and_independent_of_severity(self):
        config = CaptureResolutionConfig(probability=1, min_scale=.25, max_scale=.75,
                                         blur_sigma=.4)
        for side in (256, 512):
            draws = []
            for seed in range(32):
                first = sample_capture_resolution(side, seed, .2, config)
                self.assertEqual(first, sample_capture_resolution(side, seed, .9, config))
                self.assertEqual(first, sample_capture_resolution(side, seed, .2, config))
                self.assertGreaterEqual(first["side"], side//4)
                self.assertLessEqual(first["side"], side*3//4)
                self.assertGreaterEqual(first["blurSigma"], .2)
                self.assertLessEqual(first["blurSigma"], .4)
                draws.append(first["side"])
            self.assertGreater(len(set(draws)), 20)
            self.assertEqual(sample_capture_resolution(side, 1, 0, config),
                             {"side": side, "blurSigma": 0})
            self.assertEqual(sample_capture_resolution(side, 1, .9,
                CaptureResolutionConfig(probability=0)), {"side": side, "blurSigma": 0})
            fractional = CaptureResolutionConfig(probability=1, min_scale=.3, max_scale=.3)
            self.assertEqual(sample_capture_resolution(side, 1, .65, fractional)["side"], round(side*.3))

    def test_complete_corruption_is_deterministic_and_preserves_other_random_streams(self):
        clean = np.random.default_rng(3).uniform(-1, 1, (3, 256, 256)).astype(np.float32)
        original = clean.copy()
        enabled = CaptureResolutionConfig(probability=1, min_scale=.25, max_scale=.25)
        disabled = CaptureResolutionConfig(probability=0)
        first = corrupt_burst(clean, 9, 3, .65, resolution=enabled)
        np.testing.assert_array_equal(first, corrupt_burst(clean, 9, 3, .65, resolution=enabled))
        np.testing.assert_array_equal(clean, original)
        self.assertEqual(first.shape, (3, 3, 256, 256))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        self.assertLessEqual(float(np.abs(first).max()), 1.)
        self.assertFalse(np.array_equal(first[0], first[1]))
        unmodified = corrupt_burst(clean, 9, 3, .65, resolution=disabled)
        self.assertFalse(np.array_equal(first, unmodified))
        with patch("synthetic.apply_capture_resolution", side_effect=lambda image, _: image):
            bypassed = corrupt_burst(clean, 9, 3, .65, resolution=enabled)
        np.testing.assert_array_equal(bypassed, unmodified)
        overshoot = clean*1.4
        np.testing.assert_array_equal(corrupt_burst(overshoot, 9, 3, 0, resolution=enabled),
                                      np.repeat(overshoot[None], 3, axis=0))

    def test_invalid_resolution_controls_are_rejected(self):
        for field in ("probability", "min_scale", "max_scale", "blur_sigma"):
            for value in (-.01, 1.01, float("nan"), float("inf"), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    CaptureResolutionConfig(**{field: value})
        for options in ({"min_scale": .24}, {"max_scale": .24},
                        {"min_scale": .8, "max_scale": .7}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CaptureResolutionConfig(**options)


if __name__ == "__main__":
    unittest.main()
