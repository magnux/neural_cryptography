"""Photometric corruption checks independent of decoder accuracy or payload labels."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from synthetic import (PhotometricConfig, apply_color_exposure,
                           apply_rectangle_shade, corrupt_burst, rectangle_alpha,
                           sample_photometric)
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for synthetic camera corruptions")
class PhotometricTests(unittest.TestCase):
    def test_white_balance_exposure_and_saturation_on_neutral_rgb(self):
        neutral = torch.full((3, 8, 8), .4)
        tinted = apply_color_exposure(neutral, [1.25, 1, .8], 0)
        torch.testing.assert_close(tinted[:, 0, 0], torch.tensor([.5, .4, .32]))
        torch.testing.assert_close(apply_color_exposure(neutral, [1, 1, 1], 1),
                                   torch.full_like(neutral, .8))
        torch.testing.assert_close(apply_color_exposure(neutral, [1, 1, 1], -1),
                                   torch.full_like(neutral, .2))
        saturated = apply_color_exposure(neutral, [1, 1, 1], 2)
        self.assertTrue(torch.all(saturated == 1))

    def test_rectangles_cover_quarter_and_blend_without_changing_outside(self):
        image = torch.full((3, 256, 256), .5)
        shade = {"x": .2, "y": .2, "width": .5, "height": .5,
                 "opacity": .4, "feather": .005, "driftX": 0,
                 "driftY": 0, "phase": 0, "color": 0}
        alpha = rectangle_alpha(256, shade, 0)
        self.assertAlmostEqual(float((alpha > 0).float().mean()), .25, delta=.005)
        self.assertTrue(torch.any((alpha > 0) & (alpha < .4)))
        for color, center in ((0, .3), (1, .7)):
            with self.subTest(color=color):
                result = apply_rectangle_shade(image, {**shade, "color": color})
                torch.testing.assert_close(result[:, alpha == 0], image[:, alpha == 0],
                                           rtol=0, atol=0)
                torch.testing.assert_close(result[:, 128, 128], torch.full((3,), center))
                self.assertGreater(float(result.min()), 0)
                self.assertLess(float(result.max()), 1)
        self.assertIs(apply_rectangle_shade(image, None), image)

    def test_seeded_trajectories_include_both_exposure_signs_and_shade_colors(self):
        config = PhotometricConfig(shade_probability=1)
        signs, shade_colors = set(), set()
        saw_exposure_off = saw_balance_off = False
        for seed in range(48):
            first = sample_photometric(np.random.default_rng(seed), 7, .65, config)
            repeated = sample_photometric(np.random.default_rng(seed), 7, .65, config)
            gains, exposure = first["whiteBalanceGains"], first["exposureStops"]
            np.testing.assert_array_equal(gains, repeated["whiteBalanceGains"])
            np.testing.assert_array_equal(exposure, repeated["exposureStops"])
            self.assertEqual(first["shade"], repeated["shade"])
            self.assertLessEqual(float(np.max(np.abs(np.log2(gains)))), .65*.6+1e-6)
            np.testing.assert_allclose(np.prod(gains, axis=1), 1, rtol=1e-6)
            self.assertLessEqual(float(np.max(np.abs(exposure))), .65*1.5+1e-6)
            # Neighboring captures share a lighting trajectory, not independent jumps.
            self.assertLess(float(np.max(np.abs(np.diff(np.log2(gains), axis=0)))), .05)
            self.assertLess(float(np.max(np.abs(np.diff(exposure)))), .05)
            signs.update(np.sign(exposure).tolist())
            saw_exposure_off |= bool(np.all(exposure == 0))
            saw_balance_off |= bool(np.all(gains == 1))
            shade = first["shade"]
            shade_colors.add(shade["color"])
            area = shade["width"]*shade["height"]
            self.assertGreaterEqual(area, .2)
            self.assertLessEqual(area, .3)
            # Its footprint survives the burst; positional drift is small.
            masks = [rectangle_alpha(256, shade, frame) for frame in (0, 6)]
            for alpha in masks:
                self.assertLessEqual(float(alpha.max()), .65*.45)
                self.assertAlmostEqual(float((alpha > 0).float().mean()), area, delta=.007)
            intersection = ((masks[0] > 0) & (masks[1] > 0)).sum()
            union = ((masks[0] > 0) | (masks[1] > 0)).sum()
            self.assertGreater(float(intersection/union), .85)
        self.assertTrue({-1., 1.}.issubset(signs))
        self.assertEqual(shade_colors, {0, 1})
        self.assertTrue(saw_exposure_off and saw_balance_off)

    def test_zero_controls_disable_their_effects(self):
        config = PhotometricConfig(white_balance_stops=0, exposure_stops=0,
                                   shade_opacity=0, shade_probability=1)
        result = sample_photometric(np.random.default_rng(8), 7, 1, config)
        np.testing.assert_array_equal(result["whiteBalanceGains"], np.ones((7, 3)))
        np.testing.assert_array_equal(result["exposureStops"], np.zeros(7))
        self.assertIsNone(result["shade"])
        result = sample_photometric(np.random.default_rng(8), 7, 1,
                                    PhotometricConfig(shade_probability=0))
        self.assertIsNone(result["shade"])
        clean = np.linspace(-1.5, 1.5, 3*256*256, dtype=np.float32).reshape(3, 256, 256)
        np.testing.assert_array_equal(corrupt_burst(clean, 8, 2, 0),
                                      np.repeat(clean[None], 2, axis=0))

    def test_corrupted_outputs_are_deterministic_finite_at_both_sizes(self):
        for side in (256, 512):
            with self.subTest(side=side):
                clean = np.random.default_rng(12).uniform(-1, 1, (3, side, side)).astype(np.float32)
                frames = corrupt_burst(clean, 4, 2, .8)
                self.assertEqual(frames.shape, (2, 3, side, side))
                self.assertEqual(frames.dtype, np.float32)
                self.assertTrue(np.isfinite(frames).all())
                self.assertLessEqual(float(np.max(np.abs(frames))), 1)
                self.assertFalse(np.array_equal(frames[0], frames[1]))
                np.testing.assert_array_equal(frames, corrupt_burst(clean, 4, 2, .8))

    def test_invalid_photometric_options_are_rejected(self):
        for name, maximum in (("white_balance_stops", 4), ("exposure_stops", 4),
                              ("shade_opacity", .95), ("shade_probability", 1)):
            for value in (-.1, maximum+.01, np.nan, np.inf, -np.inf, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PhotometricConfig(**{name: value})


if __name__ == "__main__":
    unittest.main()
