"""Camera-artifact checks independent of decoder accuracy or payload labels."""

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
    from synthetic import (PhotometricConfig, apply_color_exposure, apply_overlays,
                           apply_motion_blur, corrupt_burst, motion_blur_grids,
                           overlay_alpha, rolling_shutter_maps, sample_motion_blur,
                           sample_photometric, sample_rolling_shutter)
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

    @staticmethod
    def overlay(**changes):
        return {"cx": .5, "cy": .5, "width": .5, "height": .5,
                "shape": "rectangle", "kind": "shadow", "color": [0., 0., 0.],
                "angle": 0, "opacity": .4, "feather": .1, "gradient": 0,
                "driftX": 0, "driftY": 0, "phase": np.pi/2, **changes}

    def test_shadows_attenuate_and_reflections_add_without_changing_outside(self):
        image = torch.full((3, 256, 256), .5)
        base = self.overlay()
        alpha = overlay_alpha(256, base, 0)
        self.assertAlmostEqual(float((alpha > 0).float().mean()), .25, delta=.005)
        self.assertTrue(torch.any((alpha > 0) & (alpha < .4)))
        shadow = apply_overlays(image, [base])
        reflected = apply_overlays(image, [self.overlay(kind="reflection", color=[1., .8, .75])])
        torch.testing.assert_close(shadow[:, 128, 128], torch.full((3,), .3))
        torch.testing.assert_close(reflected[:, 128, 128], torch.tensor([.9, .82, .8]))
        for result in (shadow, reflected):
            torch.testing.assert_close(result[:, alpha == 0], image[:, alpha == 0], rtol=0, atol=0)
        self.assertIs(apply_overlays(image, []), image)
        bright = apply_overlays(torch.full_like(image, .9), [self.overlay(kind="reflection", color=[1]*3)])
        self.assertEqual(float(bright.max()), 1.)

    def test_rotation_ellipse_and_gradient_produce_different_soft_shapes(self):
        base = self.overlay(width=.6, height=.35)
        plain = overlay_alpha(128, base, 0)
        rotated = overlay_alpha(128, {**base, "angle": .65}, 0)
        ellipse = overlay_alpha(128, {**base, "shape": "ellipse"}, 0)
        gradient = overlay_alpha(128, {**base, "gradient": .4}, 0)
        self.assertGreater(float((plain-rotated).abs().mean()), .01)
        self.assertLess(int((ellipse > 0).sum()), int((plain > 0).sum()))
        self.assertLess(float(gradient[64, 48]), float(gradient[64, 80]))
        for alpha in (plain, rotated, ellipse, gradient):
            self.assertTrue(torch.isfinite(alpha).all())
            self.assertGreaterEqual(float(alpha.min()), 0)
            self.assertLessEqual(float(alpha.max()), .4+1e-6)

    def test_seeded_trajectories_include_exposure_signs_and_varied_overlays(self):
        config = PhotometricConfig(shade_probability=1)
        signs, kinds, shapes = set(), set(), set()
        saw_exposure_off = saw_balance_off = False
        for seed in range(48):
            first = sample_photometric(np.random.default_rng(seed), 7, .65, config)
            repeated = sample_photometric(np.random.default_rng(seed), 7, .65, config)
            gains, exposure = first["whiteBalanceGains"], first["exposureStops"]
            np.testing.assert_array_equal(gains, repeated["whiteBalanceGains"])
            np.testing.assert_array_equal(exposure, repeated["exposureStops"])
            self.assertEqual(first["overlays"], repeated["overlays"])
            self.assertLessEqual(float(np.max(np.abs(np.log2(gains)))), .65*.6+1e-6)
            np.testing.assert_allclose(np.prod(gains, axis=1), 1, rtol=1e-6)
            self.assertLessEqual(float(np.max(np.abs(exposure))), .65*1.5+1e-6)
            self.assertLess(float(np.max(np.abs(np.diff(np.log2(gains), axis=0)))), .05)
            self.assertLess(float(np.max(np.abs(np.diff(exposure)))), .05)
            signs.update(np.sign(exposure).tolist())
            saw_exposure_off |= bool(np.all(exposure == 0))
            saw_balance_off |= bool(np.all(gains == 1))
            for overlay in first["overlays"]:
                kinds.add(overlay["kind"])
                shapes.add(overlay["shape"])
                masks = [overlay_alpha(64, overlay, frame) for frame in (0, 6)]
                for alpha in masks:
                    self.assertLessEqual(float(alpha.max()), .65*.45)
                intersection = ((masks[0] > 0) & (masks[1] > 0)).sum()
                union = ((masks[0] > 0) | (masks[1] > 0)).sum()
                self.assertGreater(float(intersection/union), .7)
        self.assertTrue({-1., 1.}.issubset(signs))
        self.assertEqual(kinds, {"shadow", "reflection"})
        self.assertEqual(shapes, {"rectangle", "ellipse"})
        self.assertTrue(saw_exposure_off and saw_balance_off)

    def test_scanline_bands_have_row_color_variation_and_drift(self):
        config = PhotometricConfig(rolling_shutter_probability=1)
        shutter = sample_rolling_shutter(np.random.default_rng(7), 1, config)
        self.assertEqual(shutter, sample_rolling_shutter(np.random.default_rng(7), 1, config))
        # Until rectification tilts the bands, a sensor row shares exposure.
        shutter["tilt"] = 0
        gains, skew = rolling_shutter_maps(256, shutter, 0)
        torch.testing.assert_close(gains[:, :, 0], gains[:, :, -1])
        self.assertGreater(float(gains.std()), .01)
        self.assertFalse(torch.equal(gains[0], gains[1]))
        self.assertLessEqual(float(torch.log2(gains).abs().max()), config.rolling_shutter_stops)
        self.assertLessEqual(float(skew.abs().max()), config.rolling_shutter_skew)
        changed, shifted = rolling_shutter_maps(256, shutter, 2)
        self.assertFalse(torch.equal(gains, changed))
        self.assertFalse(torch.equal(skew, shifted))
        # Geometric scanline correction remains much smaller than a cell.
        maximum = sample_rolling_shutter(np.random.default_rng(7), 1,
                    PhotometricConfig(rolling_shutter_probability=1, rolling_shutter_skew=.75))
        _, offsets = rolling_shutter_maps(512, maximum, 3)
        self.assertLess(float(offsets.abs().max())+.9, 2.)

    @staticmethod
    def motion(**changes):
        return {"linearPixels": 0., "rotationDegrees": 0., "zoomFraction": 0.,
                "angle": 0., "angleDrift": 0., "cx": .5, "cy": .5,
                "phase": np.pi/2, "velocity": .3, "samples": 7, **changes}

    @staticmethod
    def spatial_moments(image):
        y, x = torch.meshgrid(torch.arange(image.shape[-2]), torch.arange(image.shape[-1]), indexing="ij")
        p = image[0]/image[0].sum()
        x, y = x-(p*x).sum(), y-(p*y).sum()
        return float((p*x*x).sum()), float((p*y*y).sum()), float((p*x*y).sum())

    def test_integrated_linear_blur_follows_arbitrary_direction(self):
        image = torch.zeros((3, 65, 65))
        image[:, 32, 32] = 1
        diagonal = apply_motion_blur(image, self.motion(linearPixels=6., angle=np.pi/4))
        vx, vy, cov = self.spatial_moments(diagonal)
        self.assertGreater(vx, 1)
        self.assertAlmostEqual(vx, vy, places=4)
        self.assertGreater(cov/np.sqrt(vx*vy), .8)
        self.assertLess(float(diagonal.max()), 1)
        self.assertIs(apply_motion_blur(image, None), image)

    def test_rotation_and_radial_zoom_have_different_spatial_blur(self):
        image = torch.zeros((3, 257, 257))
        image[:, 128, 208] = 1
        rotation = apply_motion_blur(image, self.motion(rotationDegrees=5.))
        zoom = apply_motion_blur(image, self.motion(zoomFraction=.05))
        rx, ry, _ = self.spatial_moments(rotation)
        zx, zy, _ = self.spatial_moments(zoom)
        self.assertGreater(ry, 10*rx+.1)  # tangential at a right-of-center impulse
        self.assertGreater(zx, 10*zy+.1)  # radial at the same impulse
        center = torch.zeros_like(image)
        center[:, 128, 128] = 1
        self.assertAlmostEqual(float(apply_motion_blur(center, self.motion(rotationDegrees=5.))[0, 128, 128]), 1., places=5)

    def test_motion_trajectories_and_centered_grids_have_known_bounds(self):
        config = PhotometricConfig(motion_probability=1)
        saw = [False, False, False]
        for seed in range(32):
            motion = sample_motion_blur(np.random.default_rng(seed), .65, config)
            self.assertEqual(motion, sample_motion_blur(np.random.default_rng(seed), .65, config))
            for i, (name, bound) in enumerate((("linearPixels", 2.5), ("rotationDegrees", .35), ("zoomFraction", .006))):
                self.assertLessEqual(abs(motion[name]), .65*bound)
                saw[i] |= motion[name] != 0
            grids = motion_blur_grids(65, motion, 0)
            y, x = torch.meshgrid(torch.linspace(-1, 1, 65), torch.linspace(-1, 1, 65), indexing="ij")
            reference = torch.stack((x, y), dim=-1)
            torch.testing.assert_close(grids[3], reference, atol=2e-7, rtol=0)
            radius = 64*np.hypot(max(motion["cx"], 1-motion["cx"]), max(motion["cy"], 1-motion["cy"]))
            bound = motion["linearPixels"]/2 + 2*radius*np.sin(np.deg2rad(abs(motion["rotationDegrees"]))/4) + radius*abs(motion["zoomFraction"])/2
            displacement = torch.linalg.vector_norm((grids-reference)*32, dim=-1)
            self.assertLessEqual(float(displacement.max()), bound+1e-5)
            self.assertFalse(torch.equal(grids, motion_blur_grids(65, motion, 2)))
        self.assertTrue(all(saw))
        sampled = [sample_motion_blur(np.random.default_rng(i), 1, PhotometricConfig()) for i in range(32)]
        self.assertTrue(any(item is None for item in sampled))
        self.assertTrue(any(item is not None for item in sampled))
        for config in (PhotometricConfig(motion_probability=0),
                       PhotometricConfig(linear_blur_pixels=0, rotation_blur_degrees=0, zoom_blur_fraction=0)):
            self.assertIsNone(sample_motion_blur(np.random.default_rng(7), 1, config))

    def test_controls_disable_only_the_selected_artifact(self):
        config = PhotometricConfig(white_balance_stops=0, exposure_stops=0,
                                   shade_opacity=0, shade_probability=1,
                                   rolling_shutter_stops=0, rolling_shutter_skew=0)
        result = sample_photometric(np.random.default_rng(8), 7, 1, config)
        np.testing.assert_array_equal(result["whiteBalanceGains"], np.ones((7, 3)))
        np.testing.assert_array_equal(result["exposureStops"], np.zeros(7))
        self.assertEqual(result["overlays"], [])
        self.assertIsNone(sample_rolling_shutter(np.random.default_rng(8), 1, config))
        self.assertIsNone(sample_rolling_shutter(np.random.default_rng(8), 1,
                          PhotometricConfig(rolling_shutter_probability=0)))
        result = sample_photometric(np.random.default_rng(8), 7, 1,
                                    PhotometricConfig(shade_probability=0))
        self.assertEqual(result["overlays"], [])
        for stops, skew in ((0, .5), (.65, 0)):
            shutter = sample_rolling_shutter(np.random.default_rng(8), 1,
                PhotometricConfig(rolling_shutter_probability=1,
                                  rolling_shutter_stops=stops, rolling_shutter_skew=skew))
            gains, offsets = rolling_shutter_maps(64, shutter, 0)
            self.assertEqual(bool(torch.all(gains == 1)), stops == 0)
            self.assertEqual(bool(torch.all(offsets == 0)), skew == 0)
        # The mixture retains bursts with no rolling shutter, even at severity 1.
        sampled = [sample_rolling_shutter(np.random.default_rng(i), 1, PhotometricConfig()) for i in range(32)]
        self.assertTrue(any(item is None for item in sampled))
        self.assertTrue(any(item is not None for item in sampled))

    def test_zero_severity_preserves_exact_input_and_corruptions_are_deterministic(self):
        clean = np.linspace(-1.5, 1.5, 3*256*256, dtype=np.float32).reshape(3, 256, 256)
        np.testing.assert_array_equal(corrupt_burst(clean, 8, 2, 0), np.repeat(clean[None], 2, axis=0))
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
                              ("shade_opacity", .95), ("shade_probability", 1),
                              ("rolling_shutter_stops", 3), ("rolling_shutter_probability", 1),
                              ("rolling_shutter_skew", .75), ("motion_probability", 1),
                              ("linear_blur_pixels", 12), ("rotation_blur_degrees", 5),
                              ("zoom_blur_fraction", .05)):
            for value in (-.1, maximum+.01, np.nan, np.inf, -np.inf, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PhotometricConfig(**{name: value})
        for value in (0, 2, 4, 16, 3.5, True, None):
            with self.subTest(motion_samples=value), self.assertRaises(ValueError):
                PhotometricConfig(motion_samples=value)


if __name__ == "__main__":
    unittest.main()
