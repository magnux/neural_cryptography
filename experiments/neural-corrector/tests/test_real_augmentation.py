"""Real-only photometric augmentation preserves crops, targets and burst meaning."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from augmentation import RealAugmentation


class RealAugmentationTests(unittest.TestCase):
    def test_real_only_does_not_mutate_source_and_preserves_capture_overshoot(self):
        pixels = np.full((2, 15, 16, 16), 1.4, dtype=np.float32)
        pixels[:, :, ::2] = -1.4
        original = pixels.copy()
        transform = RealAugmentation(seed=7, noise_std=0., tint_stops=.01, exposure_stops=.01)
        augmented = transform(pixels, ["real", "synthetic"])
        np.testing.assert_array_equal(pixels, original)
        np.testing.assert_array_equal(augmented[1], original[1])
        self.assertFalse(np.array_equal(augmented[0], original[0]))
        self.assertGreater(float(augmented[0].max()), 1.)
        self.assertLess(float(augmented[0].min()), -1.)
        self.assertEqual(augmented.dtype, np.float32)

    def test_shared_gain_keeps_identical_exposures_identical_without_noise(self):
        pixels = np.full((2, 15, 16, 16), .1, dtype=np.float32)
        augmented = RealAugmentation(seed=8, noise_std=0.)(pixels, ["real", "real"])
        frames = augmented.reshape(2, 5, 3, 16, 16)
        for index in range(1, 5):
            np.testing.assert_array_equal(frames[:, index], frames[:, 0])
        self.assertFalse(np.array_equal(augmented[0], augmented[1]))

    def test_independent_noise_uses_rgb01_units_and_keeps_capture_bounds(self):
        pixels = np.zeros((1, 15, 128, 128), dtype=np.float32)
        augmented = RealAugmentation(seed=11, noise_std=.02, tint_stops=0., exposure_stops=0.)(pixels, ["real"])
        self.assertAlmostEqual(float(augmented.std()), .04, delta=.001)
        self.assertFalse(np.array_equal(augmented[0, :3], augmented[0, 3:6]))
        extreme = RealAugmentation(seed=11, noise_std=1., tint_stops=4., exposure_stops=4.)(pixels, ["real"])
        self.assertGreaterEqual(float(extreme.min()), -1.5)
        self.assertLessEqual(float(extreme.max()), 1.5)
        self.assertTrue(np.isfinite(extreme).all())

    def test_seed_reproducible_but_draws_evolve_and_global_rng_is_untouched(self):
        pixels = np.zeros((2, 15, 16, 16), dtype=np.float32)
        left, right = RealAugmentation(seed=17), RealAugmentation(seed=17)
        np.random.seed(123)
        expected_global = np.random.random(3)
        np.random.seed(123)
        first = left(pixels, ["real", "synthetic"])
        np.testing.assert_array_equal(first, right(pixels, ["real", "synthetic"]))
        second = left(pixels, ["real", "synthetic"])
        np.testing.assert_array_equal(second, right(pixels, ["real", "synthetic"]))
        self.assertFalse(np.array_equal(first[0], second[0]))
        np.testing.assert_array_equal(np.random.random(3), expected_global)

    def test_disabled_identity_and_synthetic_batches_do_not_consume_randomness(self):
        pixels = np.full((1, 15, 8, 8), 1.4, dtype=np.float32)
        disabled = RealAugmentation(seed=1, noise_std=0., tint_stops=0., exposure_stops=0.)
        self.assertFalse(disabled.settings()["enabled"])
        np.testing.assert_array_equal(disabled(pixels, ["real"]), pixels)
        left, right = RealAugmentation(seed=1), RealAugmentation(seed=1)
        np.testing.assert_array_equal(left(pixels, ["synthetic"]), pixels)
        np.testing.assert_array_equal(left(pixels, ["real"]), right(pixels, ["real"]))

    def test_invalid_strength_rejected(self):
        for field in ("noise_std", "tint_stops", "exposure_stops"):
            for value in (-.01, float("nan"), float("inf"), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    RealAugmentation(seed=7, **{field: value})


if __name__ == "__main__":
    unittest.main()
