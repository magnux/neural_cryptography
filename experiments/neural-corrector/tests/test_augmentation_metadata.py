"""Spatial augmentation flags preserve canonical byte membership for RS loss."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from augmentation import PhaseAugmentation
from phases import PHASE_MASKS


class AugmentationMetadataTests(unittest.TestCase):
    @staticmethod
    def batch(count=8):
        # Distinct slots, asymmetric masks and labels expose swaps and both flips.
        pixels = np.arange(count*32*16*16, dtype=np.float32).reshape(count, 32, 16, 16)
        labels = (np.arange(count*4*4).reshape(count, 4, 4) % 16).astype(np.int64)
        masks = np.ones_like(labels, dtype=np.bool_)
        masks[:, 0, 1] = False
        masks[:, 2, 0] = False
        return pixels, labels, masks

    def test_flags_match_transforms_and_preserve_rng_draw_order(self):
        pixels, labels, masks = self.batch()
        originals = tuple(value.copy() for value in (pixels, labels, masks))
        augmentation = PhaseAugmentation(seed=51, flip_probability=.5)
        expected_rng = np.random.default_rng(np.random.SeedSequence(augmentation.seed))
        # Test consecutive calls too: drawing flags must not consume extra RNG.
        for _ in range(2):
            result, targets, result_masks, flags = augmentation(pixels, labels, masks)
            self.assertEqual(flags.shape, (len(pixels), 2))
            self.assertEqual(flags.dtype, np.bool_)
            for index in range(len(pixels)):
                delta = int(expected_rng.choice(PHASE_MASKS))
                horizontal = expected_rng.random() < .5
                vertical = expected_rng.random() < .5
                expected = PhaseAugmentation.transform(
                    pixels[index], labels[index], masks[index], delta=delta,
                    horizontal=horizontal, vertical=vertical)
                np.testing.assert_array_equal(flags[index], [horizontal, vertical])
                for actual, wanted in zip((result[index], targets[index], result_masks[index]), expected):
                    np.testing.assert_array_equal(actual, wanted)
                canonical_target, canonical_mask = targets[index], result_masks[index]
                for axis, enabled in ((-1, flags[index, 0]), (-2, flags[index, 1])):
                    if enabled:
                        canonical_target = np.flip(canonical_target, axis=axis)
                        canonical_mask = np.flip(canonical_mask, axis=axis)
                np.testing.assert_array_equal(canonical_mask, masks[index])
                relabeled = labels[index].copy()
                relabeled[masks[index]] ^= delta
                np.testing.assert_array_equal(canonical_target, relabeled)
            for value in (result, targets, result_masks, flags):
                self.assertTrue(value.flags.c_contiguous)
        for actual, original in zip((pixels, labels, masks), originals):
            np.testing.assert_array_equal(actual, original)

    def test_disabled_and_forced_flips(self):
        pixels, labels, masks = self.batch(1)
        for probability in (0., 1.):
            result = PhaseAugmentation(seed=7, flip_probability=probability,
                                       permutations=False)(pixels, labels, masks)
            np.testing.assert_array_equal(result[3], [[bool(probability), bool(probability)]])
            for actual, original in zip(result[:3], (pixels, labels, masks)):
                expected = np.flip(original, axis=(-2, -1)) if probability else original
                np.testing.assert_array_equal(actual, expected)

    def test_mismatched_batch_lengths_reject_instead_of_silent_zip_truncation(self):
        pixels, labels, masks = self.batch(2)
        augmentation = PhaseAugmentation(seed=7)
        for arrays in ((pixels, labels[:1], masks), (pixels, labels, masks[:1]),
                       (pixels[:0], labels[:0], masks[:0])):
            with self.subTest(lengths=[len(value) for value in arrays]), self.assertRaises(ValueError):
                augmentation(*arrays)


if __name__ == "__main__":
    unittest.main()
