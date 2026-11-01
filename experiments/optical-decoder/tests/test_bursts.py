"""Burst membership, distinct observations, and deterministic held-out windows."""

import copy
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bursts import BurstIndex


class Exposure:
    def __init__(self, index, *, source="real", burst="physical-burst", session="receiver",
                 phase=0, split="train", manifest="manifest.json", **changes):
        self.id = f"{source}-{session}-{burst}-{phase}-{index}"
        self.root, self.manifest = Path("fixture"), Path(manifest)
        self.shape = (64, 16)
        self.metadata = {"id": self.id, "sessionId": session, "senderSessionId": "sender",
            "payloadId": "a"*64, "source": source, "split": split, "burstId": burst,
            "timestamp": index*100, "info": {"size": 64, "colors": 16, "phase": phase},
            "hashes": {"labels": "b"*64, "mask": "c"*64, "expected": "d"*64}, **changes}
        self.value = index

    def tensors(self):
        return (np.full((3, 256, 256), self.value, dtype=np.float32),
                np.zeros((64, 64), dtype=np.int64), np.ones((64, 64), dtype=bool))


class BurstTests(unittest.TestCase):
    def test_training_distinct_subset_order_and_stack_shape(self):
        rows = [Exposure(i) for i in range(8)]
        original = copy.deepcopy([row.metadata for row in rows])
        index = BurstIndex(rows, 4)
        index.validate_training()
        first = index.train_sample(rows[2], random.Random(7))
        repeat = index.train_sample(rows[2], random.Random(7))
        self.assertEqual([r.id for r in first.members], [r.id for r in repeat.members])
        self.assertEqual(len({r.id for r in first.members}), 4)
        self.assertIn(rows[2], first.members)
        self.assertIs(first.anchor, rows[2])
        pixels, labels, mask = first.tensors()
        self.assertEqual(pixels.shape, (12, 256, 256))
        self.assertEqual(labels.shape, (64, 64))
        for offset, member in enumerate(first.members):
            np.testing.assert_array_equal(pixels[3*offset:3*offset+3], member.value)
        self.assertTrue(mask.all())
        self.assertEqual([row.metadata for row in rows], original)
        orders = {tuple(r.id for r in index.train_sample(rows[2], random.Random(seed)).members)
                  for seed in range(12)}
        self.assertGreater(len(orders), 1)

    def test_evaluation_nonoverlap_chronology_and_remainder_are_explicit(self):
        rows = [Exposure(i, burst=burst) for burst in ("a", "b") for i in range(9)]
        first, stats = BurstIndex(rows, 4).eval_samples()
        repeat, other = BurstIndex(list(reversed(rows)), 4).eval_samples()
        self.assertEqual([[m.id for m in r.members] for r in first],
                         [[m.id for m in r.members] for r in repeat])
        ids = [member.id for sample in first for member in sample.members]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(stats, {"framesPerInput": 4, "inputCaptures": 18,
            "usedCaptures": 16, "discardedCaptures": 2, "insufficientBursts": 0,
            "completeBursts": 4})
        self.assertEqual(stats, other)
        for sample in first:
            self.assertEqual([r.value for r in sample.members], sorted(r.value for r in sample.members))
            self.assertEqual(sample.id, sample.members[0].id)

    def test_never_combines_sources_phases_sessions_splits_or_bursts(self):
        variants = [{"source": "synthetic"}, {"phase": 1}, {"session": "other"},
                    {"split": "validation"}, {"burst": "other-burst"},
                    {"manifest": "other-manifest.json"}, {"payloadId": "e"*64},
                    {"senderSessionId": "other-sender"}]
        for variant in variants:
            with self.subTest(variant=variant):
                rows = [Exposure(i) for i in range(3)] + [Exposure(99, **variant)]
                index = BurstIndex(rows, 4)
                with self.assertRaisesRegex(ValueError, "fewer than 4"):
                    index.validate_training()
                windows, stats = index.eval_samples()
                self.assertEqual(windows, [])
                self.assertEqual(stats["discardedCaptures"], 4)
                self.assertEqual(stats["insufficientBursts"], 2)

    def test_synthetic_order_uses_frame_number_not_inherited_target_timestamp(self):
        rows = [Exposure(i, source="synthetic", synthetic={"frame": 8-i}) for i in range(8)]
        windows, _ = BurstIndex(rows, 4).eval_samples()
        self.assertEqual([member.value for member in windows[0].members], [7, 6, 5, 4])

    def test_short_missing_duplicate_or_inconsistent_records_reject(self):
        rows = [Exposure(i) for i in range(4)]
        broken = copy.deepcopy(rows)
        broken[1].metadata["hashes"]["labels"] = "f"*64
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            BurstIndex(broken, 4)
        broken = copy.deepcopy(rows)
        broken[0].metadata.pop("burstId")
        with self.assertRaisesRegex(ValueError, "burstId"):
            BurstIndex(broken, 4)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            BurstIndex(rows+[rows[0]], 4)
        with self.assertRaisesRegex(ValueError, "fewer than 4"):
            BurstIndex(rows[:3], 4).train_sample(rows[0], random.Random(1))
        for value in (0, 65, True, False, 4.):
            with self.assertRaisesRegex(ValueError, "1 to 64"):
                BurstIndex(rows, value)

    def test_five_and_large_frame_counts_use_distinct_complete_groups(self):
        for frames in (5, 32, 64):
            with self.subTest(frames=frames):
                rows = [Exposure(i) for i in range(2*frames+1)]
                index = BurstIndex(rows, frames)
                index.validate_training()
                sample = index.train_sample(rows[2], random.Random(7))
                self.assertEqual(len({member.id for member in sample.members}), frames)
                if frames == 5:
                    self.assertEqual(sample.tensors()[0].shape, (15, 256, 256))
                windows, stats = index.eval_samples()
                self.assertEqual(len(windows), 2)
                self.assertEqual(stats["usedCaptures"], 2*frames)
                self.assertEqual(stats["discardedCaptures"], 1)
                self.assertEqual(len({member.id for window in windows for member in window.members}), 2*frames)

    def test_single_frame_keeps_capture_identity_and_shape(self):
        rows = [Exposure(i) for i in range(3)]
        index = BurstIndex(rows, 1)
        index.validate_training()
        sample = index.train_sample(rows[1], random.Random(1))
        self.assertEqual(sample.members, (rows[1],))
        self.assertEqual(sample.tensors()[0].shape, (3, 256, 256))
        windows, stats = index.eval_samples()
        self.assertEqual(len(windows), 3)
        self.assertEqual(stats["discardedCaptures"], 0)


if __name__ == "__main__":
    unittest.main()
