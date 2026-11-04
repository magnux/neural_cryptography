"""Burst membership and fixed random held-out groups of distinct observations."""

import copy
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bursts import BurstIndex, EVALUATION_POLICY, EVALUATION_SEED


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
        rng = random.Random(7)
        evolving = [index.train_sample(rows[2], rng) for _ in range(12)]
        self.assertGreater(len({tuple(m.id for m in sample.members) for sample in evolving}), 1)
        self.assertGreater(len({sample.members.index(rows[2]) for sample in evolving}), 1)

    def test_evaluation_fixed_random_nonoverlap_and_remainder_are_explicit(self):
        rows = [Exposure(i, burst=burst) for burst in ("a", "b") for i in range(32)]
        first, stats = BurstIndex(rows, 5).eval_samples()
        repeat, other = BurstIndex(list(reversed(rows)), 5).eval_samples()
        self.assertEqual([[m.id for m in r.members] for r in first],
                         [[m.id for m in r.members] for r in repeat])
        ids = [member.id for sample in first for member in sample.members]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(stats, {"method": EVALUATION_POLICY, "seed": EVALUATION_SEED,
            "framesPerInput": 5, "inputCaptures": 64,
            "usedCaptures": 60, "discardedCaptures": 4, "insufficientBursts": 0,
            "completeBursts": 12})
        self.assertEqual(stats, other)
        for sample in first:
            self.assertEqual(sample.id, sample.members[0].id)
            self.assertEqual(len({m.metadata["burstId"] for m in sample.members}), 1)
        # Randomizing only slot order within chronological windows would fail:
        # the chosen member sets themselves must change before grouping.
        expected_chronological = {frozenset(range(start, start+5)) for start in range(0, 30, 5)}
        actual_sets = {frozenset(m.value for m in sample.members) for sample in first}
        self.assertNotEqual(actual_sets, expected_chronological)
        self.assertTrue(any([m.value for m in sample.members] != sorted(m.value for m in sample.members)
                            for sample in first))

    def test_evaluation_draw_ignores_paths_timestamps_target_hashes_and_other_bursts(self):
        rows = [Exposure(i, burst=burst, phase=phase, manifest=f"/old/{burst}-{phase}/manifest.json")
                for burst in ("a", "b") for phase in (0, 1) for i in range(12)]

        def member_ids(values):
            return [tuple(m.id for m in sample.members) for sample in BurstIndex(values, 5).eval_samples()[0]]

        expected = member_ids(rows)
        relocated = copy.deepcopy(rows)
        for row in relocated:
            row.manifest = Path("/new/renamed")/row.manifest.parent.name/"manifest.json"
            row.metadata["timestamp"] = 10000-row.metadata["timestamp"]
            row.metadata["hashes"] = {"labels": "e"*64, "mask": "f"*64, "expected": "0"*64}
        random.Random(42).shuffle(relocated)
        self.assertEqual(member_ids(relocated), expected)
        extra = [Exposure(i, session="unrelated", burst="added") for i in range(17)]
        together = member_ids(extra+rows)
        original_ids = {r.id for r in rows}
        self.assertEqual([ids for ids in together if ids[0] in original_ids], expected)
        for sample in BurstIndex(rows, 5).eval_samples()[0]:
            self.assertEqual(len({(m.metadata["burstId"], m.metadata["info"]["phase"])
                                  for m in sample.members}), 1)

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

    def test_synthetic_evaluation_does_not_restore_chronological_frame_order(self):
        rows = [Exposure(i, source="synthetic", synthetic={"frame": 8-i}) for i in range(8)]
        windows, _ = BurstIndex(rows, 4).eval_samples()
        expected = [[member.id for member in sample.members] for sample in windows]
        for row in rows:
            row.metadata["synthetic"]["frame"] = row.value
        other, _ = BurstIndex(rows, 4).eval_samples()
        self.assertEqual([[member.id for member in sample.members] for sample in other], expected)
        self.assertNotEqual([member.value for sample in windows for member in sample.members], list(range(8)))

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
