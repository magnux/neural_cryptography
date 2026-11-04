"""Variable observations, exact pairing and missing-slot training semantics."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
from bursts import BurstIndex, VARIABLE_EVALUATION_POLICY
from sampling import BalancedSampler

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from augmentation import RealAugmentation, PhaseAugmentation
    from checkpoint import load_model_checkpoint
    from data import Dataset
    from train import draw_training_bursts, training_tensors, run_training
    torch.set_num_threads(1)


class Exposure:
    def __init__(self, index, *, source="real", session="receiver", burst="burst", split="train", phase=0, group="logical"):
        self.id = f"{source}/{session}/{burst}/{phase}/{index}"
        self.manifest = Path(f"{source}/manifest.json")
        self.root = self.manifest.parent
        self.shape = (64, 16)
        self.value = index/32
        self.metadata = {"source": source, "sessionId": session, "senderSessionId": "sender",
            "payloadId": "a"*64, "split": split, "burstId": burst, "timestamp": index*100,
            "size": 64, "colors": 16, "info": {"size": 64, "colors": 16, "phase": phase, "phases": 4, "permute": True, "id": group},
            "hashes": {"labels": format(phase+10, "x")*64, "mask": "c"*64, "expected": "d"*64}}
        if source == "synthetic":
            self.metadata["synthetic"] = {"phaseGroupId": group}

    def tensors(self):
        return (np.full((3, 256, 256), self.value, dtype=np.float32),
                np.full((64, 64), (0,15,7,8)[self.metadata["info"]["phase"]], dtype=np.int64), np.ones((64, 64), dtype=bool))


class VariableBurstTests(unittest.TestCase):
    def test_nested_random_evaluation_members_and_slots_are_fixed_for_every_count(self):
        rows = [Exposure(i, burst=burst, group=burst, phase=phase) for burst in ("a", "b") for phase in range(4) for i in range(9)]
        index = BurstIndex(rows, 8, min_frames=4, presence=True)
        complete, stats = index.eval_samples(actual_frames=8)
        self.assertEqual(len(complete), 8)
        self.assertEqual(stats["remainderCaptures"], 8)
        self.assertEqual(stats["method"], VARIABLE_EVALUATION_POLICY)
        original_ids = {sample.id for sample in complete}
        for count in (4, 5, 6, 7, 8):
            candidates, report = index.eval_samples(actual_frames=count)
            self.assertEqual({sample.id for sample in candidates}, original_ids)
            self.assertEqual(report["usedCaptures"], 8*count)
            self.assertEqual(report["unusedCapacityCaptures"], 8*(8-count))
            self.assertEqual(report["discardedCaptures"], 72-8*count)
            observed = []
            for partial, full in zip(candidates, complete):
                self.assertEqual(partial.members, full.members[:count])
                self.assertEqual(partial.slots, full.slots[:count])
                self.assertEqual(partial.id, partial.members[0].id)
                observed.extend(member.id for member in partial.members)
            self.assertEqual(len(observed), len(set(observed)))
        reversed_index = BurstIndex(list(reversed(rows)), 8, min_frames=4, presence=True)
        again, _ = reversed_index.eval_samples(actual_frames=5)
        self.assertEqual([(s.id, s.slots, tuple(m.id for m in s.members)) for s in again],
                         [(s.id, s.slots, tuple(m.id for m in s.members))
                          for s in index.eval_samples(actual_frames=5)[0]])

    def test_training_draws_variable_counts_distinct_members_and_scattered_zero_slots(self):
        rows = [Exposure(i, phase=phase) for phase in range(4) for i in range(3)]
        index = BurstIndex(rows, 8, min_frames=4, presence=True)
        index.validate_training()
        rng, counts, placements = random.Random(17), set(), set()
        for _ in range(24):
            sample = index.train_sample(rows[1], rng)
            counts.add(len(sample.members)); placements.add(sample.slots)
            self.assertEqual(len({m.id for m in sample.members}), len(sample.members))
            self.assertIn(rows[1], sample.members)
        self.assertEqual(counts, {4, 5, 6, 7, 8})
        self.assertGreater(len(placements), 4)
        sample = index.train_sample(rows[1], rng, actual_frames=5)
        pixels, _, _ = sample.tensors()
        self.assertEqual(pixels.shape, (32, 256, 256))
        for slot in range(8):
            self.assertTrue(np.all(pixels[24+slot] == int(slot in sample.slots)))
            if slot not in sample.slots:
                self.assertFalse(np.any(pixels[3*slot:3*slot+3]))
        with self.assertRaisesRegex(ValueError, "two distinct captures per phase"):
            BurstIndex(rows[:7], 8, min_frames=4, presence=True).validate_training()
        with self.assertRaises(ValueError):
            index.train_sample(rows[0], rng, actual_frames=3)

    def test_canonical_targets_and_phase_slots_are_consistent(self):
        rows = [Exposure(i, phase=phase) for phase in range(4) for i in range(3)]
        index = BurstIndex(rows, 8, min_frames=4, presence=True)
        for count in range(4, 9):
            sample = index.train_sample(rows[7], random.Random(9), actual_frames=count)
            pixels, labels, _ = sample.tensors()
            self.assertTrue(np.all(labels == 0))
            self.assertEqual(set(m.metadata["info"]["phase"] for m in sample.members), set(range(4)))
            for member, slot in zip(sample.members, sample.slots):
                self.assertEqual(slot//2, member.metadata["info"]["phase"])
            self.assertTrue(np.all(pixels[24:, 0, 0].reshape(4, 2).sum(1) >= 1))
            self.assertEqual(int(pixels[24:, 0, 0].sum()), count)

    def test_phase_groups_reject_incomplete_or_independent_synthetic_trajectories(self):
        rows = [Exposure(i, phase=phase) for phase in range(4) for i in range(2)]
        broken = copy.deepcopy(rows)
        broken[-1].metadata["info"]["id"] = "other-logical-frame"
        with self.assertRaisesRegex(ValueError, "two distinct captures per phase"):
            BurstIndex(broken, 8, min_frames=4, presence=True).validate_training()
        broken = [Exposure(i, phase=phase, source="synthetic") for phase in range(4) for i in range(2)]
        for row in broken:
            row.metadata["synthetic"]["phaseGroupId"] += str(row.metadata["info"]["phase"])
        with self.assertRaisesRegex(ValueError, "two distinct captures per phase"):
            BurstIndex(broken, 8, min_frames=4, presence=True).validate_training()
        broken[0].metadata["synthetic"].pop("phaseGroupId")
        with self.assertRaisesRegex(ValueError, "phaseGroupId"):
            BurstIndex(broken, 8, min_frames=4, presence=True)



@unittest.skipUnless(HAS_TORCH, "PyTorch needed for training integration")
class VariableTrainingTests(unittest.TestCase):
    def test_real_augmentation_precedes_padding_and_never_changes_presence(self):
        rows = [Exposure(i, phase=phase) for phase in range(4) for i in range(2)]
        sample = BurstIndex(rows, 8, min_frames=4, presence=True).train_sample(
            rows[0], random.Random(7), actual_frames=5)
        original, _, _ = sample.tensors()
        pixels, labels, mask = training_tensors([sample], RealAugmentation(seed=8))
        self.assertEqual(pixels.shape, (1, 32, 256, 256))
        np.testing.assert_array_equal(pixels[0, 24:], original[24:])
        for slot in range(8):
            if slot not in sample.slots:
                self.assertFalse(np.any(pixels[0, 3*slot:3*slot+3]))
            else:
                self.assertFalse(np.array_equal(pixels[0, 3*slot:3*slot+3],
                                                original[3*slot:3*slot+3]))
        self.assertEqual(labels.shape, (1, 64, 64)); self.assertTrue(mask.all())

    def test_paired_counts_survive_shuffling_accumulation_and_duplicate_anchors(self):
        rows = ([Exposure(i, phase=phase) for phase in range(4) for i in range(2)]
                + [Exposure(i, phase=phase, source="synthetic", burst="synthetic") for phase in range(4) for i in range(2)])
        sampler = BalancedSampler(rows, {"receiver": "phone-to-pc"}, paired_targets=True, seed=19)
        index, rng = BurstIndex(rows, 8, min_frames=4, presence=True), random.Random(28)
        observed = set()
        for _ in range(8):
            # 20 pairs ensure anchor IDs repeat; individual pair IDs must survive.
            batches = sampler.draw_update(4, 10)
            sampled = draw_training_bursts(batches, sampler, index, rng)
            pairs = {}
            for batch, identifiers in zip(sampled, sampler.last_pair_ids):
                for sample, identifier in zip(batch, identifiers):
                    pairs.setdefault(identifier, []).append(sample)
                    observed.add(len(sample.members))
            self.assertEqual(len(pairs), 20)
            for pair in pairs.values():
                self.assertEqual(len(pair), 2)
                self.assertEqual({s.metadata["source"] for s in pair}, {"real", "synthetic"})
                self.assertEqual(len(pair[0].members), len(pair[1].members))
        self.assertEqual(observed, {4, 5, 6, 7, 8})

    def test_phase_permutations_and_flips_preserve_supervision_and_presence(self):
        labels = np.arange(64*64, dtype=np.int64).reshape(64, 64) % 16
        mask = np.ones_like(labels, dtype=bool)
        mask[::9] = False
        mask[:, ::9] = False
        pixels = np.zeros((32, 256, 256), dtype=np.float32)
        masks = (0, 15, 7, 8)
        for phase in range(4):
            physical = labels.copy()
            physical[mask] ^= masks[phase]
            image = np.repeat(np.repeat(physical, 4, axis=0), 4, axis=1)
            pixels[6*phase:6*phase+3] = image
            pixels[24+2*phase] = 1
        original = pixels.copy()
        for delta in masks:
            transformed, target, result_mask = PhaseAugmentation.transform(pixels, labels, mask, delta=delta)
            np.testing.assert_array_equal(target[~mask], labels[~mask])
            np.testing.assert_array_equal(target[mask], labels[mask] ^ delta)
            for phase in range(4):
                expected = target.copy()
                expected[mask] ^= masks[phase]
                np.testing.assert_array_equal(transformed[6*phase, ::4, ::4], expected)
                self.assertTrue(np.all(transformed[24+2*phase] == 1))
                self.assertFalse(np.any(transformed[6*phase+3:6*phase+6]))
            for horizontal, vertical in ((True, False), (False, True), (True, True)):
                first = PhaseAugmentation.transform(pixels, labels, mask, delta=delta,
                                                    horizontal=horizontal, vertical=vertical)
                restored = PhaseAugmentation.transform(*first, delta=delta,
                                                       horizontal=horizontal, vertical=vertical)
                for got, expected in zip(restored, (pixels, labels, mask)):
                    np.testing.assert_array_equal(got, expected)
                    self.assertTrue(got.flags.c_contiguous)
        np.testing.assert_array_equal(pixels, original)

    def test_actual_attention_training_reports_each_count_and_saves_strict_checkpoint(self):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="variable-training-", dir=base) as temporary:
            root, mapping = Path(temporary), {}
            for split, seed in (("train", 71811), ("validation", 71821)):
                directory = root/split
                process = subprocess.run(["node", str(ROOT/"scripts/corrector-dataset.mjs"), "generate",
                    "--out", str(directory), "--count", "1", "--size", "64", "--colors", "16",
                    "--phases", "4", "--seed", str(seed), "--split", split], capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                path = directory/"manifest.json"
                manifest = json.loads(path.read_text())
                session = f"variable-{split}-receiver"
                mapping[session] = "phone-to-pc"
                rows = []
                for source in manifest["samples"]:
                    source.update(source="real", sessionId=session, senderSessionId=f"sender-{split}")
                    phase = source["info"]["phase"]
                    for kind in (("real", "synthetic") if split == "train" else ("real",)):
                        for frame in range(2):
                            row = copy.deepcopy(source)
                            row.update(id=f"{kind}-{split}-{phase}-{frame}", source=kind,
                                       burstId=f"{kind}-{split}-{phase}-burst", timestamp=frame*100)
                            if kind == "synthetic":
                                row["synthetic"] = {"phaseGroupId": f"{split}-trajectory"}
                            rows.append(row)
                manifest["samples"] = rows; path.write_text(json.dumps(manifest))
            train, validation = Dataset([root/"train"], "train"), Dataset([root/"validation"], "validation")
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                result = run_training(train, validation, root/"run", model_profile="U-tube-2x-attn",
                    steps=1, batch_size=1, accumulate=2, paired_targets=True, session_groups=mapping,
                    eval_every=1, device="cpu", no_progress=True)
            self.assertEqual(result["steps"], 1)
            self.assertIn("K4:", console.getvalue()); self.assertIn("phone-to-pc", console.getvalue())
            self.assertNotIn("Goal phone→PC", console.getvalue())
            self.assertNotIn("target 50%", console.getvalue())
            checkpoint_model, checkpoint = load_model_checkpoint(result["checkpoint"])
            self.assertEqual(checkpoint_model.profile, "U-tube-2x-attn")
            self.assertEqual(checkpoint["training"]["frameCounts"], [4, 5, 6, 7, 8])
            self.assertEqual(checkpoint["training"]["selectionAggregation"], "frame-count-group-balanced-real")
            last = result["history"][-1]
            self.assertEqual(set(last["validationByFrameCount"]), {"4", "5", "6", "7", "8"})
            for count, report in last["validationByFrameCount"].items():
                self.assertEqual(report["samples"], 1)
                self.assertEqual(report["recovery"]["framesPerInput"], int(count))
                self.assertEqual(report["recovery"]["inputCaptures"], int(count))
            self.assertEqual(sum(last["trainFrameCounts"].values()), 2)
            self.assertEqual(last["trainSources"]["real"]["inputCaptures"],
                             last["trainSources"]["synthetic"]["inputCaptures"])
            self.assertEqual(last["phoneToPCGoal"]["targetExactRate"], .5)


if __name__ == "__main__":
    unittest.main()
