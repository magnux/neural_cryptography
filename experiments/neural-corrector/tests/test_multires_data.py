"""Neural-QR tensor contracts, phase draws, augmentations and balanced updates."""

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from augmentation import PhaseAugmentation
from bursts import BurstIndex
from data import Dataset, MULTIRES_SIZES, NEURAL_QR_DATASET_FORMAT, assert_disjoint
from phases import PHASE_MASKS
from sampling import BalancedSampler, infer_session_groups


def write_dataset(path, size=8, *, frames=2, source="real", split="train", session="receiver",
                  sender="sender", logical_seed=7, direction=("mobile", "desktop")):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(logical_seed)
    logical = rng.integers(0, 16, (size, size), dtype=np.uint8)
    y, x = np.mgrid[:size, :size]
    mask = ((y % 9 != 0) & (x % 9 != 0) & (y != size-1) & (x != size-1)).astype(np.uint8)
    rows = []
    for phase in range(4):
        labels = logical.copy()
        labels[mask.astype(bool)] ^= PHASE_MASKS[phase]
        for frame in range(frames):
            identifier = f"{source}-{session}-{split}-g{size}-p{phase}-f{frame}"
            info = {"size": size, "colors": 16, "phases": 4, "phase": phase, "pixelsPerCell": 2,
                    "permute": True, "gridBlock": 8, "parity": 0, "holdMs": 4000, "id": "deadbeef"}
            row = {"id": identifier, "sessionId": session, "senderSessionId": sender,
                   "payloadId": hashlib.sha256(logical.tobytes()).hexdigest(), "source": source,
                   "split": split, "size": size, "colors": 16, "pixelsPerCell": 2,
                   "inputShape": [3, 2*size, 2*size], "info": info,
                   "burstId": f"{session}/{size}/{phase}", "timestamp": frame*100,
                   "provenance": {"senderEnvironment": {"deviceClass": direction[0]},
                                  "receiverEnvironment": {"deviceClass": direction[1]}}, "hashes": {}}
            if source == "synthetic":
                row["synthetic"] = {"phaseGroupId": f"{session}/{size}/trajectory", "frame": frame}
            for name, value in {
                "input": np.full((3, 2*size, 2*size), .1*frame, dtype="<f4"),
                "labels": labels, "mask": mask, "expected": logical,
                "baseline": np.zeros((16, size, size), dtype="<f4"),
            }.items():
                file = path/f"{identifier}.{name}"
                file.write_bytes(value.tobytes())
                row[name] = file.name
                row["hashes"][name] = hashlib.sha256(value.tobytes()).hexdigest()
            rows.append(row)
    manifest = {"format": NEURAL_QR_DATASET_FORMAT, "version": 1, "samples": rows}
    (path/"manifest.json").write_text(json.dumps(manifest))
    return manifest


class MultiresDataTests(unittest.TestCase):
    def test_all_grids_use_two_pixels_and_preserve_raw_overshoot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size in MULTIRES_SIZES:
                write_dataset(root/f"g{size}", size, frames=1, session=f"r{size}")
            dataset = Dataset([root], split="train")
            self.assertEqual(dataset.dataset_format, NEURAL_QR_DATASET_FORMAT)
            self.assertEqual(dataset.pixels_per_cell, 2)
            self.assertEqual(set(dataset.groups()), {(g, 16) for g in MULTIRES_SIZES})
            for sample in dataset.samples:
                pixels, labels, mask = sample.tensors()
                self.assertEqual(pixels.shape, (3, sample.shape[0]*2, sample.shape[0]*2))
                self.assertEqual(labels.shape, mask.shape)
            sample = dataset.samples[0]
            value = np.full((3, 2*sample.shape[0], 2*sample.shape[0]), 1.4, dtype="<f4")
            (sample.root/sample.metadata["input"]).write_bytes(value.tobytes())
            sample.metadata["hashes"]["input"] = hashlib.sha256(value.tobytes()).hexdigest()
            np.testing.assert_array_equal(sample.tensors()[0], value)

    def test_verified_reads_cache_only_metadata_and_invalidate_after_rewrite(self):
        import data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_dataset(root)
            sample = Dataset([root]).samples[0]
            data._VERIFIED_READS.clear()
            original_hash = hashlib.sha256
            with patch("data.hashlib.sha256", wraps=original_hash) as compute_hash:
                first = sample.read("input", "<f4")
                second = sample.read("input", "<f4")
                self.assertEqual(compute_hash.call_count, 1)
                np.testing.assert_array_equal(first, second)
                file = sample.root/sample.metadata["input"]
                changed = first.copy()
                changed[0] = .25
                file.write_bytes(changed.tobytes())
                with self.assertRaisesRegex(ValueError, "mismatched input hash"):
                    sample.read("input", "<f4")
                self.assertEqual(compute_hash.call_count, 2)
            self.assertTrue(all(value is True for value in data._VERIFIED_READS.values()))

    def test_strict_profile_truth_and_split_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = write_dataset(root/"train")
            path = root/"train/manifest.json"
            original = copy.deepcopy(value)
            for change in ({"pixelsPerCell": 4}, {"inputShape": [3, 32, 32]}, {"colors": 4}):
                value = copy.deepcopy(original)
                value["samples"][0].update(change)
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    Dataset([path])
            path.write_text(json.dumps(original))
            sample = Dataset([path]).samples[0]
            file = sample.root/sample.metadata["labels"]
            wrong = bytearray(file.read_bytes())
            wrong[9] ^= 1
            file.write_bytes(wrong)
            sample.metadata["hashes"]["labels"] = hashlib.sha256(wrong).hexdigest()
            with self.assertRaisesRegex(ValueError, "phase labels"):
                sample.tensors()
            write_dataset(root/"validation", split="validation", logical_seed=8)
            with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
                Dataset([root])
            train = Dataset([root/"train"])
            validation = Dataset([root/"validation"])
            with self.assertRaisesRegex(ValueError, "session overlap"):
                assert_disjoint(train, validation)

    def test_phase_bursts_have_distinct_random_members_and_nested_fixed_presets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_dataset(root, frames=8)
            dataset = Dataset([root])
            index = BurstIndex(dataset.samples, 8, min_frames=4, presence=True)
            index.validate_training()
            for count in range(4, 9):
                burst = index.train_sample(dataset.samples[0], random.Random(53), actual_frames=count)
                pixels, labels, mask = burst.tensors()
                self.assertEqual(pixels.shape, (32, 16, 16))
                self.assertEqual(len({sample.id for sample in burst.members}), count)
                self.assertEqual(set(member.metadata["info"]["phase"] for member in burst.members), set(range(4)))
                self.assertEqual(int(pixels[24:, 0, 0].sum()), count)
                expected = burst.read("expected", "u1", 64).reshape(8, 8)
                np.testing.assert_array_equal(labels, expected)
            presets = {count: index.eval_samples(count)[0] for count in range(4, 9)}
            self.assertEqual({len(rows) for rows in presets.values()}, {4})
            for count in range(4, 8):
                for first, second in zip(presets[count], presets[count+1]):
                    self.assertLess({x.id for x in first.members}, {x.id for x in second.members})
            other = BurstIndex(list(reversed(dataset.samples)), 8, min_frames=4, presence=True)
            self.assertEqual([[m.id for m in b.members] for b in presets[8]],
                             [[m.id for m in b.members] for b in other.eval_samples(8)[0]])
            chronological = {frozenset((i, i+1)) for i in range(0, 8, 2)}
            actual = {frozenset(m.metadata["timestamp"]//100 for m in b.members if m.metadata["info"]["phase"] == 0)
                      for b in presets[8]}
            self.assertNotEqual(actual, chronological)

    def test_phase_permutations_and_flips_apply_to_inputs_labels_and_masks(self):
        pixels = np.arange(32*16*16, dtype=np.float32).reshape(32, 16, 16)
        labels = (np.arange(64).reshape(8, 8) % 16).astype(np.int64)
        mask = labels % 3 != 0
        for delta in PHASE_MASKS:
            result, target, selected = PhaseAugmentation.transform(pixels, labels, mask, delta=delta,
                                                                   horizontal=True, vertical=True)
            expected = labels.copy()
            expected[mask] ^= delta
            np.testing.assert_array_equal(target, expected[::-1, ::-1])
            np.testing.assert_array_equal(selected, mask[::-1, ::-1])
            for phase, value in enumerate(PHASE_MASKS):
                source = PHASE_MASKS.index(value ^ delta)
                np.testing.assert_array_equal(result[6*phase:6*phase+6], pixels[6*source:6*source+6, ::-1, ::-1])
                np.testing.assert_array_equal(result[24+2*phase:26+2*phase], pixels[24+2*source:26+2*source, ::-1, ::-1])

    def test_paired_updates_balance_sizes_sources_and_recorded_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size in (8, 16):
                for source in ("real", "synthetic"):
                    write_dataset(root/f"{source}-{size}", size, source=source, session=f"s{size}",
                                  direction=("mobile", "desktop") if size == 8 else ("desktop", "mobile"))
            dataset = Dataset([root])
            mapping = infer_session_groups(dataset.samples)
            self.assertEqual(mapping, {"s8": "phone-to-pc", "s16": "pc-to-phone"})
            sampler = BalancedSampler(dataset.samples, mapping, paired_targets=True, require_all_groups=False)
            observed = set()
            for _ in range(24):
                batches = sampler.draw_update(2, 2)
                flat = [sample for batch in batches for sample in batch]
                self.assertEqual(len({sample.shape for sample in flat}), 1)
                self.assertEqual(sum(sample.metadata["source"] == "real" for sample in flat), 2)
                self.assertEqual(sampler.last_counts["pairedTargets"], 2)
                observed.add(flat[0].shape[0])
            self.assertEqual(observed, {8, 16})
            unknown = SimpleNamespace(metadata={"source": "real", "sessionId": "unknown",
                "devicePair": {"sender": "a", "receiver": "b"}})
            self.assertTrue(infer_session_groups([unknown])["unknown"].startswith("device-pair-"))


if __name__ == "__main__":
    unittest.main()
