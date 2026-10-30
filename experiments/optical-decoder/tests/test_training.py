"""M2 tests: split isolation, correlated data, masked learning, and exact recovery."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))

from data import Dataset, assert_disjoint, check_checkpoint_overlap

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from checkpoint import load_checkpoint
    from evaluate import canonical_recovery, masked_loss
    from model import OpticalDecoder
    from synthetic import CORRUPTION_KIND, PhotometricConfig, corrupt_burst, generate
    from train import run_training, training_defaults, prepare_batch_groups, choose_batch
    torch.set_num_threads(1)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="m2-tests-", dir=base)
        cls.directory = Path(cls.temporary.name)
        for split, seed in (("train", 921), ("validation", 922), ("test", 923)):
            result = subprocess.run(["node", str(ROOT/"scripts/decoder-dataset.mjs"), "generate",
                "--out", str(cls.directory/split), "--count", "1", "--size", "64", "--colors", "4",
                "--phases", "1", "--split", split, "--seed", str(seed)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_canonical_files_and_group_splits(self):
        train = Dataset([self.directory/"train"], "train")
        validation = Dataset([self.directory/"validation"], "validation")
        test = Dataset([self.directory/"test"], "test")
        assert_disjoint(train, validation, test)
        sample = train.samples[0]
        pixels, labels, mask = sample.tensors()
        self.assertEqual(pixels.shape, (3, 256, 256))
        self.assertEqual(labels.shape, (64, 64))
        self.assertTrue(np.any(mask))
        self.assertFalse(np.all(mask))
        self.assertTrue(np.all(labels < 4))
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint(train, train)

    def test_joined_sender_sessions_and_payloads_cannot_cross_splits(self):
        train = Dataset([self.directory/"train"], "train")
        validation = Dataset([self.directory/"validation"], "validation")
        train.samples[0].metadata["senderSessionId"] = "one-physical-session"
        validation.samples[0].metadata["senderSessionId"] = "one-physical-session"
        with self.assertRaisesRegex(ValueError, "session overlap"):
            assert_disjoint(train, validation)
        checkpoint = {"training": {"trainSessionIds": ["one-physical-session"],
            "trainPayloadIds": [], "validationSessionIds": [], "validationPayloadIds": []}}
        with self.assertRaisesRegex(ValueError, "sessions overlap"):
            check_checkpoint_overlap(checkpoint, validation, "test")

    def test_hash_and_escape_checks(self):
        sample = Dataset([self.directory/"train"]).samples[0]
        sample.metadata["hashes"]["input"] = "0"*64
        with self.assertRaisesRegex(ValueError, "hash"):
            sample.tensors()
        sample.metadata["input"] = "../escape.f32"
        with self.assertRaisesRegex(ValueError, "escapes"):
            sample.tensors()

    def test_large_offline_manifest_and_generation_count(self):
        original = json.loads((self.directory/"train"/"manifest.json").read_text())
        sample = original["samples"][0]
        # Metadata only: exercise the old count/byte ceilings without writing
        # 50,001 image tensors. Reusing binary paths is valid for this loader.
        entries = [{**sample, "id": f"scale-{i}"} for i in range(50_001)]
        path = self.directory/"train"/"large-manifest.json"
        try:
            path.write_text(json.dumps({"format": original["format"], "version": 1, "samples": entries}))
            self.assertGreater(path.stat().st_size, 16*1024*1024)
            loaded = Dataset([path], "train")
            self.assertEqual(len(loaded.samples), 50_001)
            for invalid in ([], {}, None):
                path.write_text(json.dumps({"format": original["format"], "version": 1, "samples": invalid}))
                with self.assertRaisesRegex(ValueError, "nonempty list"):
                    Dataset([path])
        finally:
            path.unlink(missing_ok=True)
        module = json.dumps((ROOT/"scripts/decoder-dataset.mjs").as_uri())
        script = f"""import assert from 'node:assert/strict';
import {{generationCount}} from {module};
for (const count of [2048, 2**20]) assert.equal(generationCount(String(count)), count);
for (const count of [0, -1, 1.5, 2**20+1, 'x']) assert.throws(()=>generationCount(count));
"""
        subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True)

    @unittest.skipUnless(HAS_TORCH, "PyTorch needed for training defaults and batch selection")
    def test_training_defaults_and_prepared_source_pools(self):
        import random
        self.assertEqual(training_defaults(), (20_000, 1e-4))
        self.assertEqual(training_defaults("checkpoint.pt"), (32_000, 5e-5))
        self.assertEqual(training_defaults("checkpoint.pt", 320_000, 1e-5), (320_000, 1e-5))
        class Example:
            def __init__(self, source, marker):
                self.metadata = {"source": source}
                self.marker = marker
            def tensors(self):
                return np.full((3, 4, 4), self.marker, np.float32), np.zeros((1, 1), np.int64), np.ones((1, 1), bool)
        real, synthetic = Example("real", 1), Example("synthetic", 0)
        prepared = prepare_batch_groups({(64, 4): [real, synthetic]})
        self.assertEqual(prepared[(64, 4)]["real"], [real])
        self.assertEqual(prepared[(64, 4)]["synthetic"], [synthetic])
        # choose_batch no longer walks sample metadata, even in a large corpus.
        real.metadata = synthetic.metadata = None
        self.assertTrue(torch.all(choose_batch(prepared, random.Random(1), 4, 1)[0] == 1))
        self.assertTrue(torch.all(choose_batch(prepared, random.Random(1), 4, 0)[0] == 0))

    @unittest.skipUnless(HAS_TORCH, "PyTorch needed for corruptions and learning")
    def test_correlated_corruptions_and_materialized_provenance(self):
        sample = Dataset([self.directory/"train"]).samples[0]
        clean, labels, mask = sample.tensors()
        np.testing.assert_array_equal(corrupt_burst(clean, 7, 2, 0), np.stack([clean, clean]))
        burst = corrupt_burst(clean, 7, 3, .6)
        np.testing.assert_array_equal(burst, corrupt_burst(clean, 7, 3, .6))
        self.assertFalse(np.array_equal(burst[0], clean))
        self.assertFalse(np.array_equal(burst[0], burst[1]))
        residuals = (burst-clean[None]).reshape(3, -1)
        self.assertGreater(float(np.corrcoef(residuals[0], residuals[1])[0, 1]), .3)
        self.assertTrue(np.isfinite(burst).all())
        out = self.directory/"corrupted"
        result = generate([self.directory/"train"], out, frames=2, seed=55)
        self.assertEqual(result["generator"]["kind"], CORRUPTION_KIND)
        self.assertEqual(result["generator"]["photometric"]["exposure_stops"], 1.5)
        dataset = Dataset([out], "train")
        for item in dataset.samples:
            _, actual_labels, actual_mask = item.tensors()
            np.testing.assert_array_equal(actual_labels, labels)
            np.testing.assert_array_equal(actual_mask, mask)
            self.assertEqual(item.metadata["payloadId"], sample.metadata["payloadId"])
            self.assertEqual(item.metadata["sessionId"], sample.metadata["sessionId"])
            self.assertEqual(item.metadata["split"], sample.metadata["split"])
            self.assertEqual(item.metadata["synthetic"]["recipeSha256"], result["generator"]["recipeSha256"])
            for field in ("labels", "mask", "expected"):
                self.assertEqual(item.metadata["hashes"][field], sample.metadata["hashes"][field])
            self.assertEqual(item.metadata["baselineMode"], "canonical-from-corrupted-input")
            self.assertNotEqual(item.metadata["hashes"]["baseline"], sample.metadata["hashes"]["baseline"])
        # Distinct corruption settings can coexist for the same payload/seed.
        # Identical settings remain deterministic, including the canonical baseline.
        repeated = generate([self.directory/"train"], self.directory/"repeat-corrupted", frames=2, seed=55)
        self.assertEqual(result, repeated)
        ablated = self.directory/"ablated"
        disabled = generate([self.directory/"train"], ablated, frames=2, seed=55,
                            photometric=PhotometricConfig(0, 0, 0, 0))
        self.assertNotEqual(result["generator"]["recipeSha256"], disabled["generator"]["recipeSha256"])
        self.assertEqual(len(Dataset([out, ablated], "train").samples), 4)
        with self.assertRaisesRegex(ValueError, "canonical synthetic clean"):
            generate([out], self.directory/"double")

    @unittest.skipUnless(HAS_TORCH, "PyTorch needed for palette masking")
    def test_loss_excludes_reference_cells_and_inactive_classes(self):
        logits = torch.zeros((1, 16, 2, 2), requires_grad=True)
        labels = torch.zeros((1, 2, 2), dtype=torch.long)
        mask = torch.tensor([[[True, False], [False, False]]])
        loss = masked_loss(logits, labels, mask, 4)
        self.assertAlmostEqual(float(loss.detach()), np.log(4), places=6)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:, 4:] == 0))
        self.assertTrue(torch.all(logits.grad[:, :, 1:] == 0))
        self.assertTrue(torch.all(logits.grad[:, :, 0, 1] == 0))
        self.assertLess(float(logits.grad[0, 0, 0, 0]), 0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch needed for optimizer smoke")
    def test_training_checkpoint_and_canonical_oracle_boundary(self):
        train = Dataset([self.directory/"train"], "train")
        validation = Dataset([self.directory/"validation"], "validation")
        out = self.directory/"run"
        result = run_training(train, validation, out, steps=2, batch_size=1, accumulate=1,
                              eval_every=2, validation_samples=1, seed=19, device="cpu", model_profile="compact")
        model = OpticalDecoder()
        checkpoint = load_checkpoint(result["checkpoint"], model)
        self.assertEqual(checkpoint["training"]["selection"], "validation-exact")
        torch.manual_seed(19)
        initial = OpticalDecoder()
        self.assertFalse(torch.equal(initial.head.bias, model.head.bias))
        fine = run_training(train, validation, self.directory/"fine", initialize=result["checkpoint"],
                            steps=1, batch_size=1, accumulate=1, eval_every=1,
                            validation_samples=1, device="cpu")
        fine_checkpoint = load_checkpoint(fine["checkpoint"], OpticalDecoder("compact"))
        self.assertEqual(fine_checkpoint["training"]["learningRate"], 5e-5)
        with self.assertRaisesRegex(ValueError, "differs from.*checkpoint profile"):
            run_training(train, validation, self.directory/"wrong-profile", initialize=result["checkpoint"],
                         model_profile="wide", steps=1, batch_size=1, accumulate=1)
        # Ground truth logits are a diagnostic of the downstream boundary only.
        # They never enter the model or any training/validation inference above.
        sample = validation.samples[0]
        _, labels, _ = sample.tensors()
        logits = np.full((16, 64, 64), -12, dtype=np.float32)
        for color in range(4):
            logits[color][labels == color] = 12
        prediction = out/"oracle.f32"
        prediction.write_bytes(logits.astype("<f4").tobytes())
        decoded = canonical_recovery(validation, [sample], {sample.id: prediction}, out)
        self.assertEqual(decoded["summary"]["learned"]["exactFrames"], 1)
        self.assertEqual(decoded["summary"]["learned"]["wrongAcceptedFrames"], 0)
        check_checkpoint_overlap(checkpoint, Dataset([self.directory/"test"], "test"), "test")
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_checkpoint_overlap(checkpoint, validation, "test")


if __name__ == "__main__":
    unittest.main()
