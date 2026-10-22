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
    from synthetic import corrupt_burst, generate
    from train import run_training
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
        generate([self.directory/"train"], out, frames=2, seed=55)
        dataset = Dataset([out], "train")
        for item in dataset.samples:
            _, actual_labels, actual_mask = item.tensors()
            np.testing.assert_array_equal(actual_labels, labels)
            np.testing.assert_array_equal(actual_mask, mask)
            self.assertEqual(item.metadata["payloadId"], sample.metadata["payloadId"])
            self.assertEqual(item.metadata["sessionId"], sample.metadata["sessionId"])
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
                              eval_every=2, validation_samples=1, seed=19, device="cpu")
        model = OpticalDecoder()
        checkpoint = load_checkpoint(result["checkpoint"], model)
        self.assertEqual(checkpoint["training"]["selection"], "validation-exact")
        torch.manual_seed(19)
        initial = OpticalDecoder()
        self.assertFalse(torch.equal(initial.head.bias, model.head.bias))
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
