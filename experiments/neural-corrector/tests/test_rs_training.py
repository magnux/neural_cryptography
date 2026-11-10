"""RS-weighted objective plumbing, augmentation and checkpoint continuation."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from checkpoint import load_model_checkpoint
from data import Dataset
from train import (format_training_metrics, run_training, training_loss,
                   training_loss_options)
from test_wtube_training import fixture_v2

torch.set_num_threads(1)


class RSTrainingTests(unittest.TestCase):
    def test_objective_options_reject_invalid_combinations(self):
        self.assertIsNone(training_loss_options("ce", 0, .25))
        policy = training_loss_options("rs-weighted-ce", 32, .25)
        self.assertEqual(policy["weightedFraction"], .25)
        self.assertEqual(policy["errorCountUpperBounds"], [12, 24])
        for mode, parity, mix in (("other", 32, .25), ("rs-weighted-ce", 0, .25),
                                 ("rs-weighted-ce", 31, .25), ("ce", 32, -1),
                                 ("ce", 32, float("nan")), ("ce", 32, True)):
            with self.assertRaises(ValueError):
                training_loss_options(mode, parity, mix)

    def test_dispatch_preserves_plain_ce_and_rejects_unknown_loss(self):
        logits = torch.randn(2, 16, 16, 16, requires_grad=True)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        plain, ce = training_loss(logits, labels, mask, 16)
        disabled, reported = training_loss(logits, labels, mask, 16,
            loss="rs-weighted-ce", rs_loss_mix=0)
        torch.testing.assert_close(plain, disabled, rtol=0, atol=0)
        torch.testing.assert_close(ce, reported, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            training_loss(logits, labels, mask, 16, loss="unknown")

    def test_console_distinguishes_objective_from_unweighted_ce(self):
        interval = {"updates": 2, "loss": .4, "objective": .7,
                    "wrongCells": 2, "dataCells": 20, "sources": {}, "groups": {}}
        text = format_training_metrics(interval)
        self.assertIn("CE 0.2000", text)
        self.assertIn("objective 0.3500", text)
        self.assertIn("CER 10.00%", text)

    def test_cli_can_continue_ce_checkpoint_with_weighting_and_augmented_bursts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_v2(root/"train", "train", 32, paired=True, seed=301)
            fixture_v2(root/"validation", "validation", 32, seed=302)
            train, validation = Dataset([root/"train"], "train"), Dataset([root/"validation"], "validation")
            common = dict(model_profile="W-tube", frames=8, paired_targets=True,
                          steps=1, batch_size=2, accumulate=2, no_progress=True,
                          origin_ratios="native=1,resampled=0,header-g64=0")
            with contextlib.redirect_stdout(io.StringIO()):
                baseline = run_training(train, validation, root/"plain", **common)
            before = Path(baseline["checkpoint"]).read_bytes()
            out = root/"weighted"
            command = [sys.executable, str(EXPERIMENT/"train.py"),
                "--train", str(root/"train"), "--validation", str(root/"validation"),
                "--model", "W-tube", "--frames", "8", "--paired-targets",
                "--origin-ratios", "native=1,resampled=0,header-g64=0",
                "--initialize", baseline["checkpoint"], "--out", str(out),
                "--loss", "rs-weighted-ce", "--rs-loss-mix", ".25", "--rs-parity", "32",
                "--phase-flip-probability", "1", "--steps", "1", "--batch-size", "2",
                "--accumulate", "2", "--device", "cpu", "--no-progress"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stderr[-3000:])
            self.assertIn("| objective ", process.stdout)
            self.assertEqual(Path(baseline["checkpoint"]).read_bytes(), before)
            settings = json.loads((out/"settings.json").read_text())
            self.assertEqual(settings["loss"], "rs-weighted-cross-entropy")
            self.assertEqual(settings["rsLoss"]["parityBytesPerBlock"], 32)
            self.assertEqual(settings["rsLoss"]["weightedFraction"], .25)
            self.assertEqual(settings["initialLearningRate"], 5e-5)
            self.assertEqual(settings["phaseAugmentation"]["horizontalFlipProbability"], 1.)
            history = json.loads((out/"history.json").read_text())
            summary = history["history"][-1]["training"]
            self.assertTrue(np.isfinite(summary["meanObjective"]))
            self.assertIsNone(history["history"][0]["training"]["meanObjective"])
            rows = [dict(zip(history["tables"]["training"], row)) for row in summary["rows"]]
            mean_ce = sum(row["lossSum"] for row in rows)/sum(row["samples"] for row in rows)
            self.assertIn(f"CE {mean_ce:.4f}", process.stdout)
            model, checkpoint = load_model_checkpoint(out/"best.pt")
            self.assertEqual(model.profile, "W-tube")
            self.assertEqual(checkpoint["training"]["settings"]["rsLoss"], settings["rsLoss"])
            self.assertTrue(settings["selectionMetric"].endswith("exact-logical-data-matrix"))

    def test_training_passes_flip_metadata_through_to_weighted_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_v2(root/"train", "train", 16, paired=True, seed=401)
            fixture_v2(root/"validation", "validation", 16, seed=402)
            train, validation = Dataset([root/"train"], "train"), Dataset([root/"validation"], "validation")
            with patch("train.training_loss", wraps=training_loss) as observed, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_training(train, validation, root/"run", model_profile="W-tube", frames=8,
                    paired_targets=True, steps=1, batch_size=2, accumulate=2, no_progress=True,
                    origin_ratios="native=1,resampled=0,header-g64=0", loss="rs-weighted-ce",
                    phase_flip_probability=1.)
            self.assertEqual(observed.call_count, 2)
            for call in observed.call_args_list:
                flags = call.kwargs["spatial_flips"]
                np.testing.assert_array_equal(flags, np.ones((2, 2), dtype=bool))
                self.assertEqual(call.kwargs["loss"], "rs-weighted-ce")


if __name__ == "__main__":
    unittest.main()
