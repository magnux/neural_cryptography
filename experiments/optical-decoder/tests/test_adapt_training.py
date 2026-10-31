"""Adaptation wiring on canonical fixtures, not evidence of real-device recovery."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from adaptation import OnlineRealAugment
    from checkpoint import load_model_checkpoint
    from data import Dataset, digest
    from evaluate import evaluate_model
    from model import OpticalDecoder
    from train import run_training, save_checkpoint
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for adaptation integration")
class AdaptTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="adapt-training-", dir=base)
        cls.root = Path(cls.temporary.name)
        cls.mapping = {}
        for split, seed in (("train", 191_301), ("validation", 191_401)):
            directory = cls.root/split
            process = subprocess.run(["node", str(ROOT/"scripts/decoder-dataset.mjs"), "generate",
                "--out", str(directory), "--count", "3", "--size", "64", "--colors", "4",
                "--phases", "1", "--seed", str(seed), "--split", split],
                capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stderr)
            path = directory/"manifest.json"
            manifest = json.loads(path.read_text())
            for index, row in enumerate(manifest["samples"]):
                # Relabel canonical rendered fixtures only to exercise the real
                # branch. These are not camera captures or accuracy benchmarks.
                if split == "validation" or index < 2:
                    row["source"] = "real"
                    row["sessionId"] = f"fixture-{split}-receiver-{index}"
                    group = ("phone_to_pc" if (index == 1 if split == "train" else index == 2)
                             else "pc_to_phone")
                    cls.mapping[row["sessionId"]] = group
            path.write_text(json.dumps(manifest))
        cls.train = Dataset([cls.root/"train"], "train")
        cls.validation = Dataset([cls.root/"validation"], "validation")
        cls.initial = cls.root/"fixture-pilot.pt"
        torch.manual_seed(311)
        save_checkpoint(cls.initial, OpticalDecoder("compact"), {
            "steps": 1, "seed": 311, "samples": 1, "selectionStep": 1,
            "selection": "validation-exact", "datasetFormat": "optical-decoder-dataset",
            "trainSessionIds": ["fixture-pilot-train"], "trainPayloadIds": ["1"*64],
            "validationSessionIds": ["fixture-pilot-validation"], "validationPayloadIds": ["2"*64]})
        cls.initial_hash = digest(cls.initial)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def options(self, name, **overrides):
        return {"out": self.root/name, "initialize": self.initial, "recipe": "adapt",
                "session_groups": self.mapping, "steps": 2, "batch_size": 4, "accumulate": 2,
                "eval_every": 1, "device": "cpu", **overrides}

    def test_actual_optimizer_canonical_recovery_balanced_draws_and_validation_boundary(self):
        snapshots = {sample.id: tuple(row.copy() for row in sample.tensors())
                     for sample in self.train.samples+self.validation.samples}
        source_metadata = [copy.deepcopy(sample.metadata) for sample in self.train.samples+self.validation.samples]
        augmentation, validation_calls, states = [], [], []
        validating = [False]
        testcase = self
        class TrackingAugment(OnlineRealAugment):
            def __call__(self, pixels, real_mask):
                testcase.assertFalse(validating[0], "validation must not invoke online augmentation")
                before = pixels.clone()
                output = super().__call__(pixels, real_mask)
                torch.testing.assert_close(pixels, before, rtol=0, atol=0)
                for index, real in enumerate(real_mask):
                    if not real:
                        torch.testing.assert_close(output[index], before[index], rtol=0, atol=0)
                augmentation.append((list(real_mask), not torch.equal(output, before)))
                return output
        def observed_evaluation(model, dataset, **kwargs):
            self.assertIs(dataset, self.validation)
            self.assertEqual(kwargs["max_samples"], 0)
            states.append(model.head.bias.detach().clone())
            for sample in dataset.samples:
                for actual, original in zip(sample.tensors(), snapshots[sample.id]):
                    np.testing.assert_array_equal(actual, original)
            validating[0] = True
            try:
                result = evaluate_model(model, dataset, **kwargs)
            finally:
                validating[0] = False
            validation_calls.append(result)
            return result
        with patch("train.OnlineRealAugment", TrackingAugment), \
                patch("train.evaluate_model", side_effect=observed_evaluation), \
                contextlib.redirect_stdout(io.StringIO()):
            result = run_training(self.train, self.validation, **self.options("actual"))
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(augmentation), 4)
        self.assertTrue(any(changed for _, changed in augmentation))
        self.assertEqual(sum(sum(mask) for mask, _ in augmentation), 8)
        self.assertEqual(len(validation_calls), 3)  # baseline plus both updates
        self.assertFalse(torch.equal(states[0], states[-1]), "optimizer must update real model weights")
        self.assertTrue((self.root/"actual/baseline-validation.json").is_file())
        for entry in result["history"][1:]:
            self.assertEqual(entry["trainSources"]["real"]["samples"], 4)
            self.assertEqual(entry["trainSources"]["synthetic"]["samples"], 4)
            self.assertEqual({name: row["samples"] for name, row in entry["trainGroups"].items()},
                             {"pc_to_phone": 2, "phone_to_pc": 2})
            self.assertEqual(entry["validationSelection"]["method"], "group-balanced-real")
        for report in validation_calls:
            groups = report["groups"]
            self.assertEqual(groups["pc_to_phone"]["samples"], 2)
            self.assertEqual(groups["phone_to_pc"]["samples"], 1)
            self.assertAlmostEqual(report["selectionScore"]["loss"],
                (groups["pc_to_phone"]["loss"]+groups["phone_to_pc"]["loss"])/2)
        _, checkpoint = load_model_checkpoint(result["checkpoint"])
        self.assertEqual(checkpoint["training"]["selectionAggregation"], "group-balanced-real")
        self.assertEqual(checkpoint["training"]["settings"]["weightDecay"], .05)
        self.assertEqual(checkpoint["training"]["settings"]["hardWeight"], .1)
        self.assertEqual(digest(self.initial), self.initial_hash)
        for sample, metadata in zip(self.train.samples+self.validation.samples, source_metadata):
            self.assertEqual(sample.metadata, metadata)
            for actual, original in zip(sample.tensors(), snapshots[sample.id]):
                np.testing.assert_array_equal(actual, original)

    @staticmethod
    def report(exact, loss, wrong=0):
        return {"loss": loss, "cellErrorRate": .4,
            "selectionScore": {"method": "group-balanced-real", "groups": ["pc_to_phone", "phone_to_pc"],
                               "exactRate": exact, "loss": loss, "cellErrorRate": .4},
            "groups": {"pc_to_phone": {"loss": loss, "learnedExactRate": exact},
                       "phone_to_pc": {"loss": loss, "learnedExactRate": exact}},
            "recovery": {"summary": {"learned": {"wrongAcceptedFrames": wrong, "exactRate": exact},
                                     "classical": {"exactRate": 0.}}}}

    def test_degrading_updates_reduce_lr_stop_and_preserve_loadable_step_zero(self):
        records = [self.report(.5, 1.), self.report(.4, 1.1), self.report(.3, 1.2), self.report(.2, 1.3)]
        states = []
        def evaluate(model, *_args, **_kwargs):
            states.append(model.head.bias.detach().clone())
            return records[len(states)-1]
        with patch("train.evaluate_model", side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            result = run_training(self.train, self.validation, **self.options("degrade", steps=8,
                lr_patience=1, stop_patience=3, min_lr=1e-6, batch_size=2, accumulate=2))
        self.assertEqual((result["steps"], result["selectedStep"], result["stopReason"]),
                         (3, 0, "early-stopping"))
        self.assertFalse(torch.equal(states[0], states[-1]))
        self.assertEqual([entry["selected"] for entry in result["history"]], [True, False, False, False])
        np.testing.assert_allclose([entry["nextLearningRate"] for entry in result["history"]],
                                   [1e-5, 5e-6, 2.5e-6, 1.25e-6], rtol=0, atol=1e-15)
        model, selected = load_model_checkpoint(result["checkpoint"])
        original, _ = load_model_checkpoint(self.initial)
        self.assertTrue(selected["training"]["baselineSelection"])
        self.assertEqual(selected["training"]["steps"], 0)
        self.assertEqual(selected["training"]["initialCheckpointSha256"], self.initial_hash)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original.state_dict()[name], rtol=0, atol=0)
        history = json.loads((self.root/"degrade/history.json").read_text())
        self.assertEqual(history["completedSteps"], 3)
        self.assertEqual(history["requestedSteps"], 8)
        self.assertEqual(json.loads((self.root/"degrade/validation.json").read_text()), records[0])
        self.assertEqual(json.loads((self.root/"degrade/final-validation.json").read_text()), records[-1])
        self.assertEqual(digest(self.initial), self.initial_hash)

    def test_missing_groups_reject_before_updates_or_validation(self):
        with patch("train.evaluate_model") as evaluate, \
                patch.object(torch.optim.AdamW, "step", autospec=True) as step:
            with self.assertRaisesRegex(ValueError, "Missing session groups"):
                run_training(self.train, self.validation, **self.options("missing", session_groups={}))
            evaluate.assert_not_called()
            step.assert_not_called()
        self.assertFalse((self.root/"missing").exists())

    def test_adapt_cli_forwards_recipe_and_group_options(self):
        sidecar = self.root/"cli-session-groups.json"
        sidecar.write_text(json.dumps({"format": "optical-decoder-session-groups",
                                      "version": 1, "sessions": self.mapping}))
        out = self.root/"cli-adapt"
        process = subprocess.run([sys.executable, str(EXPERIMENT/"train.py"),
            "--train", str(self.root/"train"), "--validation", str(self.root/"validation"),
            "--initialize", str(self.initial), "--session-groups", str(sidecar),
            "--recipe", "adapt", "--steps", "1", "--batch-size", "4", "--accumulate", "1",
            "--device", "cpu", "--validation-samples", "0", "--eval-every", "1",
            "--out", str(out)], capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr[-3000:])
        self.assertIn('"stopReason": "budget"', process.stdout)
        history = json.loads((out/"history.json").read_text())
        settings = json.loads((out/"settings.json").read_text())
        self.assertEqual((history["completedSteps"], history["requestedSteps"]), (1, 1))
        self.assertEqual(settings["recipe"], "adapt")
        self.assertEqual(settings["validationSamples"], 0)
        self.assertEqual(settings["initialLearningRate"], 1e-5)
        self.assertEqual(settings["hardWeight"], .1)
        update = history["history"][-1]
        self.assertEqual(update["trainSources"]["real"]["samples"], 2)
        self.assertEqual(update["trainSources"]["synthetic"]["samples"], 2)
        self.assertEqual({name: row["samples"] for name, row in update["trainGroups"].items()},
                         {"pc_to_phone": 1, "phone_to_pc": 1})
        self.assertEqual(update["validationSelection"]["method"], "group-balanced-real")
        _, checkpoint = load_model_checkpoint(out/"best.pt")
        self.assertEqual(checkpoint["training"]["recipe"], "adapt")
        self.assertEqual(digest(self.initial), self.initial_hash)

    def test_wrong_acceptance_at_baseline_aborts_before_sampling_or_updates(self):
        with patch("train.evaluate_model", return_value=self.report(.5, 1., wrong=1)), \
                patch("train.BalancedSampler.draw_update") as draw, \
                patch.object(torch.optim.AdamW, "step", autospec=True) as step:
            with self.assertRaisesRegex(RuntimeError, "incorrect payload bytes"):
                run_training(self.train, self.validation, **self.options("wrong-accepted"))
            draw.assert_not_called()
            step.assert_not_called()
        self.assertTrue((self.root/"wrong-accepted/validation-failed.json").is_file())
        self.assertFalse((self.root/"wrong-accepted/best.pt").exists())
        self.assertEqual(digest(self.initial), self.initial_hash)


if __name__ == "__main__":
    unittest.main()
