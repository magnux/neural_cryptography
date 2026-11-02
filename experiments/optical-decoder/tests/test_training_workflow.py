"""Training wiring on canonical fixtures, not evidence of real-device recovery."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
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
    from augmentation import RealAugmentation
    from bursts import BurstIndex
    from checkpoint import load_model_checkpoint
    from data import Dataset, digest
    from evaluate import evaluate_model
    from model import DEFAULT_MODEL_PROFILE, OpticalDecoder, parameter_count
    from train import run_training, save_checkpoint
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for training integration")
class TrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="training-workflow-", dir=base)
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
            if split == "train":
                # Paired synthetic fixtures carry exactly the same target identity
                # and supervision as their real counterpart. Only the source/ID
                # differ. Actual corruption/target rendering has separate tests.
                real = manifest["samples"][:2]
                manifest["samples"] = real + [{**copy.deepcopy(row),
                    "id": "paired-"+row["id"], "source": "synthetic"} for row in real]
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
        return {"out": self.root/name, "initialize": self.initial, "paired_targets": True,
                "session_groups": self.mapping, "steps": 2, "batch_size": 4, "accumulate": 2,
                "eval_every": 1, "device": "cpu", **overrides}

    def test_actual_optimizer_canonical_recovery_balanced_draws_and_validation_boundary(self):
        snapshots = {sample.id: tuple(row.copy() for row in sample.tensors())
                     for sample in self.train.samples+self.validation.samples}
        source_metadata = [copy.deepcopy(sample.metadata) for sample in self.train.samples+self.validation.samples]
        validation_calls, states, augmentations = [], [], []
        augment = RealAugmentation.__call__
        def observed_augmentation(transform, pixels, sources):
            original = pixels.copy()
            result = augment(transform, pixels, sources)
            np.testing.assert_array_equal(pixels, original)
            for index, source in enumerate(sources):
                if source == "synthetic":
                    np.testing.assert_array_equal(result[index], original[index])
                else:
                    self.assertFalse(np.array_equal(result[index], original[index]))
            augmentations.append(tuple(sources))
            return result
        def observed_evaluation(model, dataset, **kwargs):
            self.assertIs(dataset, self.validation)
            self.assertEqual(kwargs["max_samples"], 0)
            states.append(model.head.bias.detach().clone())
            for sample in dataset.samples:
                for actual, original in zip(sample.tensors(), snapshots[sample.id]):
                    np.testing.assert_array_equal(actual, original)
            result = evaluate_model(model, dataset, **kwargs)
            validation_calls.append(result)
            return result
        console = io.StringIO()
        with patch("train.RealAugmentation.__call__", autospec=True, side_effect=observed_augmentation), \
                patch("train.evaluate_model", side_effect=observed_evaluation), \
                patch.object(console, "isatty", return_value=True), contextlib.redirect_stdout(console):
            result = run_training(self.train, self.validation, **self.options("actual"))
        self.assertIn("Training", console.getvalue())
        self.assertIn("2/2", console.getvalue())
        self.assertIn("| lr", console.getvalue())
        self.assertIn("| val CER", console.getvalue())
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(validation_calls), 3)  # baseline plus both updates
        self.assertEqual(len(augmentations), 4)  # train microbatches only, never validation
        self.assertFalse(torch.equal(states[0], states[-1]), "optimizer must update real model weights")
        self.assertTrue((self.root/"actual/baseline-validation.json").is_file())
        for entry in result["history"][1:]:
            self.assertEqual(entry["trainSources"]["real"]["samples"], 4)
            self.assertEqual(entry["trainSources"]["synthetic"]["samples"], 4)
            self.assertEqual({name: row["samples"] for name, row in entry["trainGroups"].items()},
                             {"pc_to_phone": 2, "phone_to_pc": 2})
            self.assertEqual(entry["validationSelection"]["method"], "group-balanced-real")
            self.assertEqual(entry["sampling"]["pairedTargets"], 4)
        for report in validation_calls:
            groups = report["groups"]
            self.assertEqual(groups["pc_to_phone"]["samples"], 2)
            self.assertEqual(groups["phone_to_pc"]["samples"], 1)
            self.assertAlmostEqual(report["selectionScore"]["loss"],
                (groups["pc_to_phone"]["loss"]+groups["phone_to_pc"]["loss"])/2)
        _, checkpoint = load_model_checkpoint(result["checkpoint"])
        self.assertEqual(checkpoint["training"]["selectionAggregation"], "group-balanced-real")
        self.assertEqual(checkpoint["training"]["settings"]["weightDecay"], .01)
        self.assertEqual(checkpoint["training"]["settings"]["realAugmentation"]["noiseStdRGB01"], .008)
        self.assertIs(checkpoint["training"]["pairedTargets"], True)
        self.assertNotIn("recipe", checkpoint["training"]["settings"])
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
            "recovery": {"summary": {"learned": {"wrongAcceptedFrames": wrong, "exactRate": exact,
                                                  "frames": 10, "exactFrames": round(10*exact)},
                                     "classical": {"exactRate": 0.}}}}

    def test_degrading_updates_preserve_loadable_step_zero(self):
        records = [self.report(.5, 1.), self.report(.4, 1.1), self.report(.3, 1.2), self.report(.2, 1.3)]
        states = []
        def evaluate(model, *_args, **_kwargs):
            states.append(model.head.bias.detach().clone())
            return records[len(states)-1]
        with patch("train.evaluate_model", side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            result = run_training(self.train, self.validation, **self.options("degrade", steps=3,
                batch_size=2, accumulate=2))
        self.assertEqual((result["steps"], result["selectedStep"], result["stopReason"]),
                         (3, 0, "budget"))
        self.assertFalse(torch.equal(states[0], states[-1]))
        self.assertEqual([entry["selected"] for entry in result["history"]], [True, False, False, False])
        self.assertEqual([entry["learningRate"] for entry in result["history"]], [5e-5]*4)
        model, selected = load_model_checkpoint(result["checkpoint"])
        original, _ = load_model_checkpoint(self.initial)
        self.assertTrue(selected["training"]["baselineSelection"])
        self.assertEqual(selected["training"]["steps"], 0)
        self.assertEqual(selected["training"]["initialCheckpointSha256"], self.initial_hash)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original.state_dict()[name], rtol=0, atol=0)
        history = json.loads((self.root/"degrade/history.json").read_text())
        self.assertEqual(history["completedSteps"], 3)
        self.assertEqual(history["requestedSteps"], 3)
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

    def test_cli_forwards_pairing_and_group_options(self):
        sidecar = self.root/"cli-session-groups.json"
        sidecar.write_text(json.dumps({"format": "optical-decoder-session-groups",
                                      "version": 1, "sessions": self.mapping}))
        out = self.root/"cli-paired"
        process = subprocess.run([sys.executable, str(EXPERIMENT/"train.py"),
            "--train", str(self.root/"train"), "--validation", str(self.root/"validation"),
            "--initialize", str(self.initial), "--session-groups", str(sidecar),
            "--paired-targets", "--steps", "1", "--batch-size", "4", "--accumulate", "1",
            "--real-noise-std", ".003", "--real-tint-stops", ".02", "--real-exposure-stops", ".03",
            "--device", "cpu", "--validation-samples", "0", "--eval-every", "1",
            "--out", str(out)], capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr[-3000:])
        self.assertIn("Step 0/1", process.stdout)
        self.assertIn("Step 1/1 | train CE", process.stdout)
        self.assertIn("| val CE", process.stdout)
        self.assertIn("| CER", process.stdout)
        self.assertIn("| exact", process.stdout)
        self.assertIn("Finished 1/1 updates", process.stdout)
        self.assertNotIn('"trainSources"', process.stdout)
        self.assertNotIn("\r", process.stdout)  # Noninteractive output has no redraws.
        history = json.loads((out/"history.json").read_text())
        settings = json.loads((out/"settings.json").read_text())
        self.assertEqual((history["completedSteps"], history["requestedSteps"]), (1, 1))
        self.assertEqual(history["stopReason"], "budget")
        self.assertIs(settings["pairedTargets"], True)
        self.assertEqual(settings["validationSamples"], 0)
        self.assertEqual(settings["initialLearningRate"], 5e-5)
        self.assertEqual(settings["realAugmentation"]["noiseStdRGB01"], .003)
        self.assertEqual(settings["realAugmentation"]["tintStops"], .02)
        self.assertEqual(settings["realAugmentation"]["exposureStops"], .03)
        self.assertNotIn("hardWeight", settings)
        update = history["history"][-1]
        self.assertEqual(update["trainSources"]["real"]["samples"], 2)
        self.assertEqual(update["trainSources"]["synthetic"]["samples"], 2)
        self.assertEqual({name: row["samples"] for name, row in update["trainGroups"].items()},
                         {"pc_to_phone": 1, "phone_to_pc": 1})
        self.assertEqual(update["validationSelection"]["method"], "group-balanced-real")
        _, checkpoint = load_model_checkpoint(out/"best.pt")
        self.assertIs(checkpoint["training"]["pairedTargets"], True)
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

    def test_interrupt_retains_completed_validation_history(self):
        with patch("train.evaluate_model", return_value=self.report(.5, 1.)), \
                patch("train.BalancedSampler.draw_update", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_training(self.train, self.validation, **self.options("interrupted"))
        history = json.loads((self.root/"interrupted/history.json").read_text())
        self.assertEqual(history["stopReason"], "interrupted")
        self.assertEqual(history["completedSteps"], 0)
        self.assertEqual([row["step"] for row in history["history"]], [0])
        self.assertEqual(history["selectedStep"], 0)
        load_model_checkpoint(self.root/"interrupted/best.pt")

    def burst_directories(self, prefix, *, overshoot=False):
        directories = {}
        for split in ("train", "validation"):
            destination = self.root/f"{prefix}-{split}"
            shutil.copytree(self.root/split, destination)
            path = destination/"manifest.json"
            manifest = json.loads(path.read_text())
            if overshoot:
                # Calibrated recordings retain valid out-of-camera-range RGB.
                # The model boundary, not stored files, must clamp these values.
                for relative in {row["input"] for row in manifest["samples"]}:
                    target = destination/relative
                    pixels = np.fromfile(target, dtype="<f4")
                    pixels[:2] = [1.4, -1.4]
                    pixels.tofile(target)
                for row in manifest["samples"]:
                    row["hashes"]["input"] = digest(destination/row["input"])
            # Distinct recorded identities with canonical pixels exercise burst
            # stacking and the actual optimizer; these remain synthetic fixtures.
            exposures = []
            for row in manifest["samples"]:
                for frame in range(5):
                    item = copy.deepcopy(row)
                    item["id"] = f"{row['id']}-exposure-{frame}"
                    item["timestamp"] = frame*100
                    exposures.append(item)
            manifest["samples"] = exposures
            path.write_text(json.dumps(manifest))
            directories[split] = destination
        return directories

    def test_five_frame_cli_warmstart_runs_optimizer_and_preserves_recordings(self):
        directories = self.burst_directories("burst")
        sidecar = self.root/"burst-session-groups.json"
        sidecar.write_text(json.dumps({"format": "optical-decoder-session-groups",
                                      "version": 1, "sessions": self.mapping}))
        out = self.root/"cli-burst"
        process = subprocess.run([sys.executable, str(EXPERIMENT/"train.py"),
            "--train", str(directories["train"]), "--validation", str(directories["validation"]),
            "--initialize", str(self.initial), "--session-groups", str(sidecar),
            "--frames", "5", "--paired-targets", "--steps", "1", "--batch-size", "4",
            "--accumulate", "1", "--device", "cpu", "--eval-every", "1", "--no-progress", "--out", str(out)],
            capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr[-3000:])
        history = json.loads((out/"history.json").read_text())
        self.assertEqual(history["completedSteps"], 1)
        self.assertEqual(history["settings"]["framesPerInput"], 5)
        self.assertEqual(history["settings"]["evaluationGrouping"], "fixed-seeded-shuffle-within-burst")
        self.assertEqual(history["settings"]["evaluationSeed"], 7)
        update = history["history"][-1]
        self.assertEqual(update["trainSources"]["real"]["samples"], 2)
        self.assertEqual(update["trainSources"]["real"]["inputCaptures"], 10)
        self.assertEqual(update["trainSources"]["synthetic"]["inputCaptures"], 10)
        model, checkpoint = load_model_checkpoint(out/"best.pt")
        self.assertEqual(model.frames, 5)
        self.assertEqual(checkpoint["training"]["framesPerInput"], 5)
        self.assertEqual(checkpoint["training"]["initialCheckpointSha256"], self.initial_hash)
        self.assertEqual(digest(self.initial), self.initial_hash)

    def test_fresh_five_frame_default_trains_normalized_model_without_rewriting_crops(self):
        directories = self.burst_directories("normalized", overshoot=True)
        recorded = {path: digest(path) for directory in directories.values()
                    for path in directory.rglob("*") if path.is_file()}
        sidecar = self.root/"normalized-session-groups.json"
        sidecar.write_text(json.dumps({"format": "optical-decoder-session-groups",
                                      "version": 1, "sessions": self.mapping}))
        out = self.root/"cli-normalized"
        process = subprocess.run([sys.executable, str(EXPERIMENT/"train.py"),
            "--train", str(directories["train"]), "--validation", str(directories["validation"]),
            "--session-groups", str(sidecar), "--frames", "5", "--paired-targets",
            "--steps", "1", "--batch-size", "4", "--accumulate", "1",
            "--device", "cpu", "--eval-every", "1", "--no-progress", "--out", str(out)],
            capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr[-3000:])
        history = json.loads((out/"history.json").read_text())
        settings = history["settings"]
        self.assertEqual(history["completedSteps"], 1)
        self.assertEqual(settings["modelProfile"], "normalized-compact")
        self.assertEqual(settings["framesPerInput"], 5)
        self.assertEqual(settings["networkInputClip"], [-1., 1.])
        self.assertEqual(settings["initialLearningRate"], 1e-4)
        self.assertEqual(settings["evaluationGrouping"], "fixed-seeded-shuffle-within-burst")
        self.assertEqual(settings["evaluationSeed"], 7)
        model, checkpoint = load_model_checkpoint(out/"best.pt")
        self.assertEqual(model.profile, DEFAULT_MODEL_PROFILE)
        self.assertEqual(parameter_count(model), 125600)
        self.assertFalse(hasattr(model, "local_color"))
        self.assertEqual(checkpoint["training"]["networkInputClip"], [-1., 1.])
        self.assertEqual(checkpoint["training"]["modelProfile"], model.profile)
        self.assertNotIn("initialCheckpointSha256", checkpoint["training"])
        torch.manual_seed(7)
        initial = OpticalDecoder(DEFAULT_MODEL_PROFILE, frames=5)
        self.assertFalse(torch.equal(initial.head.weight, model.head.weight))

        dataset = Dataset([directories["validation"]], "validation")
        sample = BurstIndex(dataset.samples, 5).eval_samples()[0][0]
        pixels = sample.tensors()[0]
        self.assertGreater(float(pixels.max()), 1.)
        self.assertLess(float(pixels.min()), -1.)
        seen = []
        hook = model.inlet[0][0].register_forward_pre_hook(
            lambda _module, inputs: seen.append(inputs[0].detach().clone()))
        try:
            with torch.inference_mode():
                logits = model(torch.from_numpy(pixels[None]))
        finally:
            hook.remove()
        self.assertEqual(tuple(logits.shape), (1, 16, 64, 64))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(seen[0], torch.from_numpy(pixels[None]).clamp(-1., 1.), rtol=0, atol=0)
        self.assertEqual({path: digest(path) for path in recorded}, recorded)

        # Continue the actual normalized five-frame best checkpoint with the
        # new augmentation; validation must reproduce its unaugmented baseline.
        prior_hash = digest(out/"best.pt")
        prior_validation = json.loads((out/"validation.json").read_text())
        continued = self.root/"cli-normalized-continued"
        process = subprocess.run([sys.executable, str(EXPERIMENT/"train.py"),
            "--train", str(directories["train"]), "--validation", str(directories["validation"]),
            "--initialize", str(out/"best.pt"), "--session-groups", str(sidecar),
            "--paired-targets", "--steps", "1", "--batch-size", "4", "--accumulate", "1",
            "--device", "cpu", "--eval-every", "1", "--no-progress", "--out", str(continued)],
            capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr[-3000:])
        history = json.loads((continued/"history.json").read_text())
        self.assertEqual(history["completedSteps"], 1)
        self.assertEqual(history["settings"]["framesPerInput"], 5)
        self.assertEqual(history["settings"]["modelProfile"], DEFAULT_MODEL_PROFILE)
        self.assertEqual(history["settings"]["initialLearningRate"], 5e-5)
        self.assertTrue(history["history"][0]["selected"])
        self.assertGreater(history["history"][-1]["trainLoss"], 0.)
        baseline = json.loads((continued/"baseline-validation.json").read_text())
        self.assertEqual(baseline["loss"], prior_validation["loss"])
        self.assertEqual(baseline["cellErrorRate"], prior_validation["cellErrorRate"])
        self.assertEqual(baseline["recovery"]["summary"], prior_validation["recovery"]["summary"])
        _, checkpoint = load_model_checkpoint(continued/"best.pt")
        self.assertEqual(checkpoint["training"]["initialCheckpointSha256"], prior_hash)
        self.assertEqual(digest(out/"best.pt"), prior_hash)
        self.assertEqual({path: digest(path) for path in recorded}, recorded)


if __name__ == "__main__":
    unittest.main()
