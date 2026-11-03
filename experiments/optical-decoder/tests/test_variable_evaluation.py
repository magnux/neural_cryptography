"""Nested-count decoder wiring with canonical truth fixtures, not optical accuracy."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from data import Dataset
    from evaluate import evaluate_model
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for variable-frame evaluation")
class VariableEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=base)
        cls.root = Path(cls.temporary.name)
        data = cls.root/"data"
        subprocess.run(["node", str(ROOT/"scripts/decoder-dataset.mjs"), "generate",
                        "--out", str(data), "--count", "1", "--size", "64",
                        "--colors", "16", "--phases", "4", "--split", "validation", "--seed", "12046"],
                       check=True, capture_output=True, text=True)
        path = data/"manifest.json"
        manifest = json.loads(path.read_text())
        originals = manifest["samples"]
        # Distinct fixture IDs exercise observation accounting. These copies
        # are not independent camera observations or an accuracy benchmark.
        manifest["samples"] = [{**copy.deepcopy(original), "id": f"phase-{phase}-exposure-{index}",
            "source": "real", "sessionId": "fixture-session", "burstId": f"fixture-phase-{phase}",
            "timestamp": index*100} for phase, original in enumerate(originals) for index in range(5)]
        path.write_text(json.dumps(manifest))
        cls.dataset = Dataset([data], "validation")
        labels = cls.dataset.samples[0].tensors()[1]
        logits = np.full((1, 16, 64, 64), -20., np.float32)
        y, x = np.indices(labels.shape)
        logits[0, labels, y, x] = 20

        class FixtureModel(torch.nn.Module):
            frames = 8
            min_frames = 4
            variable_frames = True
            input_channels = 32
            supported_sizes = (64,)

            def __init__(self):
                super().__init__()
                self.register_buffer("fixture_logits", torch.from_numpy(logits))
                self.observed_counts = []

            def forward(self, pixels):
                if tuple(pixels.shape) != (1, 32, 256, 256):
                    raise ValueError("Expected eight RGB slots and eight presence planes")
                presence = pixels[:, 24:, 0, 0]
                if not bool(torch.all(pixels[:, 24:] == presence[:, :, None, None])):
                    raise ValueError("Presence must be spatially constant")
                if not bool(torch.all((presence == 0) | (presence == 1))):
                    raise ValueError("Presence must be binary")
                missing = presence[0] == 0
                if not bool(torch.all(pixels[:, :24].reshape(1, 8, 3, 256, 256)[:, missing] == 0)):
                    raise ValueError("Missing observations must be padded with zero")
                count = int(presence.sum())
                if not bool(torch.all(presence.reshape(1, 4, 2).sum(dim=2) >= 1)):
                    raise ValueError("Every phase needs at least one capture")
                self.observed_counts.append(count)
                # Deliberate failure for K=4 tests separate metrics/goal status;
                # it does not model a learned decoder's behavior.
                return torch.zeros_like(self.fixture_logits) if count == 4 else self.fixture_logits

        cls.model_class = FixtureModel

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_nested_presets_use_actual_captures_and_equal_classical_budgets(self):
        model = self.model_class().train()
        result = evaluate_model(model, self.dataset, work_dir=self.root/"exact",
                                session_groups={"fixture-session": "phone-to-pc"})
        self.assertTrue(model.training)
        self.assertEqual(model.observed_counts, [4, 4, 5, 5, 6, 6, 7, 7, 8, 8])
        self.assertEqual(result["uniqueCandidates"], 2)
        self.assertEqual(result["samples"], 10)
        self.assertEqual(result["observationCases"], 10)
        self.assertEqual(result["inputCaptures"], 60)
        self.assertEqual(result["uniqueInputCaptures"], 16)
        self.assertEqual(result["availableInputCaptures"], 20)
        self.assertEqual(result["framesPerInput"], 8)
        self.assertEqual(result["actualFrameCounts"], [4, 5, 6, 7, 8])
        previous = None
        for count in range(4, 9):
            row = result["byFrameCount"][str(count)]
            self.assertEqual(row["samples"], 2)
            self.assertEqual(row["frameCapacity"], 8)
            self.assertEqual(row["framesPerInput"], count)
            self.assertEqual(row["burstGrouping"]["remainderCaptures"], 4)
            self.assertEqual(row["burstGrouping"]["unusedCapacityCaptures"], 2*(8-count))
            self.assertEqual(row["recovery"]["framesPerInput"], count)
            self.assertEqual(row["recovery"]["inputCaptures"], 2*count)
            members = {sample["id"]: sample["memberIds"] for sample in row["recovery"]["samples"]}
            self.assertTrue(all(len(ids) == count and len(set(ids)) == count for ids in members.values()))
            self.assertEqual(len(set(sum(members.values(), []))), 2*count)
            if previous:
                self.assertEqual(set(previous), set(members))
                for identity, ids in previous.items():
                    self.assertEqual(ids, members[identity][:-1])
            previous = members
            summary = row["recovery"]["summary"]
            self.assertEqual(summary["classical"]["exactFrames"], 2)
            self.assertEqual(summary["learned"]["exactFrames"], 0 if count == 4 else 2)
            self.assertEqual(summary["learned"]["wrongAcceptedFrames"], 0)
        self.assertEqual(result["selectionScore"]["method"], "frame-count-group-balanced-real")
        self.assertEqual(result["selectionScore"]["exactRate"], .8)
        self.assertAlmostEqual(result["selectionScore"]["loss"],
            sum(row["selectionScore"]["loss"] for row in result["byFrameCount"].values())/5)
        goal = result["phoneToPCGoal"]
        self.assertEqual(goal["targetExactRate"], .5)
        self.assertTrue(goal["available"])
        self.assertFalse(goal["achievedForAllFrameCounts"])
        self.assertFalse(goal["byFrameCount"]["4"]["achieved"])
        self.assertTrue(goal["byFrameCount"]["8"]["achieved"])

    def test_metrics_only_and_capping_keep_unknown_recovery_and_same_candidates(self):
        report = evaluate_model(self.model_class(), self.dataset, exact=False, max_samples=1,
                                work_dir=self.root/"metrics")
        self.assertEqual(report["uniqueCandidates"], 1)
        self.assertEqual(report["observationCases"], 5)
        self.assertEqual(report["uniqueInputCaptures"], 8)
        self.assertIsNone(report["recovery"])
        self.assertIsNone(report["selectionScore"]["exactRate"])
        self.assertEqual(report["selectionScore"]["method"], "frame-count-balanced")
        self.assertFalse(report["phoneToPCGoal"]["available"])
        self.assertIsNone(report["phoneToPCGoal"]["byFrameCount"]["8"]["exactRate"])
        for row in report["byFrameCount"].values():
            self.assertEqual(row["samples"], 1)
            self.assertEqual(row["availableSamples"], 2)
            self.assertIsNone(row["recovery"])
        # Input-list ordering must not change nested observations or slot layout.
        reversed_data = SimpleNamespace(samples=list(reversed(self.dataset.samples)),
                                        manifest_hash=self.dataset.manifest_hash)
        repeated = evaluate_model(self.model_class(), reversed_data, exact=False, max_samples=1,
                                  work_dir=self.root/"metrics-repeat")
        self.assertEqual(report["confusion"], repeated["confusion"])
        self.assertEqual(report["selectionScore"], repeated["selectionScore"])

    def test_incomplete_capacity_groups_and_unsupported_grid_fail_before_inference(self):
        short = SimpleNamespace(samples=self.dataset.samples[:7], manifest_hash=self.dataset.manifest_hash)
        model = self.model_class()
        with self.assertRaisesRegex(ValueError, "No complete 8-capture"):
            evaluate_model(model, short, exact=False, work_dir=self.root/"short")
        self.assertEqual(model.observed_counts, [])
        unsupported = SimpleNamespace(samples=[SimpleNamespace(shape=(128, 4))], manifest_hash="0"*64)
        with self.assertRaisesRegex(ValueError, "only matrix sizes"):
            evaluate_model(model, unsupported, exact=False, work_dir=self.root/"unsupported")


if __name__ == "__main__":
    unittest.main()
