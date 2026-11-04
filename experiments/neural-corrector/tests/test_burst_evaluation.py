"""Equal-observation burst evaluation on canonical fixtures, not camera metrics."""

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
    from bursts import EVALUATION_POLICY, EVALUATION_SEED
    from data import Dataset, Sample
    from evaluate import evaluate_model
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for burst evaluation")
class BurstEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=base)
        cls.root = Path(cls.temporary.name)
        cls.dataset_root = cls.root/"data"
        subprocess.run(["node", str(ROOT/"scripts/corrector-dataset.mjs"), "generate",
                        "--out", str(cls.dataset_root), "--count", "1", "--size", "64",
                        "--colors", "4", "--phases", "1", "--split", "validation", "--seed", "12045"],
                       check=True, capture_output=True, text=True)
        path = cls.dataset_root/"manifest.json"
        manifest = json.loads(path.read_text())
        original = manifest["samples"][0]
        # Copies exercise burst accounting and canonical recovery. They are not
        # independent camera observations or an optical-accuracy benchmark.
        manifest["samples"] = [{**copy.deepcopy(original), "id": f"exposure-{index}",
            "source": "real", "sessionId": "fixture-session", "burstId": "fixture-burst",
            "timestamp": index*100} for index in range(11)]
        path.write_text(json.dumps(manifest))
        cls.dataset = Dataset([cls.dataset_root], "validation")
        labels = cls.dataset.samples[0].tensors()[1]
        logits = np.full((1, 16, 64, 64), -20., np.float32)
        y, x = np.indices(labels.shape)
        logits[0, labels, y, x] = 20

        class FixedFixtureModel(torch.nn.Module):
            frames = 5

            def __init__(self):
                super().__init__()
                self.register_buffer("fixture_logits", torch.from_numpy(logits))
                self.shapes = []

            def forward(self, pixels):
                self.shapes.append(tuple(pixels.shape))
                if pixels.shape[1] != 15:
                    raise ValueError("Expected five RGB inputs")
                return self.fixture_logits.expand(len(pixels), -1, -1, -1)

        cls.model_class = FixedFixtureModel

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_five_capture_budget_nonoverlap_and_remainder(self):
        model = self.model_class().train()
        result = evaluate_model(model, self.dataset, work_dir=self.root/"evaluation",
                                session_groups={"fixture-session": "phone-to-pc"}, detailed=True)
        self.assertTrue(model.training)
        self.assertEqual(model.shapes, [(1, 15, 256, 256)]*2)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["availableSamples"], 2)
        self.assertEqual(result["inputCaptures"], 10)
        self.assertEqual(result["availableInputCaptures"], 11)
        self.assertEqual(result["burstGrouping"]["discardedCaptures"], 1)
        self.assertEqual(result["burstGrouping"]["method"], EVALUATION_POLICY)
        self.assertEqual(result["burstGrouping"]["seed"], EVALUATION_SEED)
        self.assertEqual(result["selection"]["seed"], EVALUATION_SEED)
        self.assertTrue(result["selection"]["method"].startswith(EVALUATION_POLICY))
        self.assertEqual(result["framesPerInput"], 5)
        recovered = result["recovery"]
        self.assertEqual(recovered["inputCaptures"], 10)
        for method in ("classical", "learned"):
            self.assertEqual(recovered["summary"][method]["exactFrames"], 2)
            self.assertEqual(recovered["summary"][method]["wrongAcceptedFrames"], 0)
        members = [row["memberIds"] for row in recovered["samples"]]
        self.assertNotEqual({frozenset(ids) for ids in members},
                            {frozenset(f"exposure-{i}" for i in range(5)),
                             frozenset(f"exposure-{i}" for i in range(5, 10))})
        self.assertEqual(len(set(sum(members, []))), 10)
        repeated = evaluate_model(model, self.dataset, work_dir=self.root/"repeat", detailed=True)
        self.assertEqual([row["memberIds"] for row in repeated["recovery"]["samples"]], members)
        capped = evaluate_model(model, self.dataset, work_dir=self.root/"capped", max_samples=1)
        self.assertEqual(capped["samples"], 1)
        self.assertEqual(capped["inputCaptures"], 5)
        self.assertEqual(capped["availableSamples"], 2)
        self.assertEqual(capped["reportDetail"], "summary")
        self.assertNotIn("sessions", capped)
        self.assertNotIn("samples", capped["recovery"])

    def test_empty_bursts_or_missing_real_direction_fail_explicitly(self):
        short = SimpleNamespace(samples=self.dataset.samples[:3], manifest_hash=self.dataset.manifest_hash)
        with self.assertRaisesRegex(ValueError, "No complete 5-frame"):
            evaluate_model(self.model_class(), short, exact=False, work_dir=self.root/"short")
        rows = []
        for index, sample in enumerate(self.dataset.samples[:9]):
            metadata = copy.deepcopy(sample.metadata)
            metadata["sessionId"] = "complete" if index < 5 else "short"
            rows.append(Sample(sample.root, sample.manifest, metadata))
        mixed = SimpleNamespace(samples=rows, manifest_hash=self.dataset.manifest_hash)
        with self.assertRaisesRegex(ValueError, "omits real groups"):
            evaluate_model(self.model_class(), mixed, exact=False, work_dir=self.root/"mixed",
                           session_groups={"complete": "pc-to-phone", "short": "phone-to-pc"})


if __name__ == "__main__":
    unittest.main()
