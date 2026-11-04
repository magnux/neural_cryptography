"""Direction-balanced evaluation without changing decoder evidence."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from evaluate import canonical_recovery, evaluate_model, summarize_groups
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for evaluation")
class GroupEvaluationTests(unittest.TestCase):
    def test_direction_macro_score_does_not_follow_capture_count_or_synthetic_mix(self):
        def row(source, samples, cells, wrong, loss, exact):
            return {"source": source, "samples": samples, "dataCells": cells,
                    "wrongCells": wrong, "loss": loss,
                    "learnedExactFrames": exact, "classicalExactFrames": 0}
        sessions = {
            "a": row("real", 3, 12, 0, .1, 3),
            "b": row("real", 1, 4, 4, 2.1, 0),
            "c": row("real", 1, 8, 4, 1.1, 0),
            "synthetic": row("synthetic", 100, 400, 0, .01, 100),
        }
        report = summarize_groups(sessions, {"a": "easy", "b": "hard", "c": "hard"})
        easy, hard = report["groups"]["easy"], report["groups"]["hard"]
        self.assertAlmostEqual(hard["loss"], (2.1*4+1.1*8)/12)
        self.assertEqual(easy["learnedExactRate"], 1)
        self.assertEqual(hard["learnedExactRate"], 0)
        score = report["selectionScore"]
        self.assertEqual(score["method"], "group-balanced-real")
        self.assertEqual(score["exactRate"], .5)
        self.assertAlmostEqual(score["loss"], (.1+hard["loss"])/2)
        self.assertAlmostEqual(score["cellErrorRate"], 1/3)
        self.assertEqual(report["sources"]["real"]["learnedExactRate"], 3/5)
        self.assertEqual(report["groups"]["source:synthetic"]["learnedExactFrames"], 100)
        global_score = summarize_groups(sessions)["selectionScore"]
        self.assertEqual(global_score["method"], "global")
        self.assertEqual(global_score["exactRate"], 103/105)

    def test_metrics_only_evaluation_keeps_recovery_unknown_and_restores_model_mode(self):
        class Model(torch.nn.Module):
            def forward(self, values):
                output = torch.zeros((len(values), 16, 2, 2), device=values.device)
                output[:, 0] = 3
                return output
        class Sample:
            def __init__(self, session, index, label):
                self.id = f"{session}-{index}"
                self.shape = (64, 4)
                self.label = label
                self.metadata = {"sessionId": session, "source": "real", "size": 64,
                                 "payloadId": "a"*64, "split": "validation",
                                 "colors": 4, "info": {"size": 64, "colors": 4, "phase": index % 4}}
            def tensors(self):
                return (np.zeros((3, 8, 8), np.float32), np.full((2, 2), self.label, np.int64),
                        np.ones((2, 2), bool))
        dataset = SimpleNamespace(samples=[Sample("a", i, 0) for i in range(3)]+[Sample("b", 0, 1)],
                                  manifest_hash="0"*64)
        model = Model().train()
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            report = evaluate_model(model, dataset, exact=False, work_dir=directory,
                                    session_groups={"a": "easy", "b": "hard"}, detailed=True)
            self.assertTrue(model.training)
            self.assertIsNone(report["recovery"])
            self.assertIsNone(report["selectionScore"]["exactRate"])
            self.assertEqual(report["cellErrorRate"], .25)
            self.assertEqual(report["selectionScore"]["cellErrorRate"], .5)
            self.assertEqual(report["sessions"]["a"]["phases"], {"0": 1, "1": 1, "2": 1})
            self.assertAlmostEqual(report["sessions"]["b"]["loss"]-report["sessions"]["a"]["loss"], 3, places=6)
            for group in report["groups"].values():
                self.assertIsNone(group["learnedExactFrames"])
                self.assertIsNone(group["classicalExactRate"])
            with self.assertRaisesRegex(ValueError, "omits real groups"):
                evaluate_model(model, dataset, exact=False, work_dir=directory, max_samples=1,
                               session_groups={"a": "easy", "b": "hard"})
            with self.assertRaises(ValueError):
                evaluate_model(model, dataset, exact=False, work_dir=directory,
                               session_groups={"a": "easy"})

    def test_recovery_identities_are_attached_after_the_decoder_returns(self):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root = Path(directory)
            sample = SimpleNamespace(id="sample-one", manifest=root/"data/manifest.json",
                metadata={"sessionId": "session-one", "source": "real", "info": {"phase": 3}})
            predictions = {sample.id: root/"prediction.f32"}
            outcome = {"exact": True, "accepted": True, "wrongAccepted": False,
                       "rawCRC": False, "hardCRC": True}
            response = {"samples": [{"id": sample.id, "memberIds": [sample.id], "learned": outcome, "classical": {**outcome, "exact": False}}]}
            process = SimpleNamespace(returncode=0, stdout=json.dumps(response))
            with patch("evaluate.subprocess.run", return_value=process):
                report = canonical_recovery(None, [sample], predictions, root,
                                            session_groups={"session-one": "phone_to_pc"})
            row = report["samples"][0]
            self.assertEqual((row["sessionId"], row["phase"], row["group"]),
                             ("session-one", 3, "phone_to_pc"))
            self.assertEqual(report["summary"]["learned"]["exactRate"], 1)
            sent = json.loads((root/"predictions-0.json").read_text())
            self.assertEqual(sent, {"framesPerInput": 1, "outputLabelSpace": "physical", "samples": [{"id": "sample-one", "memberIds": ["sample-one"], "logits": "prediction.f32"}]})
            process.stdout = json.dumps({"samples": response["samples"]*2})
            with patch("evaluate.subprocess.run", return_value=process), self.assertRaisesRegex(RuntimeError, "different sample set"):
                canonical_recovery(None, [sample], predictions, root)


if __name__ == "__main__":
    unittest.main()
