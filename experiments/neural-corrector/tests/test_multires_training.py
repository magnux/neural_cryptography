"""Small mixed-grid optimizer/scoring checks; no claim about camera accuracy."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from checkpoint import load_model_checkpoint
    from data import Dataset
    from evaluate import evaluate_model
    from model import NeuralCorrector
    from multires_evaluation import _selection, add, counter
    from reporting import compact_report
    from train import run_training, save_checkpoint
    torch.set_num_threads(1)


def fixture(root, split, size, *, paired=False, seed=1):
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    logical = rng.integers(0, 16, (size, size), dtype=np.uint8)
    mask = np.ones((size, size), dtype=np.uint8)
    mask[0] = 0
    expected = logical.tobytes()
    payload = hashlib.sha256(expected).hexdigest()
    session = f"{split}-session-{size}"
    samples = []
    def store(name, array):
        raw = array if isinstance(array, bytes) else array.tobytes()
        (root/name).write_bytes(raw)
        return name, hashlib.sha256(raw).hexdigest()
    for phase, xor in enumerate((0, 15, 7, 8)):
        labels = logical.copy()
        labels[mask.astype(bool)] ^= xor
        info = {"size": size, "colors": 16, "phases": 4, "gridBlock": 8,
                "parity": 0, "permute": True, "pixelsPerCell": 2, "phase": phase}
        for source in (["real", "synthetic"] if paired else ["real"]):
            for frame in range(2):
                name = f"{split}-{size}-{phase}-{source}-{frame}"
                pixels = rng.uniform(-1.2, 1.2, (3, size*2, size*2)).astype("<f4")
                row = {"id": name, "sessionId": session, "senderSessionId": f"sender-{split}-{size}",
                       "payloadId": payload, "split": split, "source": source,
                       "size": size, "colors": 16, "pixelsPerCell": 2,
                       "inputShape": list(pixels.shape), "info": info,
                       "burstId": f"{session}-{source}-{phase}", "timestamp": frame*100,
                       "provenance": {"senderEnvironment": {"deviceClass": "mobile"},
                                      "receiverEnvironment": {"deviceClass": "desktop"}},
                       "hashes": {}}
                if source == "synthetic":
                    row["synthetic"] = {"phaseGroupId": "trajectory-0", "frame": frame}
                for key, value in {"input": pixels, "labels": labels, "mask": mask,
                                   "expected": expected, "baseline": np.zeros((size, size, 16), dtype="<f4")}.items():
                    row[key], row["hashes"][key] = store(name+"."+key, value)
                samples.append(row)
    (root/"manifest.json").write_text(json.dumps({"format": "neural-qr-dataset", "version": 1, "samples": samples}))
    return root


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class MultiresTrainingTests(unittest.TestCase):
    def test_selection_balances_sizes_directions_and_counts_not_pixel_count(self):
        leaves = {}
        for size, wrong in ((8, 64), (256, 0)):
            for k in range(4, 9):
                row = counter()
                for _ in range(20 if size == 8 else 1):
                    add(row, loss=2. if size == 8 else 0., wrong=wrong,
                        cells=size*size, captures=k)
                leaves[(size, "real", "phone-to-pc", k)] = row
        synthetic = counter()
        add(synthetic, loss=100, wrong=65536, cells=65536, captures=4)
        leaves[(256, "synthetic", "source:synthetic", 4)] = synthetic
        score = _selection(leaves)
        self.assertEqual(score["exactRate"], .5)
        self.assertEqual(score["loss"], 1.)
        self.assertEqual(score["cellErrorRate"], .5)

    def test_paired_mixed_grid_training_checkpoint_and_matrix_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, seed in (("train", 1), ("validation", 10)):
                for size in (8, 16):
                    fixture(root/split/f"g{size}", split, size, paired=split == "train", seed=seed+size)
            train = Dataset([root/"train"], "train")
            validation = Dataset([root/"validation"], "validation")
            with patch("evaluate.canonical_recovery", side_effect=AssertionError("No packet decoder for raw Neural-QR")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                result = run_training(train, validation, root/"run", model_profile="multires", frames=8,
                                      paired_targets=True, steps=2, batch_size=2, accumulate=1,
                                      eval_every=2, no_progress=True, seed=7)
            self.assertEqual(result["steps"], 2)
            self.assertIn("Val grids", output.getvalue())
            model, checkpoint = load_model_checkpoint(result["checkpoint"])
            metadata = checkpoint["training"]
            self.assertEqual(metadata["datasetFormat"], "neural-qr-dataset")
            self.assertEqual(metadata["pixelsPerCell"], 2)
            self.assertTrue(set(metadata["trainedSizes"]) <= {8, 16})
            self.assertTrue(metadata["trainedSizes"])
            self.assertEqual(metadata["selectionAggregation"], "size-group-frame-balanced-real")
            update = result["history"][-1]
            self.assertEqual(update["trainSources"]["real"]["samples"], 2)
            self.assertEqual(update["trainSources"]["synthetic"]["samples"], 2)
            self.assertNotIn("classicalExact", update)
            report = evaluate_model(model, validation, detailed=True)
            lean = compact_report(report)
            self.assertEqual(lean, compact_report(lean))
            self.assertNotIn("cases", lean)
            self.assertNotIn("recovery", lean)
            self.assertEqual(set(lean["bySize"]), {"8", "16"})
            self.assertEqual(set(lean["byFrameCount"]), {"4", "5", "6", "7", "8"})
            self.assertEqual(lean["samples"], 10)
            self.assertEqual(lean["uniqueCandidates"], 2)
            self.assertEqual(len(report["cases"]), 10)
            self.assertLess(len(json.dumps(result["history"])), 100_000)
            by_count = {k: [row for row in report["cases"] if row["frames"] == k] for k in range(4, 9)}
            for a, b in zip(by_count[4], by_count[8]):
                self.assertEqual(a["id"], b["id"])
                self.assertTrue(set(a["memberIds"]) <= set(b["memberIds"]))
            with self.assertRaisesRegex(ValueError, "cap omits"):
                evaluate_model(model, validation, max_samples=1)
            # Explicit OBM attention warm start transfers useful tensors, but its
            # G64 anchor is not marked trained for the new two-pixel cell geometry.
            old_model = NeuralCorrector("U-tube-2x-attn", frames=8)
            old_metadata = copy.deepcopy(metadata)
            old_metadata.update({"modelProfile": "U-tube-2x-attn", "datasetFormat": "optical-decoder-dataset",
                                 "trainSessionIds": [], "trainPayloadIds": [],
                                 "validationSessionIds": [], "validationPayloadIds": [],
                                 "steps": 1, "selectionStep": 1, "baselineSelection": False})
            for name in ("trainedSizes", "supportedSizes", "pixelsPerCell"):
                old_metadata.pop(name, None)
            save_checkpoint(root/"old.pt", old_model, old_metadata)
            with contextlib.redirect_stdout(io.StringIO()):
                warm = run_training(train, validation, root/"warm", model_profile="multires", frames=8,
                                    initialize=root/"old.pt", paired_targets=True, steps=1, batch_size=2,
                                    accumulate=1, eval_every=1, no_progress=True, seed=7)
            _, warm_checkpoint = load_model_checkpoint(warm["checkpoint"])
            self.assertFalse(warm["history"][0]["selected"])
            self.assertNotIn(64, warm_checkpoint["training"]["trainedSizes"])
            self.assertIn("warmStartTransfer", warm_checkpoint["training"]["settings"])
            # Independent G64+ anchors received no optimizer update or weight decay.
            torch.manual_seed(7)
            fresh = NeuralCorrector("multires", frames=8)
            for size in (32, 64, 128, 256):
                torch.testing.assert_close(model.anchors[str(size)], fresh.anchors[str(size)], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
