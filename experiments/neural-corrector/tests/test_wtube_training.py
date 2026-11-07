"""Optimizer/checkpoint integration on small v2 fixtures, not camera accuracy."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
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
from evaluate import evaluate_model
from model import NeuralCorrector
from train import run_training
from test_multires_training import fixture

torch.set_num_threads(1)


def fixture_v2(root, split, size, *, paired=False, seed=1):
    """Use the established phase fixture at the new four-pixel raster contract."""
    fixture(root, split, size, paired=paired, seed=seed)
    path = root/"manifest.json"
    value = json.loads(path.read_text())
    value["version"] = 2
    for row in value["samples"]:
        pixels = np.fromfile(root/row["input"], dtype="<f4").reshape(3, size*2, size*2)
        pixels = pixels.repeat(2, axis=1).repeat(2, axis=2)
        pixels.tofile(root/row["input"])
        row["hashes"]["input"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        row["inputShape"] = list(pixels.shape)
        row["pixelsPerCell"] = row["info"]["pixelsPerCell"] = 4
        row["origin"] = row["provenance"]["origin"] = "native"
    path.write_text(json.dumps(value))
    return root


class WTubeTrainingTests(unittest.TestCase):
    def test_v2_paired_mixed_grid_optimizer_checkpoint_evaluation_and_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, seed in (("train", 1), ("validation", 31)):
                for size in (16, 32):
                    fixture_v2(root/split/f"g{size}", split, size,
                               paired=split == "train", seed=seed+size)
            train = Dataset([root/"train"], "train")
            validation = Dataset([root/"validation"], "validation")
            training_shapes = []
            forward = NeuralCorrector.forward

            def observed_forward(model, pixels, *args, **kwargs):
                if model.training and torch.is_grad_enabled():
                    training_shapes.append(tuple(pixels.shape))
                return forward(model, pixels, *args, **kwargs)

            with patch.object(NeuralCorrector, "forward", observed_forward), \
                    patch("evaluate.canonical_recovery", side_effect=AssertionError("Raw Neural-QR has no packet recovery")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = run_training(train, validation, root/"run", model_profile="W-tube", frames=8,
                                      paired_targets=True, real_fraction=.5, steps=2, batch_size=2,
                                      accumulate=2, eval_every=2, no_progress=True, seed=7)
            self.assertEqual(result["steps"], 2)
            self.assertEqual(len(training_shapes), 4)
            self.assertTrue(all(shape[:2] == (2, 32) and shape[2] in (64, 128)
                                and shape[2] == shape[3] for shape in training_shapes))
            # Accumulated microbatches retain one geometry per optimizer update.
            self.assertEqual(training_shapes[0], training_shapes[1])
            self.assertEqual(training_shapes[2], training_shapes[3])
            update = result["history"][-1]
            self.assertEqual(update["validationPopulation"], "native")
            self.assertEqual(update["primarySizes"], [16, 32])
            self.assertEqual(set(update["validationOrigins"]), {"native"})
            self.assertEqual(update["trainSources"]["real"]["samples"], 4)
            self.assertEqual(update["trainSources"]["synthetic"]["samples"], 4)
            model, checkpoint = load_model_checkpoint(result["checkpoint"])
            meta = checkpoint["training"]
            self.assertEqual(model.profile, "W-tube")
            self.assertEqual(meta["datasetVersion"], 2)
            self.assertEqual(meta["pixelsPerCell"], 4)
            self.assertEqual(meta["supportedSizes"], [16, 32, 64, 128])
            self.assertTrue(set(meta["trainedSizes"]) <= {16, 32})
            self.assertTrue(meta["trainedSizes"])
            self.assertEqual(meta["bottleneck"]["stages"], 2)
            settings = json.loads((root/"run/settings.json").read_text())
            self.assertEqual(settings["frameCounts"], [4, 5, 6, 7, 8])
            self.assertEqual(settings["realFraction"], .5)
            self.assertTrue(settings["selectionMetric"].startswith("native-first/"))
            self.assertEqual(settings["gradientAccumulation"], 2)
            self.assertIsNotNone(settings["phaseAugmentation"])
            self.assertIsNotNone(settings["realAugmentation"])
            report = evaluate_model(model, validation)
            self.assertEqual(report["uniqueCandidates"], 2)
            self.assertEqual(set(report["bySize"]), {"16", "32"})
            self.assertEqual(set(report["byFrameCount"]), {"4", "5", "6", "7", "8"})
            before = Path(result["checkpoint"]).read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                continued = run_training(train, validation, root/"continued", initialize=result["checkpoint"],
                                         paired_targets=True, steps=1, batch_size=2, accumulate=1,
                                         eval_every=1, no_progress=True)
            self.assertEqual(continued["steps"], 1)
            continued_model, continued_checkpoint = load_model_checkpoint(continued["checkpoint"])
            self.assertEqual(continued_model.profile, "W-tube")
            self.assertEqual(continued_checkpoint["training"]["datasetVersion"], 2)
            self.assertEqual(json.loads((root/"continued/settings.json").read_text())["initialLearningRate"], 5e-5)
            self.assertEqual(Path(result["checkpoint"]).read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "differs from the initialized checkpoint"):
                run_training(train, validation, root/"wrong-initialization", initialize=result["checkpoint"],
                             model_profile="multires", paired_targets=True, steps=1)
            self.assertFalse((root/"wrong-initialization").exists())

    def test_v1_and_v2_cannot_silently_reach_the_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for version, factory in ((1, fixture), (2, fixture_v2)):
                for split, seed in (("train", 101), ("validation", 201)):
                    factory(root/f"v{version}"/split, split, 16,
                            paired=split == "train", seed=seed)
                train = Dataset([root/f"v{version}"/"train"], "train")
                validation = Dataset([root/f"v{version}"/"validation"], "validation")
                out = root/f"wrong-v{version}"
                with self.assertRaises(ValueError):
                    run_training(train, validation, out, model_profile="W-tube" if version == 1 else "multires",
                                 paired_targets=True, steps=1)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
