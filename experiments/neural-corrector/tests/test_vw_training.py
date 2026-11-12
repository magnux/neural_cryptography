"""VW training/evaluation uses the established four-pixel phase-burst corpus."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from checkpoint import load_model_checkpoint
from data import Dataset
from evaluate import evaluate_model, require_implemented_model_contract
from model import NeuralCorrector
from train import origin_sampling_options, run_training
from test_wtube_training import fixture_v2
from test_wtube_model import checkpoint as model_checkpoint

HAS_EXPORT = importlib.util.find_spec("onnx") is not None


def fingerprints(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class VWTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)

    def test_vw_uses_four_pixel_contract_and_global_origin_policy(self):
        data = SimpleNamespace(dataset_format="neural-qr-dataset", dataset_version=2,
                               pixels_per_cell=4)
        require_implemented_model_contract(data, "VW")
        self.assertEqual(origin_sampling_options("VW"),
                         (None, {"native": .5, "resampled": .35, "header-g64": .15}))
        self.assertEqual(origin_sampling_options("VW", .6), (.6, None))
        self.assertEqual(origin_sampling_options("VW", origin_ratios="native=.60,resampled=.25,header-g64=.15"),
                         (None, {"native": .6, "resampled": .25, "header-g64": .15}))
        for wrong in (
                SimpleNamespace(dataset_format="neural-qr-dataset", dataset_version=1, pixels_per_cell=2),
                SimpleNamespace(dataset_format="optical-decoder-dataset", dataset_version=1, pixels_per_cell=4)):
            with self.assertRaises(ValueError):
                require_implemented_model_contract(wrong, "VW")

    def test_cli_scratch_weighted_training_evaluation_and_own_profile_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root/"data"
            for split, seed in (("train", 501), ("validation", 601)):
                for size in (16, 32):
                    fixture_v2(data_root/split/f"g{size}", split, size,
                               paired=split == "train", seed=seed+size)
            before = fingerprints(data_root)
            out = root/"fresh"
            command = [sys.executable, str(EXPERIMENT/"train.py"),
                "--train", str(data_root/"train"), "--validation", str(data_root/"validation"),
                "--model", "VW", "--frames", "8", "--paired-targets", "--real-fraction", ".5",
                "--origin-ratios", "native=1,resampled=0,header-g64=0",
                "--loss", "rs-weighted-ce", "--rs-loss-mix", ".25", "--rs-parity", "32",
                "--phase-flip-probability", "1", "--steps", "2", "--batch-size", "2",
                "--accumulate", "2", "--device", "cpu", "--no-progress", "--out", str(out)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stderr[-4000:])
            self.assertIn("| objective ", process.stdout)
            self.assertIn("RS replay |", process.stdout)
            self.assertIn("native G16/G32", process.stdout)
            settings = json.loads((out/"settings.json").read_text())
            self.assertEqual(settings["modelProfile"], "VW")
            self.assertEqual(settings["bottleneck"]["stages"], 3)
            self.assertEqual(settings["initialLearningRate"], 1e-4)
            self.assertEqual(settings["datasetVersion"], 2)
            self.assertEqual(settings["pixelsPerCell"], 4)
            self.assertEqual(settings["loss"], "rs-weighted-cross-entropy")
            self.assertEqual(settings["rsLoss"]["weightedFraction"], .25)
            self.assertEqual(settings["phaseAugmentation"]["horizontalFlipProbability"], 1.)
            self.assertTrue(settings["phaseAugmentation"]["phasePermutations"])
            self.assertTrue(settings["selectionMetric"].startswith("native-first/"))
            self.assertEqual(settings["originSampling"]["allocation"],
                             "global-real-origin/eligible-grid/available-direction")
            model, checkpoint = load_model_checkpoint(out/"best.pt")
            self.assertEqual(model.profile, "VW")
            self.assertEqual(len(model.stages), 3)
            self.assertEqual(checkpoint["training"]["bottleneck"]["stages"], 3)
            history = json.loads((out/"history.json").read_text())
            summary = history["history"][-1]["training"]
            self.assertIsNotNone(summary["meanObjective"])
            rows = [dict(zip(history["tables"]["training"], row)) for row in summary["rows"]]
            self.assertEqual(sum(row["samples"] for row in rows), 8)
            self.assertEqual({row["origin"] for row in rows}, {"native"})
            self.assertEqual({row["source"] for row in rows}, {"real", "synthetic"})
            self.assertEqual(sum(summary["frameCounts"]), 8)
            validation = Dataset([data_root/"validation"], "validation")
            report = evaluate_model(model, validation, rs_parity=32, work_dir=root/"replay")
            self.assertEqual(report["modelProfile"], "VW")
            self.assertEqual(report["evaluationPopulation"], "native")
            self.assertEqual(report["primarySizes"], [16, 32])
            self.assertEqual(report["rsReplay"]["eligibleCases"], 10)
            self.assertEqual(set(report["byFrameCount"]), {"4", "5", "6", "7", "8"})
            # A real CLI evaluation also enforces the trained-anchor boundary.
            trained_size = checkpoint["training"]["trainedSizes"][0]
            evaluation = root/"evaluation.json"
            process = subprocess.run([sys.executable, str(EXPERIMENT/"evaluate.py"),
                "--dataset", str(data_root/"validation"/f"g{trained_size}"),
                "--checkpoint", str(out/"best.pt"), "--split", "validation",
                "--device", "cpu", "--rs-parity", "32", "--out", str(evaluation)],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stderr[-4000:])
            self.assertEqual(json.loads(evaluation.read_text())["metadata"]["modelProfile"], "VW")
            original_checkpoint = (out/"best.pt").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                continued = run_training(Dataset([data_root/"train"], "train"), validation,
                    root/"continued", initialize=out/"best.pt", model_profile="VW",
                    paired_targets=True, steps=1, batch_size=2, accumulate=1, no_progress=True,
                    origin_ratios="native=1,resampled=0,header-g64=0",
                    loss="rs-weighted-ce", rs_loss_mix=.25)
            loaded, metadata = load_model_checkpoint(continued["checkpoint"])
            self.assertEqual(loaded.profile, "VW")
            self.assertEqual(len(loaded.stages), 3)
            self.assertEqual(metadata["training"]["settings"]["initialLearningRate"], 5e-5)
            self.assertEqual((out/"best.pt").read_bytes(), original_checkpoint)
            self.assertEqual(fingerprints(data_root), before)

    def test_two_pass_checkpoint_cannot_silently_initialize_vw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_v2(root/"train", "train", 16, paired=True, seed=801)
            fixture_v2(root/"validation", "validation", 16, seed=802)
            path = root/"two-pass.pt"
            torch.save(model_checkpoint(NeuralCorrector("W-tube")), path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "differs from the initialized checkpoint"):
                run_training(Dataset([root/"train"], "train"),
                    Dataset([root/"validation"], "validation"), root/"wrong-run",
                    model_profile="VW", initialize=path, steps=1,
                    origin_ratios="native=1,resampled=0,header-g64=0")
            self.assertFalse((root/"wrong-run").exists())
            self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(HAS_EXPORT, "ONNX needed to import the exporter")
    def test_unsupported_export_grid_does_not_create_or_replace_browser_bundle(self):
        from export import export_smoke
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/"bundle"
            with self.assertRaisesRegex(ValueError, "supports grid sizes"):
                export_smoke(out, profile="VW", sizes=(8,), check_ort=False)
            self.assertFalse(out.exists())
            out.mkdir()
            sentinel = out/"keep.txt"
            sentinel.write_text("existing bundle")
            with self.assertRaisesRegex(ValueError, "supports grid sizes"):
                export_smoke(out, profile="VW", sizes=(256,), check_ort=False)
            self.assertEqual(sentinel.read_text(), "existing bundle")
            self.assertEqual(list(out.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
