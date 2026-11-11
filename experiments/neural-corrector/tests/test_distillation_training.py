"""EMA integration respects accumulation, data geometry and inference checkpoints."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from checkpoint import load_model_checkpoint
from data import Dataset
from distillation import EMADistillation, StudentAugmentation
from model import NeuralCorrector
from train import run_training
from test_wtube_model import checkpoint as model_checkpoint
from test_wtube_training import fixture_v2


def fingerprints(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class DistillationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)

    def test_accumulation_updates_teacher_once_and_keeps_geometry_and_checkpoint_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_v2(root/"data/train", "train", 16, paired=True, seed=701)
            fixture_v2(root/"data/validation", "validation", 16, seed=702)
            before = fingerprints(root/"data")
            train = Dataset([root/"data/train"], "train")
            validation = Dataset([root/"data/validation"], "validation")
            instances, events, weak_views, strong_views = [], [], [], []
            case = self

            class ObservedDistillation(EMADistillation):
                def __init__(self, student, **kwargs):
                    super().__init__(student, **kwargs)
                    self.initial_projection = {name: value.detach().clone()
                                               for name, value in self.projector.named_parameters()}
                    instances.append(self)

                def target(self, pixels):
                    weak_views.append(pixels.detach().clone())
                    result = super().target(pixels)
                    case.assertFalse(result.requires_grad)
                    case.assertFalse(self.teacher.training)
                    return result

                def update(self, student):
                    case.assertEqual(events[-1], "optimizer")
                    case.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))
                    case.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                        for parameter in self.projector.parameters()))
                    events.append("ema")
                    return super().update(student)

            original_step = torch.optim.AdamW.step
            original_augment = StudentAugmentation.__call__

            def observed_step(optimizer, *args, **kwargs):
                result = original_step(optimizer, *args, **kwargs)
                events.append("optimizer")
                return result

            def observed_augmentation(augmentation, pixels):
                result = original_augment(augmentation, pixels)
                strong_views.append(result.detach().clone())
                # Strong augmentation may touch RGB only; zero slots remain absent.
                torch.testing.assert_close(result[:, 24:], pixels[:, 24:], rtol=0, atol=0)
                rgb = result[:, :24].reshape(len(result), 8, 3, *result.shape[-2:])
                present = result[:, 24:, 0, 0].bool()
                self.assertTrue(torch.equal(rgb[~present], torch.zeros_like(rgb[~present])))
                return result

            output = io.StringIO()
            with patch("train.EMADistillation", ObservedDistillation), \
                    patch.object(torch.optim.AdamW, "step", observed_step), \
                    patch.object(StudentAugmentation, "__call__", observed_augmentation), \
                    contextlib.redirect_stdout(output):
                result = run_training(train, validation, root/"run", model_profile="W-tube",
                    frames=8, paired_targets=True, steps=2, batch_size=2, accumulate=2,
                    eval_every=2, no_progress=True, seed=7,
                    origin_ratios="native=1,resampled=0,header-g64=0",
                    loss="rs-weighted-ce", rs_loss_mix=.25, rs_parity=32,
                    self_distillation=True, distill_weight=.05, distill_ramp=2,
                    ema_decay=.9, phase_flip_probability=1.)

            self.assertEqual(events, ["optimizer", "ema", "optimizer", "ema"])
            self.assertEqual(len(instances), 1)
            self.assertEqual(len(weak_views), 4)
            self.assertEqual(len(strong_views), 4)
            for weak, strong in zip(weak_views, strong_views):
                self.assertEqual(tuple(weak.shape), (2, 32, 64, 64))
                torch.testing.assert_close(weak[:, 24:], strong[:, 24:], rtol=0, atol=0)
                self.assertFalse(torch.equal(weak[:, :24].clamp(-1, 1), strong[:, :24]))
            distiller = instances[0]
            self.assertTrue(any(not torch.equal(value, distiller.initial_projection[name])
                                for name, value in distiller.projector.named_parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in distiller.teacher.parameters()))
            self.assertFalse(distiller.teacher.training)

            model, saved = load_model_checkpoint(result["checkpoint"])
            expected = NeuralCorrector("W-tube").state_dict()
            self.assertEqual(set(saved["modelState"]), set(expected))
            self.assertFalse(any("teacher" in name or "projector" in name for name in saved["modelState"]))
            self.assertEqual(model.profile, "W-tube")
            history = json.loads((root/"run/history.json").read_text())
            self.assertEqual(history["completedSteps"], 2)
            row = history["history"][-1]["training"]
            self.assertEqual(row["updates"], 2)
            self.assertEqual(sum(row["frameCounts"]), 8)
            self.assertGreater(row["meanDistillation"], 0.)
            self.assertGreater(row["meanSupervisedObjective"], 0.)
            self.assertGreater(row["meanWeightedDistillation"], 0.)
            self.assertAlmostEqual(row["meanDistillationWeight"], .0375)
            self.assertAlmostEqual(row["meanObjective"],
                                   row["meanSupervisedObjective"]+row["meanWeightedDistillation"], places=6)
            settings = json.loads((root/"run/settings.json").read_text())
            self.assertEqual(settings["selfDistillation"]["emaDecay"], .9)
            self.assertEqual(saved["training"]["settings"]["selfDistillation"], settings["selfDistillation"])
            self.assertIn(" | SD ", output.getvalue())
            self.assertLess((root/"run/history.json").stat().st_size, 40_000)
            self.assertEqual(fingerprints(root/"data"), before)

    def test_cli_distillation_continues_an_ordinary_vw_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_v2(root/"train", "train", 16, paired=True, seed=711)
            fixture_v2(root/"validation", "validation", 16, seed=712)
            initialized = root/"original.pt"
            value = model_checkpoint(NeuralCorrector("VW"))
            value["training"]["modelProfile"] = "VW"
            torch.save(value, initialized)
            original = initialized.read_bytes()
            out = root/"continued"
            process = subprocess.run([sys.executable, str(EXPERIMENT/"train.py"),
                "--train", str(root/"train"), "--validation", str(root/"validation"),
                "--model", "VW", "--frames", "8", "--initialize", str(initialized),
                "--paired-targets", "--real-fraction", ".5",
                "--origin-ratios", "native=1,resampled=0,header-g64=0",
                "--self-distillation", "--distill-weight", ".05", "--distill-ramp", "2",
                "--ema-decay", ".9", "--distill-noise-std", ".01",
                "--distill-tint-stops", ".04", "--distill-exposure-stops", ".04",
                "--steps", "1", "--batch-size", "2", "--accumulate", "1",
                "--device", "cpu", "--no-progress", "--out", str(out)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stderr[-4000:])
            model, saved = load_model_checkpoint(out/"best.pt")
            self.assertEqual(model.profile, "VW")
            self.assertEqual(set(saved["modelState"]), set(value["modelState"]))
            self.assertEqual(initialized.read_bytes(), original)
            settings = json.loads((out/"settings.json").read_text())
            self.assertEqual(settings["initialLearningRate"], 5e-5)
            self.assertEqual(settings["modelProfile"], "VW")
            self.assertEqual(settings["selfDistillation"]["emaDecay"], .9)
            history = json.loads((out/"history.json").read_text())
            self.assertEqual(history["completedSteps"], 1)
            # Continuation starts a fresh training-only ramp; no teacher state
            # or projection metadata is required in the original checkpoint.
            self.assertAlmostEqual(history["history"][-1]["training"]["meanDistillationWeight"], .025)

    def test_invalid_distillation_options_do_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_v2(root/"train", "train", 16, paired=True, seed=721)
            fixture_v2(root/"validation", "validation", 16, seed=722)
            train = Dataset([root/"train"], "train")
            validation = Dataset([root/"validation"], "validation")
            for index, invalid in enumerate((
                    {"ema_decay": 1.}, {"ema_decay": float("nan")},
                    {"distill_weight": -.1}, {"distill_ramp": -1},
                    {"distill_noise_std": -.1}, {"distill_tint_stops": float("inf")})):
                with self.subTest(invalid=invalid):
                    out = root/f"bad-{index}"
                    with self.assertRaises(ValueError):
                        run_training(train, validation, out, model_profile="W-tube", frames=8,
                            self_distillation=True, steps=1, no_progress=True,
                            origin_ratios="native=1,resampled=0,header-g64=0", **invalid)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
