"""One-time metadata edits preserve trained tensors and fail before replacement."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "neural-corrector"))
import torch
from checkpoint import load_model_checkpoint
from model import NeuralCorrector

SPEC = importlib.util.spec_from_file_location("rename_corrector_checkpoints", ROOT / "scripts" / "rename-corrector-checkpoints.py")
rename = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rename
SPEC.loader.exec_module(rename)
torch.set_num_threads(1)


class CheckpointRenameTests(unittest.TestCase):
    def fixture(self, path, old="normalized-tube"):
        profile = rename.RENAMES.get(old, old)
        frames = 8 if profile == "U-tube-2x-attn" else 5
        model = NeuralCorrector(profile, frames=frames)
        with torch.no_grad():
            model.head.bias[0] = -0.0
            model.head.bias[1] = .1234567
        value = {
            "format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
            "modelState": model.state_dict(),
            "training": {
                "steps": 32000, "seed": 7, "framesPerInput": frames,
                "modelProfile": old, "datasetFormat": "optical-decoder-dataset",
                "trainSessionIds": ["train-session"], "trainPayloadIds": ["train-payload"],
                "validationSessionIds": ["validation-session"], "validationPayloadIds": ["validation-payload"],
                "localPath": "checkpoints/optical-decoder/attention-tube",
            },
            "notes": {"archiveLabel": old, "values": [1, "do not rename", None]},
        }
        if profile == "U-tube-2x-attn":
            value["training"].update(inputLayout="phase-paired-rgb-presence-planes",
                outputLabelSpace="logical", minFramesPerInput=4,
                frameCounts=[4, 5, 6, 7, 8], frameCountPolicy="uniform-4-8", phaseMasks=[0, 15, 7, 8])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return value

    def snapshot(self, path):
        return path.read_bytes(), path.stat().st_mtime_ns

    def assert_no_temporary(self, root):
        self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_three_real_architectures_change_only_the_profile_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / str(i) / "best.pt" for i in range(3)]
            originals = [self.fixture(path, old) for path, old in zip(paths, rename.RENAMES)]
            archive = root / "history.json"
            archive.write_text('{"modelProfile":"attention-tube"}')
            output = []
            rename.rename_checkpoints(paths, emit=output.append)
            self.assertEqual(len(output), 3)
            for path, before in zip(paths, originals):
                model, after = load_model_checkpoint(path)
                self.assertEqual(model.profile, rename.RENAMES[before["training"]["modelProfile"]])
                expected = copy.deepcopy({k: v for k, v in before.items() if k != "modelState"})
                expected["training"]["modelProfile"] = model.profile
                self.assertEqual({k: v for k, v in after.items() if k != "modelState"}, expected)
                self.assertEqual(set(after["modelState"]), set(before["modelState"]))
                for name, tensor in before["modelState"].items():
                    self.assertTrue(torch.equal(tensor.contiguous().view(torch.uint8),
                                                after["modelState"][name].contiguous().view(torch.uint8)), name)
            self.assertEqual(archive.read_text(), '{"modelProfile":"attention-tube"}')
            self.assert_no_temporary(root)

    def test_dry_run_and_repeated_application_do_not_rewrite_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "best.pt"
            self.fixture(path); before = self.snapshot(path)
            messages = []
            rename.rename_checkpoints([path], dry_run=True, emit=messages.append)
            self.assertEqual(self.snapshot(path), before)
            self.assertTrue(messages[0].startswith("WOULD UPDATE"))
            rename.rename_checkpoints([path], emit=lambda _: None)
            renamed = self.snapshot(path)
            rename.rename_checkpoints([path], emit=messages.append)
            self.assertEqual(self.snapshot(path), renamed)
            self.assertTrue(messages[-1].startswith("UNCHANGED"))
            self.assert_no_temporary(root)

    def test_preflight_rejects_unknown_or_mismatched_profile_without_changing_any_file(self):
        for defect in ("unrecognized", "attention-tube"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); first = root / "first.pt"; invalid = root / "invalid.pt"
                self.fixture(first)
                value = self.fixture(invalid)
                value["training"]["modelProfile"] = defect
                torch.save(value, invalid)
                before = [self.snapshot(path) for path in (first, invalid)]
                with self.assertRaises(ValueError):
                    rename.rename_checkpoints([first, invalid], emit=lambda _: None)
                self.assertEqual([self.snapshot(path) for path in (first, invalid)], before)
                self.assert_no_temporary(root)

    def test_invalid_tensors_and_phase_contract_are_not_repaired_by_renaming(self):
        for defect in ("nan", "missing-phase-mask"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); path = root / "best.pt"
                value = self.fixture(path, "attention-tube")
                if defect == "nan":
                    value["modelState"]["head.bias"][0] = float("nan")
                else:
                    del value["training"]["phaseMasks"]
                torch.save(value, path); before = self.snapshot(path)
                with self.assertRaises(ValueError):
                    rename.rename_checkpoints([path], emit=lambda _: None)
                self.assertEqual(self.snapshot(path), before)
                self.assert_no_temporary(root)

    def test_save_and_replace_failures_leave_the_original_intact(self):
        for owner, attribute in ((rename.torch, "save"), (rename.os, "replace")):
            with self.subTest(operation=attribute), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); path = root / "best.pt"
                self.fixture(path); before = self.snapshot(path)
                with mock.patch.object(owner, attribute, side_effect=OSError("simulated I/O failure")):
                    with self.assertRaisesRegex(OSError, "simulated"):
                        rename.rename_checkpoints([path], emit=lambda _: None)
                self.assertEqual(self.snapshot(path), before)
                self.assert_no_temporary(root)

    def test_optional_absent_profile_and_default_folder_names(self):
        self.assertEqual([path.parent.name for path in rename.DEFAULT_PATHS],
                         ["u-tube-burst5", "u-tube-2x-burst5", "u-tube-2x-attn-phase8"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            value = self.fixture(path); del value["training"]["modelProfile"]
            torch.save(value, path); before = self.snapshot(path)
            rename.rename_checkpoints([path], emit=lambda _: None)
            self.assertEqual(self.snapshot(path), before)


if __name__ == "__main__":
    unittest.main()
