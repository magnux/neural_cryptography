"""Trained export preserves actual weights and rejects malformed checkpoints."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_EXPORT = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("onnx") is not None
if HAS_EXPORT:
    import torch
    from model import OpticalDecoder
    from export import export_smoke, load_checkpoint
    from checkpoint import training_provenance
    torch.set_num_threads(1)


class CustomMetadata:
    """A value weights_only must reject instead of importing arbitrary classes."""


@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX are required for checkpoint export")
class CheckpointExportTests(unittest.TestCase):
    def checkpoint(self):
        torch.manual_seed(31)
        model = OpticalDecoder().eval()
        # Distinct output bias makes accidental fresh initialization detectable.
        with torch.no_grad():
            model.head.bias.add_(.25)
        return model, {
            "format": "optical-decoder-checkpoint", "version": 1,
            "status": "trained", "modelState": model.state_dict(),
            "training": {"steps": 2, "seed": 31, "samples": 16,
                         "selection": "validation-exact", "selectionStep": 2,
                         "trainManifestSha256": "1" * 64,
                         "validationManifestSha256": "2" * 64,
                         "initialCheckpointSha256": "3" * 64,
                         "datasetFormat": "optical-decoder-dataset",
                         "localDatasetPath": "/private/local-data/session",
                         "trainSessionIds": ["private-session-description"],
                         "trainPayloadIds": ["train-payload"],
                         "validationSessionIds": ["validation-session"],
                         "validationPayloadIds": ["validation-payload"]},
        }

    def test_export_uses_checkpoint_weights_and_sanitized_provenance(self):
        model, value = self.checkpoint()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "trained.pt"
            torch.save(value, checkpoint)
            output = Path(temporary) / "bundle"
            manifest = export_smoke(output, sizes=(64,), check_ort=False, checkpoint=checkpoint)
            self.assertEqual(manifest["format"], "optical-decoder")
            self.assertEqual(manifest["status"], "trained")
            self.assertEqual(manifest["checkpointSha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(manifest["training"]["steps"], 2)
            self.assertEqual(manifest["training"]["initialCheckpointSha256"], "3" * 64)
            self.assertNotIn("localDatasetPath", manifest["training"])
            self.assertNotIn("trainSessionIds", manifest["training"])
            self.assertEqual(manifest["pythonParityStatus"]["status"], "not_run")
            case = manifest["cases"][0]
            pixels = np.fromfile(output / case["input"]["file"], dtype="<f4").reshape(case["inputShape"])
            exported = np.fromfile(output / case["expected"]["file"], dtype="<f4").reshape(case["outputShape"])
            with torch.inference_mode():
                expected = model(torch.from_numpy(pixels)).numpy()
            np.testing.assert_array_equal(exported, expected)

    def test_wrong_shape_dtype_nonfinite_keys_and_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.pt"
            for case in ("shape", "dtype", "nan", "missing", "extra", "format", "version", "steps", "hash", "overlap", "missing-splits"):
                with self.subTest(case=case):
                    model, value = self.checkpoint()
                    key = next(iter(value["modelState"]))
                    if case == "shape": value["modelState"][key] = torch.zeros(1)
                    elif case == "dtype": value["modelState"][key] = value["modelState"][key].double()
                    elif case == "nan": value["modelState"][key].view(-1)[0] = float("nan")
                    elif case == "missing": del value["modelState"][key]
                    elif case == "extra": value["modelState"]["unexpected"] = torch.zeros(1)
                    elif case == "format": value["format"] = "previous-checkpoint"
                    elif case == "version": value["version"] = True
                    elif case == "steps": value["training"]["steps"] = 0
                    elif case == "hash": value["training"]["trainManifestSha256"] = "../../path"
                    elif case == "overlap": value["training"]["validationPayloadIds"] = value["training"]["trainPayloadIds"]
                    elif case == "missing-splits": del value["training"]["trainSessionIds"]
                    torch.save(value, path)
                    with self.assertRaises(ValueError):
                        load_checkpoint(path, model)

    def test_custom_pickle_values_and_noncheckpoint_files_are_rejected(self):
        model, value = self.checkpoint()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.pt"
            value["extra"] = CustomMetadata()
            torch.save(value, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model)
            path.write_text("not a checkpoint")
            with self.assertRaises(ValueError):
                load_checkpoint(path, model)

    def test_inherited_step_zero_baseline_loads_and_exports_exact_weights(self):
        model, value = self.checkpoint()
        value["training"].update({"steps": 0, "selectionStep": 0, "baselineSelection": True,
                                  "recipe": "adapt", "selectionAggregation": "group-balanced-real",
                                  "sessionGroupsSha256": "4"*64,
                                  "sessionGroupsPath": "/private/session-groups.json"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"baseline.pt"
            torch.save(value, path)
            loaded = load_checkpoint(path, OpticalDecoder())
            for name, expected in model.state_dict().items():
                torch.testing.assert_close(loaded["modelState"][name], expected, rtol=0, atol=0)
            manifest = export_smoke(Path(temporary)/"bundle", sizes=(64,), check_ort=False, checkpoint=path)
            public = manifest["training"]
            self.assertEqual(public["steps"], 0)
            self.assertEqual(public["selectionStep"], 0)
            self.assertIs(public["baselineSelection"], True)
            self.assertEqual(public["initialCheckpointSha256"], "3"*64)
            self.assertEqual(public["recipe"], "adapt")
            self.assertEqual(public["selectionAggregation"], "group-balanced-real")
            self.assertEqual(public["sessionGroupsSha256"], "4"*64)
            self.assertNotIn("sessionGroupsPath", public)
            self.assertNotIn("trainSessionIds", public)

    def test_step_zero_requires_explicit_inherited_baseline_provenance(self):
        _, value = self.checkpoint()
        baseline = {**value["training"], "steps": 0, "selectionStep": 0, "baselineSelection": True}
        changes = ({"baselineSelection": False}, {"baselineSelection": 1},
                   {"steps": 1}, {"selectionStep": 1}, {"steps": False},
                   {"selectionStep": False}, {"initialCheckpointSha256": "bad"},
                   {"recipe": "unknown"}, {"selectionAggregation": "unknown"},
                   {"sessionGroupsSha256": "bad"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                training_provenance({**baseline, **change})
        for missing in ("baselineSelection", "initialCheckpointSha256", "selectionStep"):
            malformed = {key: item for key, item in baseline.items() if key != missing}
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                training_provenance(malformed)
        positive = {**value["training"], "baselineSelection": False,
                    "recipe": "pilot", "selectionAggregation": "global"}
        public = training_provenance(positive)
        self.assertIs(public["baselineSelection"], False)
        self.assertEqual(public["recipe"], "pilot")


if __name__ == "__main__":
    unittest.main()
