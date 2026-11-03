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
    from model import MODEL_PROFILES, OpticalDecoder
    from export import export_smoke, load_checkpoint
    from checkpoint import training_provenance, load_model_checkpoint
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

    def test_five_frame_checkpoint_and_export_preserve_exact_profile(self):
        import onnx
        for profile in ("normalized-tube", "longtube"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                _, value = self.checkpoint()
                model = OpticalDecoder(profile, frames=5).eval()
                value["modelState"] = model.state_dict()
                value["training"].update(framesPerInput=5, modelProfile=profile, networkInputClip=[-1., 1.])
                path = Path(temporary)/"burst.pt"
                torch.save(value, path)
                loaded, _ = load_model_checkpoint(path)
                self.assertEqual((loaded.profile, loaded.frames), (profile, 5))
                for name, tensor in loaded.state_dict().items():
                    torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
                manifest = export_smoke(Path(temporary)/"bundle", sizes=(64,), check_ort=False, checkpoint=path)
                self.assertEqual(manifest["framesPerInput"], 5)
                self.assertEqual(manifest["training"]["framesPerInput"], 5)
                self.assertEqual(manifest["modelProfile"], profile)
                self.assertEqual(manifest["training"]["modelProfile"], profile)
                self.assertEqual(manifest["training"]["networkInputClip"], [-1, 1])
                self.assertEqual(manifest["normalization"]["clip"], [-1.5, 1.5])
                self.assertEqual(manifest["networkInputClip"], [-1, 1])
                case = manifest["cases"][0]
                graph = onnx.load(Path(temporary)/"bundle"/case["model"]["file"])
                self.assertIn("Clip", {node.op_type for node in graph.graph.node})
                pixels = np.fromfile(Path(temporary)/"bundle"/case["input"]["file"], dtype="<f4")
                self.assertGreater(float(np.abs(pixels).max()), 1)
                with self.assertRaisesRegex(ValueError, "frame count differs"):
                    export_smoke(Path(temporary)/"bad", sizes=(64,), check_ort=False, checkpoint=path, frames=1)
                other = next(name for name in MODEL_PROFILES if name != profile)
                with self.assertRaises(ValueError):
                    load_checkpoint(path, OpticalDecoder(other, frames=5))
                with self.assertRaisesRegex(ValueError, "profile differs"):
                    export_smoke(Path(temporary)/"wrong-profile", sizes=(64,), check_ort=False,
                                 checkpoint=path, profile=other)
                value["training"]["modelProfile"] = other
                torch.save(value, path)
                with self.assertRaisesRegex(ValueError, "profile metadata disagrees"):
                    load_model_checkpoint(path)
                value["training"]["modelProfile"] = profile
                value["training"]["framesPerInput"] = 1
                torch.save(value, path)
                with self.assertRaisesRegex(ValueError, "frame metadata disagrees"):
                    load_model_checkpoint(path)

    def test_attention_checkpoint_exports_anchor_and_presence_contract_without_adaptation(self):
        model = OpticalDecoder("attention-tube").eval()
        _, value = self.checkpoint()
        with torch.no_grad():
            model.position_anchor.add_(.03)
            model.head.bias.add_(.25)
        value["modelState"] = model.state_dict()
        value["training"].update(
            framesPerInput=8, modelProfile="attention-tube", inputLayout="phase-paired-rgb-presence-planes",
            minFramesPerInput=4, frameCounts=[4, 5, 6, 7, 8], frameCountPolicy="uniform-4-8",
            outputLabelSpace="logical", phaseMasks=[0, 15, 7, 8])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"attention.pt"
            output = Path(temporary)/"bundle"
            torch.save(value, path)
            loaded, _ = load_model_checkpoint(path)
            self.assertEqual(loaded.profile, "attention-tube")
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
            manifest = export_smoke(output, check_ort=False, checkpoint=path)
            self.assertEqual(manifest["inputLayout"], "phase-paired-rgb-presence-planes")
            self.assertEqual(manifest["outputLabelSpace"], "logical")
            self.assertEqual(manifest["phaseMasks"], [0, 15, 7, 8])
            self.assertEqual(manifest["frameCounts"], [4, 5, 6, 7, 8])
            self.assertEqual(manifest["parameterCount"], 432848)
            self.assertEqual(manifest["training"]["frameCountPolicy"], "uniform-4-8")
            self.assertEqual([case["size"] for case in manifest["cases"]], [64])
            case = manifest["cases"][0]
            pixels = np.fromfile(output/case["input"]["file"], dtype="<f4").reshape(case["inputShape"])
            exported = np.fromfile(output/case["expected"]["file"], dtype="<f4").reshape(case["outputShape"])
            with torch.inference_mode():
                expected = model(torch.from_numpy(pixels)).numpy()
            np.testing.assert_array_equal(exported, expected)
            with self.assertRaisesRegex(ValueError, "frame count differs"):
                export_smoke(output, check_ort=False, checkpoint=path, frames=5)
            for field in ("inputLayout", "minFramesPerInput", "frameCounts", "frameCountPolicy", "outputLabelSpace", "phaseMasks"):
                broken = {**value, "training": dict(value["training"])}
                del broken["training"][field]
                torch.save(broken, path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_model_checkpoint(path)

    def test_normalized_profile_rejects_missing_normalization_and_rgb_branch_mixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"malformed.pt"
            for defect in ("missing-inlet-norm", "missing-backbone-norm", "rgb-branch", "wrong-stem"):
                with self.subTest(defect=defect):
                    _, value = self.checkpoint()
                    state = OpticalDecoder("normalized-tube", frames=5).state_dict()
                    if defect == "missing-inlet-norm":
                        del state["inlet.0.1.weight"]
                    elif defect == "missing-backbone-norm":
                        del state["e0.0.1.weight"]
                    elif defect == "rgb-branch":
                        state["local_color.0.weight"] = torch.zeros(16, 15, 1, 1)
                    else:
                        state["inlet.0.0.weight"] = torch.zeros(8, 15, 3, 3)
                    value["modelState"] = state
                    value["training"]["framesPerInput"] = 5
                    torch.save(value, path)
                    with self.assertRaises(ValueError):
                        load_model_checkpoint(path)

    def test_sixty_four_frame_checkpoint_loads_without_shape_adaptation(self):
        model = OpticalDecoder("longtube", frames=64).eval()
        _, value = self.checkpoint()
        value["modelState"] = model.state_dict()
        value["training"]["framesPerInput"] = 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"burst.pt"
            torch.save(value, path)
            loaded, _ = load_model_checkpoint(path)
            self.assertEqual(loaded.frames, 64)
            for name, tensor in loaded.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint tensor"):
                load_checkpoint(path, OpticalDecoder("longtube", frames=32))

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
                                  "pairedTargets": True, "selectionAggregation": "group-balanced-real",
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
            self.assertIs(public["pairedTargets"], True)
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
                   {"pairedTargets": 1}, {"framesPerInput": 0}, {"framesPerInput": 65}, {"framesPerInput": True},
                   {"modelProfile": "unknown"}, {"networkInputClip": [-1.5, 1.5]},
                   {"networkInputClip": [-1, True]}, {"networkInputClip": "-1,1"},
                   {"selectionAggregation": "unknown"},
                   {"sessionGroupsSha256": "bad"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                training_provenance({**baseline, **change})
        for missing in ("baselineSelection", "initialCheckpointSha256", "selectionStep"):
            malformed = {key: item for key, item in baseline.items() if key != missing}
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                training_provenance(malformed)
        positive = {**value["training"], "baselineSelection": False,
                    "pairedTargets": False, "selectionAggregation": "global"}
        public = training_provenance(positive)
        self.assertIs(public["baselineSelection"], False)
        self.assertIs(public["pairedTargets"], False)


if __name__ == "__main__":
    unittest.main()
