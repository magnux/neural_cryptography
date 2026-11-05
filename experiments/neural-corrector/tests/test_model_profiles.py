"""Current full-resolution profiles preserve spatial evidence and checkpoint identity."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from model import MODEL_PROFILES, DEFAULT_MODEL_PROFILE, NeuralCorrector, processing_parameter_count, profile_from_state
    from checkpoint import load_model_checkpoint, validate_split_provenance, training_provenance
    torch.set_num_threads(1)


def training_metadata(model=None):
    result = {"steps": 1, "seed": 7, "trainSessionIds": ["training"], "trainPayloadIds": ["train-payload"],
            "validationSessionIds": ["validation"], "validationPayloadIds": ["validation-payload"]}
    if model is not None and model.variable_frames:
        result.update(framesPerInput=8, inputLayout=model.input_layout, minFramesPerInput=4,
                      frameCounts=[4, 5, 6, 7, 8], frameCountPolicy="uniform-4-8",
                      outputLabelSpace="logical", phaseMasks=[0, 15, 7, 8])
    if model is not None and model.profile == "multires":
        result.update(modelProfile="multires", datasetFormat="neural-qr-dataset",
                      pixelsPerCell=2, supportedSizes=list(model.supported_sizes),
                      trainedSizes=list(model.supported_sizes))
    return result


def model_input(model, size=64, batch=1):
    side = model.pixels_per_cell*size
    result = torch.randn(batch, model.input_channels, side, side).clamp(-1.5, 1.5)
    if model.variable_frames:
        result[:, 24:] = 1
    return result


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for model profile tests")
class ProfileTests(unittest.TestCase):
    def test_profiles_output_contract_gradients_and_batch_independence(self):
        torch.manual_seed(7)
        for profile in MODEL_PROFILES:
            with self.subTest(profile=profile):
                model = NeuralCorrector(profile)
                for size in model.supported_sizes:
                    pixels = model_input(model, size)
                    with torch.inference_mode():
                        output = model(pixels)
                    self.assertEqual(tuple(output.shape), (1, 16, size, size))
                    self.assertTrue(torch.isfinite(output).all())
                model.zero_grad()
                pixels = model_input(model)
                logits = model(pixels)
                torch.nn.functional.cross_entropy(logits, torch.randint(16, (1, 64, 64))).backward()
                self.assertGreater(float(model.inlet[0][0].weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
                model.eval()
                with torch.inference_mode():
                    single = model(pixels)
                    batch = model(torch.cat((pixels, model_input(model)), dim=0))[:1]
                torch.testing.assert_close(single, batch, rtol=1e-4, atol=1e-5)
                self.assertLess(processing_parameter_count(model), 500000)
        self.assertEqual(DEFAULT_MODEL_PROFILE, "U-tube")
        self.assertEqual(NeuralCorrector().profile, DEFAULT_MODEL_PROFILE)

    def test_residual_pairs_keep_identity_projection_and_trainable_branches(self):
        torch.manual_seed(29)
        model = NeuralCorrector("U-tube-2x")
        for stage, channels in (("e0", 32), ("e1", 32), ("d1", 64)):
            with self.subTest(stage=stage):
                pair = getattr(model, stage).pairs[0]
                pixels = torch.randn(2, channels, 16, 16, requires_grad=True)
                output = pair(pixels)
                (output * torch.randn_like(output)).sum().backward()
                self.assertGreater(float(pair.conv1.weight.grad.abs().sum()), 0)
                self.assertGreater(float(pair.conv2.weight.grad.abs().sum()), 0)
                self.assertGreater(float(pixels.grad.abs().sum()), 0)
                if isinstance(pair.shortcut, torch.nn.Conv2d):
                    self.assertGreater(float(pair.shortcut.weight.grad.abs().sum()), 0)
                with torch.no_grad():
                    pair.conv2.weight.zero_()
                    pair.conv2.bias.zero_()
                    expected = torch.relu(pair.shortcut(pixels))
                    torch.testing.assert_close(pair(pixels), expected, rtol=0, atol=0)

    def test_only_current_full_resolution_profiles_are_constructible(self):
        self.assertEqual(MODEL_PROFILES, ("U-tube", "U-tube-2x", "U-tube-2x-attn", "multires"))
        for obsolete in ("compact", "wide", "normalized-compact", "unknown"):
            with self.subTest(profile=obsolete), self.assertRaises(ValueError):
                NeuralCorrector(obsolete)
        state = NeuralCorrector().state_dict()
        # No outlet is the signature of an unsupported spatial computation.
        with self.assertRaises(ValueError):
            profile_from_state({name: value for name, value in state.items()
                                if not name.startswith("outlet.")})

    def test_same_checkpoint_format_selects_exact_profile_without_parameter_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            for profile in MODEL_PROFILES:
                original = NeuralCorrector(profile)
                value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
                         "modelState": original.state_dict(), "training": training_metadata(original)}
                torch.save(value, path)
                loaded, checkpoint = load_model_checkpoint(path)
                self.assertEqual(loaded.profile, profile)
                self.assertEqual(checkpoint["version"], 1)
                for name, tensor in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
                value["modelState"] = {name: tensor for name, tensor in original.state_dict().items()
                                       if not name.startswith("outlet.")}
                torch.save(value, path)
                with self.assertRaises(ValueError):
                    load_model_checkpoint(path)
                # Identity inference is followed by strict full-state checking;
                # an extra RGB bypass must never become accepted model weights.
                value["modelState"] = {**original.state_dict(),
                                       "local_color.0.weight": torch.zeros(16, 3, 1, 1)}
                torch.save(value, path)
                with self.assertRaises(ValueError):
                    load_model_checkpoint(path)
                value["modelState"] = original.state_dict()
                value["training"]["modelProfile"] = "U-tube-2x" if profile == "U-tube" else "U-tube"
                torch.save(value, path)
                with self.assertRaisesRegex(ValueError, "profile metadata disagrees"):
                    load_model_checkpoint(path)
                value["training"]["modelProfile"] = profile
                value["modelState"]["head.weight"] = torch.zeros(16, 17, 1, 1)
                torch.save(value, path)
                with self.assertRaises(ValueError):
                    load_model_checkpoint(path)

    def test_large_dataset_provenance_is_not_capped_at_one_mib_or_100k_identities(self):
        metadata = training_metadata()
        metadata["trainPayloadIds"] = [f"payload-{i:064x}" for i in range(100001)]
        public = training_provenance(metadata)
        validate_split_provenance(metadata)
        self.assertNotIn("trainPayloadIds", public)
        self.assertEqual(public["steps"], 1)


if __name__ == "__main__":
    unittest.main()
