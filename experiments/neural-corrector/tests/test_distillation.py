"""Early-U distillation preserves deployed weights and burst alignment."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distillation import (EMADistillation, StudentAugmentation,
                          distillation_weight, normalized_feature_loss)
from model import NeuralCorrector, W_TUBE_SIZES, parameter_count
from test_wtube_model import packed


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)

    def test_feature_taps_preserve_inference_and_checkpoint_state(self):
        for profile, parameters in (("W-tube", 487760), ("VW", 696400)):
            torch.manual_seed(43)
            model = NeuralCorrector(profile).eval()
            keys = tuple(model.state_dict())
            for size in W_TUBE_SIZES:
                with self.subTest(profile=profile, grid=size), torch.no_grad():
                    pixels = packed(size, counts=5)
                    reference = model(pixels)
                    logits, first, last = model.forward_with_features(pixels)
                    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
                    self.assertEqual(tuple(first.shape), (1, 32, 4*size, 4*size))
                    self.assertEqual(tuple(last.shape), tuple(first.shape))
                    torch.testing.assert_close(model.final_stage_features(pixels), last, rtol=0, atol=0)
            self.assertEqual(tuple(model.state_dict()), keys)
            self.assertEqual(parameter_count(model), parameters)
        with self.assertRaisesRegex(ValueError, "W-tube or VW"):
            NeuralCorrector("multires").forward_with_features(torch.empty(1, 32, 32, 32))

    def test_teacher_target_skips_outlet_and_auxiliary_gradient_reaches_only_early_path(self):
        torch.manual_seed(4)
        student = NeuralCorrector("VW")
        distiller = EMADistillation(student)
        distiller.train()
        self.assertFalse(distiller.teacher.training)
        self.assertTrue(distiller.projector.training)
        self.assertFalse(any(parameter.requires_grad for parameter in distiller.teacher.parameters()))
        hook = distiller.teacher.outlet.register_forward_hook(
            lambda *args: self.fail("Teacher must not compute the classification outlet"))
        try:
            weak = packed(16, counts=5)
            target = distiller.target(weak)
        finally:
            hook.remove()
        self.assertFalse(target.requires_grad)
        strong = StudentAugmentation(seed=44)(weak)
        logits, first, _ = student.forward_with_features(strong)
        mask = torch.ones((1, 16, 16), dtype=torch.bool)
        auxiliary = distiller.loss(first, target, mask)
        self.assertTrue(torch.isfinite(auxiliary))
        auxiliary.backward()
        for parameter in (student.inlet[0][0].weight, student.stages[0].e0.pairs[0].conv1.weight,
                          distiller.projector.pairs[0].conv1.weight,
                          distiller.projector.pairs[0].conv2.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in distiller.teacher.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in student.stages[1:].parameters()))
        self.assertIsNone(student.head.weight.grad)
        self.assertEqual(tuple(logits.shape), (1, 16, 16, 16))

    def test_ema_updates_parameters_and_copies_derived_buffers_once(self):
        torch.manual_seed(7)
        student = NeuralCorrector("W-tube")
        distiller = EMADistillation(student, decay=.75)
        parameter = student.inlet[0][0].weight
        before = distiller.teacher.inlet[0][0].weight.detach().clone()
        with torch.no_grad():
            parameter.add_(2.)
            student.anchor_basis_16.add_(.125)
        # Mere student changes/forwards must not implicitly update EMA.
        torch.testing.assert_close(distiller.teacher.inlet[0][0].weight, before, rtol=0, atol=0)
        distiller.update(student)
        torch.testing.assert_close(distiller.teacher.inlet[0][0].weight, before+.5)
        torch.testing.assert_close(distiller.teacher.anchor_basis_16, student.anchor_basis_16, rtol=0, atol=0)
        self.assertFalse(distiller.teacher.training)
        self.assertTrue(all(parameter.grad is None for parameter in distiller.teacher.parameters()))
        with self.assertRaisesRegex(ValueError, "profiles must match"):
            distiller.update(NeuralCorrector("VW"))

    def test_feature_distance_uses_data_mask_unit_vectors_and_detached_targets(self):
        torch.manual_seed(8)
        prediction = torch.randn(2, 32, 8, 8, requires_grad=True)
        target = torch.randn(2, 32, 8, 8, requires_grad=True)
        mask = torch.tensor([[[True, False], [False, False]],
                             [[True, True], [True, True]]])
        result = normalized_feature_loss(prediction, target, mask)
        separate = torch.stack([normalized_feature_loss(prediction[i:i+1], target[i:i+1], mask[i:i+1])
                                for i in range(2)]).mean()
        torch.testing.assert_close(result, separate)
        torch.testing.assert_close(result, normalized_feature_loss(prediction*3, target*7, mask))
        result.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(prediction.grad[0, :, 4:, :].abs().sum().item(), 0.)
        self.assertEqual(prediction.grad[0, :, :, 4:].abs().sum().item(), 0.)
        self.assertGreater(prediction.grad[0, :, :4, :4].abs().sum().item(), 0.)
        x, y = torch.zeros(1, 32, 4, 4), torch.zeros(1, 32, 4, 4)
        x[:, 0], y[:, 1] = 1., 1.
        self.assertAlmostEqual(normalized_feature_loss(x, y, torch.ones(1, 1, 1, dtype=torch.bool)).item(), 2.)
        with self.assertRaisesRegex(ValueError, "nonempty boolean"):
            normalized_feature_loss(prediction.detach(), target.detach(), torch.zeros_like(mask))

    def test_extra_student_view_preserves_phases_slots_presence_and_rng_independence(self):
        weak = packed(16, counts=4)
        for slot in (1, 3, 5, 7):
            weak[:, 3*slot:3*slot+3] = 0
        original = weak.clone()
        first, second = StudentAugmentation(seed=13), StudentAugmentation(seed=13)
        torch.manual_seed(91)
        rng_state = torch.random.get_rng_state().clone()
        strong = first(weak)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        torch.testing.assert_close(weak, original, rtol=0, atol=0)
        torch.testing.assert_close(strong, second(weak), rtol=0, atol=0)
        torch.testing.assert_close(strong[:, 24:], weak[:, 24:], rtol=0, atol=0)
        for slot in range(8):
            rgb = strong[:, 3*slot:3*slot+3]
            if slot % 2:
                self.assertEqual(rgb.abs().sum().item(), 0.)
            else:
                self.assertGreater((rgb-weak[:, 3*slot:3*slot+3]).abs().sum().item(), 0.)
        self.assertGreaterEqual(strong[:, :24].min().item(), -1.)
        self.assertLessEqual(strong[:, :24].max().item(), 1.)
        self.assertFalse(torch.equal(strong, first(weak)))
        no_change = StudentAugmentation(seed=1, noise_std=0, tint_stops=0, exposure_stops=0)(weak)
        torch.testing.assert_close(no_change, weak, rtol=0, atol=0)

    def test_invalid_configuration_and_optimizer_step_ramp(self):
        self.assertEqual(distillation_weight(0), 0.)
        self.assertEqual(distillation_weight(500), .025)
        self.assertEqual(distillation_weight(1000), .05)
        self.assertEqual(distillation_weight(2000), .05)
        self.assertEqual(distillation_weight(0, ramp_updates=0), .05)
        for kwargs in ({"update": -1}, {"update": 1, "maximum": float("nan")},
                       {"update": 1, "ramp_updates": -1}):
            with self.assertRaises(ValueError):
                distillation_weight(**kwargs)
        for decay in (-1, 1, float("inf")):
            with self.assertRaisesRegex(ValueError, "EMA decay"):
                EMADistillation(NeuralCorrector("W-tube"), decay=decay)
        with self.assertRaisesRegex(ValueError, "W-tube or VW"):
            EMADistillation(NeuralCorrector("U-tube"))
        with self.assertRaisesRegex(ValueError, "noise_std"):
            StudentAugmentation(seed=1, noise_std=-1.)


if __name__ == "__main__":
    unittest.main()
