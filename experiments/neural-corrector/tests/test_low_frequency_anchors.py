"""Independent positional priors have a hard frequency cap and stable units."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from model import (ANCHOR_MAX_MODES, MULTIRES_SIZES, NeuralCorrector,
                       dct_anchor_basis, profile_from_state, project_anchor)
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for cosine anchor checks")
class LowFrequencyAnchorTests(unittest.TestCase):
    def test_orthogonal_basis_and_spatial_energy_have_consistent_units(self):
        torch.manual_seed(7)
        model = NeuralCorrector("multires")
        for grid in MULTIRES_SIZES:
            with self.subTest(grid=grid):
                basis = dct_anchor_basis(grid)
                modes = min(grid, ANCHOR_MAX_MODES)
                torch.testing.assert_close(basis.double().T @ basis.double(),
                                           torch.eye(modes, dtype=torch.float64), rtol=0, atol=2e-7)
                coefficients = model.anchor_coefficients[str(grid)]
                anchor = model.spatial_anchor(grid)
                self.assertEqual(tuple(anchor.shape), (1, 32, grid, grid))
                torch.testing.assert_close(anchor.square().mean(), coefficients.square().mean(),
                                           rtol=2e-6, atol=1e-10)
                rms = float(anchor.detach().square().mean().sqrt())
                self.assertGreater(rms, .018)
                self.assertLess(rms, .022)
                # Bases are deterministic geometry, not serialized trainable state.
                self.assertFalse(any(name.startswith("anchor_basis_") for name in model.state_dict()))

    def test_projection_preserves_low_modes_and_removes_high_modes(self):
        grid = 64
        positions = torch.arange(grid, dtype=torch.float64) + .5
        low = (torch.cos(torch.pi * positions * 3/grid)[:, None]
               * torch.cos(torch.pi * positions * 5/grid)[None, :])
        high = (torch.cos(torch.pi * positions * 23/grid)[:, None]
                * torch.cos(torch.pi * positions * 4/grid)[None, :])
        source = (low + 2*high)[None, None].expand(1, 32, grid, grid).to(torch.float32)
        expected = low[None, None].expand_as(source).to(torch.float32)
        model = NeuralCorrector("multires")
        with torch.no_grad():
            model.anchor_coefficients[str(grid)].copy_(project_anchor(source))
        torch.testing.assert_close(model.spatial_anchor(grid), expected, rtol=2e-5, atol=3e-6)
        # G8 retains all its modes: projection is an exact change of coordinates.
        small = torch.randn(1, 32, 8, 8)
        with torch.no_grad():
            model.anchor_coefficients["8"].copy_(project_anchor(small))
        torch.testing.assert_close(model.spatial_anchor(8), small, rtol=1e-5, atol=1e-6)

    def test_constant_mode_and_gradient_touch_only_selected_size(self):
        model = NeuralCorrector("multires")
        with torch.no_grad():
            model.anchor_coefficients["128"].zero_()
            model.anchor_coefficients["128"][:, :, 0, 0] = .3*16
        anchor = model.spatial_anchor(128)
        torch.testing.assert_close(anchor, torch.full_like(anchor, .3), rtol=1e-6, atol=1e-7)
        anchor.square().mean().backward()
        self.assertGreater(float(model.anchor_coefficients["128"].grad.abs().sum()), 0)
        for grid in MULTIRES_SIZES:
            if grid != 128:
                self.assertIsNone(model.anchor_coefficients[str(grid)].grad)

    def test_projection_contract_and_obsolete_dense_state_are_rejected(self):
        for invalid in (torch.zeros(1, 32, 12, 12), torch.zeros(1, 3, 64, 64),
                        torch.zeros(1, 32, 32, 64), torch.zeros(1, 32, 64, 64, dtype=torch.int64),
                        torch.full((1, 32, 64, 64), float("nan"))):
            with self.subTest(shape=tuple(invalid.shape)), self.assertRaises(ValueError):
                project_anchor(invalid)
        model = NeuralCorrector("multires")
        state = {name: value for name, value in model.state_dict().items()
                 if not name.startswith(("anchor_coefficients.", "bottleneck_third.", "attention_third."))}
        state.update({f"anchors.{grid}": torch.zeros(1, 32, grid, grid) for grid in MULTIRES_SIZES})
        with self.assertRaisesRegex(ValueError, "low-frequency anchors.*six bottleneck blocks"):
            profile_from_state(state)


if __name__ == "__main__":
    unittest.main()
