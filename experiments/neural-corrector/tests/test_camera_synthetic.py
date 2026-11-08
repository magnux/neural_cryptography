"""Physical-unit invariants for camera-scale corruption and target pairing."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import synthetic
    from camera_geometry import CameraGeometryConfig, sample_camera_geometry
    from data import Dataset
    from synthetic import PhotometricConfig
    torch.set_num_threads(1)
    torch.backends.nnpack.set_flags(False)


def camera_sampling(side, camera_side, offset=(0., 0.)):
    return {"cameraSide": float(camera_side),
            "samplingSide": min(side, round(camera_side)),
            "samplingOffsetCamera": list(offset)}


def no_extra_photometric():
    return PhotometricConfig(white_balance_stops=0, exposure_stops=0,
        shade_probability=0, rolling_shutter_probability=0,
        motion_probability=0, glare_probability=0, rgb_offset_probability=0)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for camera-scale synthetic generation")
class CameraSyntheticTests(unittest.TestCase):
    def test_moire_wavelength_is_in_camera_pixels_without_factor_two_error(self):
        # Same physical image footprint gives the same number of band cycles,
        # regardless of how large the rectified training raster is stored.
        periods = []
        for grid in (16, 32, 64, 128):
            side, camera_side, period = 4*grid, 160., 20.
            phase = synthetic.moire_phase_field(side, camera_side, period, 0.)
            slope = phase[:, 1:]-phase[:, :-1]
            expected = 2*np.pi*camera_side/(period*side)
            torch.testing.assert_close(slope, torch.full_like(slope, expected),
                                       atol=8e-6, rtol=1e-5)
            torch.testing.assert_close(phase[1:], phase[:-1])
            periods.append(2*np.pi/float(slope.mean()))
        np.testing.assert_allclose(periods, [8, 16, 32, 64], rtol=2e-6)

    def test_sampling_phase_preserves_constant_and_linear_ramp_alignment(self):
        side = 128
        colors = torch.tensor([.2, .5, .8])[:, None, None].expand(3, side, side)
        axis = (torch.arange(side, dtype=torch.float32)+.5)/side
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        ramp = torch.stack((.2+.5*x, .2+.5*y, .2+.2*x+.3*y))
        for offset in ((0., 0.), (.37, -.42), (-.49, .49)):
            camera = camera_sampling(side, 32, offset)
            observed = synthetic.apply_camera_sampling(colors, camera,
                np.random.default_rng(3), 1., add_noise=False)
            torch.testing.assert_close(observed, colors, atol=2e-7, rtol=1e-6)
            observed = synthetic.apply_camera_sampling(ramp, camera,
                np.random.default_rng(3), 1., add_noise=False)
            # Border extension deliberately differs from a continued ramp;
            # interior labels must not inherit the sensor-lattice translation.
            torch.testing.assert_close(observed[:, 12:-12, 12:-12],
                ramp[:, 12:-12, 12:-12], atol=2e-7, rtol=1e-6)

    def test_noise_is_added_after_sensor_integration_and_averages_above_crop_resolution(self):
        side = 128
        constant = torch.full((3, side, side), .5)
        observed = synthetic.apply_camera_sampling(constant, camera_sampling(side, 32),
            np.random.default_rng(31), 1.)
        noise_std = float((observed-constant)[:, 8:-8, 8:-8].std())
        # 32x32 sensor noise is bilinearly reconstructed. Adding noise on the
        # 128x128 raster *before* area integration would divide it by about four.
        self.assertGreater(noise_std, .012)
        self.assertLess(noise_std, .024)
        ordinary = synthetic.apply_camera_sampling(constant, camera_sampling(side, 128),
            np.random.default_rng(31), 1.)
        oversampled = synthetic.apply_camera_sampling(constant, camera_sampling(side, 512),
            np.random.default_rng(31), 1.)
        torch.testing.assert_close(oversampled-constant, (ordinary-constant)/4,
                                   atol=5e-8, rtol=1e-4)

    def test_low_camera_footprint_mixes_cells_without_inventing_detail_at_large_footprint(self):
        grid, side = 32, 128
        cells = torch.from_numpy(np.random.default_rng(7).uniform(.1, .9,
            (3, grid, grid)).astype(np.float32))
        image = cells.repeat_interleave(4, -1).repeat_interleave(4, -2)
        low = synthetic.apply_camera_sampling(image, camera_sampling(side, 32, (.35, -.4)),
            np.random.default_rng(7), 1., add_noise=False)
        high = synthetic.apply_camera_sampling(image, camera_sampling(side, 512, (.35, -.4)),
            np.random.default_rng(7), 1., add_noise=False)
        torch.testing.assert_close(high, image, atol=0, rtol=0)
        self.assertGreater(float((low-image).square().mean()), .01)
        self.assertLess(float(low.var()), .6*float(image.var()))

    def test_large_gaussian_has_requested_width_and_preserves_constant(self):
        constant = torch.full((1, 3, 129, 129), .4)
        torch.testing.assert_close(synthetic.gaussian_blur(constant, 7.), constant,
                                   atol=2e-7, rtol=1e-6)
        impulse = torch.zeros((1, 1, 129, 129))
        impulse[0, 0, 64, 64] = 1
        blurred = synthetic.gaussian_blur(impulse, 7.)
        marginal = blurred[0, 0].sum(dim=0)
        axis = torch.arange(129, dtype=torch.float32)-64
        sigma = float((marginal*axis.square()).sum().sqrt())
        self.assertAlmostEqual(float(blurred.sum()), 1., places=6)
        self.assertGreater(sigma, 6.8)
        self.assertLess(sigma, 7.1)
        self.assertGreater(float(marginal[64+14]), 0.)

    def test_motion_integration_tracks_physical_travel_with_bounded_odd_sample_count(self):
        mild = dict(linearPixels=1., rotationDegrees=0., zoomFraction=0., samples=7)
        stronger = dict(mild, linearPixels=12.)
        mild_count = synthetic.camera_motion_samples(128, mild)
        strong_count = synthetic.camera_motion_samples(128, stronger)
        self.assertGreater(strong_count, mild_count)
        # Storing the same camera image at four times the raster dimensions
        # must not quadruple exposure samples or memory for an unchanged PSF.
        self.assertEqual(strong_count,
            synthetic.camera_motion_samples(512, dict(stronger, linearPixels=48.), 4.))
        radial = dict(mild, rotationDegrees=1., zoomFraction=.01)
        self.assertGreater(synthetic.camera_motion_samples(512, radial), mild_count)
        saturated = synthetic.camera_motion_samples(512,
            dict(radial, linearPixels=500., rotationDegrees=10., zoomFraction=.1))
        self.assertLessEqual(saturated, 63)
        self.assertEqual(saturated % 2, 1)

    def test_streamed_physical_trajectory_is_reproducible_and_continuous_across_phases(self):
        config = CameraGeometryConfig(footprint_min=128, footprint_max=128, easy_fraction=0)
        clean = np.random.default_rng(29).uniform(-.9, .9, (4, 3, 64, 64)).astype(np.float32)
        views = np.tile(np.arange(4), 2)
        options = dict(seed=7, severity=.6, pixels_per_cell=4,
            photometric=no_extra_photometric(), camera=sample_camera_geometry(16, 7, .6, config))
        with patch('synthetic.sample_photometric', wraps=synthetic.sample_photometric) as lighting, \
                patch('synthetic.rolling_shutter_maps', wraps=synthetic.rolling_shutter_maps) as shutter:
            first = np.stack(list(synthetic._iter_corrupt_sequence(clean, views, **options)))
        self.assertEqual(lighting.call_count, 1)
        self.assertEqual([call.args[2] for call in shutter.call_args_list], list(range(8)))
        second = np.stack(list(synthetic._iter_corrupt_sequence(clean, views, **options)))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[0], first[4]))
        for grid in (16, 32, 64, 128):
            with self.subTest(grid=grid):
                phases = np.zeros((4, 3, 4*grid, 4*grid), dtype=np.float32)
                iterator = synthetic._iter_corrupt_sequence(phases, np.arange(4), seed=7,
                    severity=.6, pixels_per_cell=4, photometric=no_extra_photometric(),
                    camera=sample_camera_geometry(grid, 7, .6, config))
                result = next(iterator)
                iterator.close()
                self.assertEqual(result.shape, (3, 4*grid, 4*grid))
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                self.assertLessEqual(float(np.abs(result).max()), 1.)

    def test_empirical_target_scale_reaches_every_generated_phase(self):
        # Reuse the canonical JS fixture: it renders clean phases from sender
        # symbols and uses the production baseline preparation, not a fake codec.
        sys.path.insert(0, str(Path(__file__).parent))
        from test_multires_synthetic import JOINED_FIXTURE
        base = ROOT/'validation/tmp'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            subprocess.run(['node', '--input-type=module', '-e', JOINED_FIXTURE, str(root), '16'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            manifest_path = root/'joined/manifest.json'
            joined = json.loads(manifest_path.read_text())
            joined['samples'][0]['provenance'] = {'stats': {'pixelsPerCell': 3.}}
            manifest_path.write_text(json.dumps(joined))
            generated = synthetic.generate([root/'clean'], root/'generated',
                paired_real=[root/'joined'], frames=2, recipe_name='w-tube', seed=17,
                severity=.5, severity_min=.5, photometric=no_extra_photometric(),
                camera_config=CameraGeometryConfig(easy_fraction=0))
            rows = generated['samples']
            self.assertEqual(len(rows), 8)
            self.assertEqual({row['info']['phase'] for row in rows}, {0, 1, 2, 3})
            geometries = [row['synthetic']['cameraGeometry'] for row in rows]
            self.assertTrue(all(geometry == geometries[0] for geometry in geometries))
            geometry = geometries[0]
            self.assertEqual(geometry['source'], 'paired-capture')
            self.assertEqual(geometry['observedPixelsPerCell'], 3.)
            self.assertGreaterEqual(geometry['cameraSide'], 48*.85)
            self.assertLessEqual(geometry['cameraSide'], 48*1.15)
            coverage = generated['generator']['cameraGeometry']['coverage']
            self.assertEqual(coverage['observedScaleCaptures'], 1)
            self.assertEqual(coverage['fallbackTargets'], 0)
            self.assertEqual([row['synthetic']['timelineFrame'] for row in rows], list(range(8)))
            for sample in Dataset([root/'generated']).samples:
                self.assertTrue(np.isfinite(sample.tensors()[0]).all())
                sample.read('baseline', '<f4')


if __name__ == '__main__':
    unittest.main()
