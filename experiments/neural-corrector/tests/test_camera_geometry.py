"""Physical footprint draws and recorded scale pools, independent of pixel truth."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_geometry import (CameraGeometryConfig, build_camera_pools,
                             camera_target_key, sample_camera_geometry)


def capture(session="phone", payload="a", split="train", grid=64, q=3.):
    return SimpleNamespace(metadata={"sessionId": session, "payloadId": payload,
        "split": split, "size": grid, "source": "real",
        "provenance": {"stats": {"pixelsPerCell": q}}})


class CameraGeometryTests(unittest.TestCase):
    def test_camera_footprint_does_not_grow_automatically_with_grid(self):
        config = CameraGeometryConfig(easy_fraction=0)
        for seed in range(20):
            small = sample_camera_geometry(16, seed, .8, config)
            big = sample_camera_geometry(128, seed, .8, config)
            self.assertEqual(small["cameraSide"], big["cameraSide"])
            self.assertEqual(small["registrationOffsetCamera"], big["registrationOffsetCamera"])
            self.assertEqual(small["blurSigmaCamera"], big["blurSigmaCamera"])
            self.assertAlmostEqual(small["cameraPixelsPerCell"]/8, big["cameraPixelsPerCell"])
            self.assertAlmostEqual(small["cameraToCrop"]*8, big["cameraToCrop"])
            self.assertEqual(small, sample_camera_geometry(16, seed, .8, config))
            self.assertLessEqual(small["samplingSide"], 64)
            self.assertLessEqual(big["samplingSide"], 512)
            json.dumps(big, allow_nan=False)

    def test_empirical_scale_is_order_independent_and_allows_subcell_sampling(self):
        config = CameraGeometryConfig(easy_fraction=0)
        for seed in range(10):
            a = sample_camera_geometry(128, seed, .8, config,
                                       observed_pixels_per_cell=[.4, .5, .6])
            b = sample_camera_geometry(128, seed, .8, config,
                                       observed_pixels_per_cell=[.6, .4, .5])
            self.assertEqual(a, b)
            self.assertEqual(a["source"], "paired-capture")
            self.assertIn(a["observedPixelsPerCell"], [.4, .5, .6])
            self.assertLess(a["cameraPixelsPerCell"], 1)
            self.assertLess(a["samplingSide"], 128)
            self.assertGreaterEqual(a["cameraSide"], .85*a["observedPixelsPerCell"]*128)
            self.assertLessEqual(a["cameraSide"], 1.15*a["observedPixelsPerCell"]*128)

    def test_easy_subset_and_footprint_clamps_are_explicit(self):
        normal = sample_camera_geometry(64, 7, .8,
            CameraGeometryConfig(footprint_min=128, footprint_max=128, easy_fraction=0))
        easy = sample_camera_geometry(64, 7, .8,
            CameraGeometryConfig(footprint_min=128, footprint_max=128, easy_fraction=1))
        self.assertFalse(normal["easy"])
        self.assertTrue(easy["easy"])
        self.assertEqual(easy["effectStrength"], .5)
        self.assertGreaterEqual(easy["cameraSide"], 128*1.25)
        self.assertLessEqual(easy["cameraSide"], 128*1.75)
        self.assertLessEqual(easy["blurSigmaCamera"], .85*.8*.5)
        for observed, bound in ((1e-6, 32), (1e6, 2048)):
            row = sample_camera_geometry(128, 3, .8, observed_pixels_per_cell=[observed])
            self.assertTrue(row["footprintClamped"])
            self.assertEqual(row["cameraSide"], bound)
            self.assertNotEqual(row["unclampedCameraSide"], row["cameraSide"])
            self.assertEqual(row["observedPixelsPerCell"], observed)

    def test_pools_keep_target_identity_and_report_missing_optional_measurements(self):
        rows = [capture(q=4), capture(q=2), capture(q=None), capture(q="bad"),
                capture(payload="b", q=7), capture(session="pc", payload="c", q=None)]
        before = copy.deepcopy([row.metadata for row in rows])
        pools, summary = build_camera_pools(rows)
        self.assertEqual(pools[("phone", "a", "train")], (2., 4.))
        self.assertEqual(pools[("phone", "b", "train")], (7.,))
        self.assertEqual(pools[("pc", "c", "train")], ())
        self.assertEqual(summary["captures"], 6)
        self.assertEqual(summary["observedScaleCaptures"], 3)
        self.assertEqual(summary["missingOrInvalidScaleCaptures"], 3)
        self.assertEqual(summary["observedScaleTargets"], 2)
        self.assertEqual(summary["fallbackTargets"], 1)
        self.assertEqual(build_camera_pools(reversed(rows)), (pools, summary))
        self.assertEqual([row.metadata for row in rows], before)
        self.assertEqual(camera_target_key(rows[0].metadata), ("phone", "a", "train"))

    def test_split_leakage_and_conflicting_target_geometry_are_rejected(self):
        for rows in ([capture(), capture(split="validation")],
                     [capture(), capture(session="other", split="validation")],
                     [capture(), capture(grid=128)]):
            with self.assertRaises(ValueError):
                build_camera_pools(rows)
        row = capture()
        row.metadata["source"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "real capture"):
            build_camera_pools([row])

    def test_zero_severity_identity_and_invalid_arguments(self):
        row = sample_camera_geometry(64, 7, 0, observed_pixels_per_cell=[.2])
        self.assertEqual(row["cameraSide"], 256)
        self.assertEqual(row["samplingSide"], 256)
        self.assertEqual(row["blurSigmaCamera"], 0)
        self.assertEqual(row["samplingOffsetCamera"], [0., 0.])
        self.assertEqual(row["registrationOffsetCamera"], [0., 0.])
        self.assertEqual(row["affineCamera"], [0.]*4)
        for kwargs in ({"footprint_min": 0}, {"footprint_min": 700, "footprint_max": 640},
                       {"easy_fraction": 1.1}, {"easy_fraction": True}, {"blur_sigma_min": 1., "blur_sigma_max": .5},
                       {"registration_pixels": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CameraGeometryConfig(**kwargs)
        for q in ([float("nan")], [float("inf")], [True], [-1], [0]):
            with self.assertRaises(ValueError):
                sample_camera_geometry(64, 7, .8, observed_pixels_per_cell=q)
        for args in ((8, 7, .8), (64, -1, .8), (64, 7, float("nan")), (64, 7, True)):
            with self.assertRaises(ValueError):
                sample_camera_geometry(*args)


if __name__ == "__main__":
    unittest.main()
