"""Canonical target reconstruction through independent corruption and pairing."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from data import Dataset, assert_disjoint
    from sampling import BalancedSampler
    from synthetic import CaptureResolutionConfig, PhotometricConfig, generate
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for synthetic generation")
class TargetSyntheticTests(unittest.TestCase):
    def test_grouped_targets_share_conditions_and_reject_incomplete_or_wrong_truth(self):
        from data import digest
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            subprocess.run(["node", str(ROOT/"scripts/corrector-dataset.mjs"), "generate",
                            "--out", str(root/"clean"), "--count", "1", "--phases", "4",
                            "--size", "64", "--colors", "16", "--seed", "913"],
                           check=True, capture_output=True, text=True)
            path = root/"clean/manifest.json"
            original = path.read_text()
            config = dict(frames=2, bursts_per_target=2, seed=71, severity=.3,
                          phase_grouped=True, severity_min=.2,
                          resolution=CaptureResolutionConfig(probability=1, min_scale=.25, max_scale=.5))
            first = generate([root/"clean"], root/"corrupted", **config)
            repeated = generate([root/"clean"], root/"repeat", **config)
            self.assertEqual(first["samples"], repeated["samples"])
            self.assertTrue(first["generator"]["phaseGrouped"])
            self.assertEqual(first["generator"]["phaseSchedule"], "round-robin-continuous")
            self.assertEqual(len(first["samples"]), 16)
            by_group = {}
            for row in first["samples"]:
                by_group.setdefault(row["synthetic"]["phaseGroupId"], []).append(row)
            self.assertEqual(len(by_group), 2)
            for rows in by_group.values():
                self.assertEqual(sorted(row["synthetic"]["timelineFrame"] for row in rows), list(range(8)))
                self.assertEqual({row["info"]["phase"] for row in rows}, set(range(4)))
                self.assertEqual(len({row["burstId"] for row in rows}), 4)
                self.assertEqual(len({row["synthetic"]["seed"] for row in rows}), 1)
                self.assertEqual(len({row["synthetic"]["severity"] for row in rows}), 1)
                self.assertEqual(len({json.dumps(row["synthetic"]["captureResolution"], sort_keys=True)
                                      for row in rows}), 1)
                self.assertEqual(len({row["hashes"]["input"] for row in rows}), 8)
            # Missing phases must fail before an output directory is created.
            broken = json.loads(original)
            broken["samples"] = [row for row in broken["samples"] if row["info"]["phase"] != 3]
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "complete set"):
                generate([root/"clean"], root/"missing", **config)
            self.assertFalse((root/"missing").exists())
            # Even consistent file hashes cannot disguise incorrect cross-phase truth.
            broken = json.loads(original)
            row = next(row for row in broken["samples"] if row["info"]["phase"] == 1)
            labels_path = root/"clean"/row["labels"]
            labels = np.frombuffer(labels_path.read_bytes(), dtype=np.uint8).copy()
            mask = np.frombuffer((root/"clean"/row["mask"]).read_bytes(), dtype=np.uint8)
            labels[np.flatnonzero(mask)[0]] ^= 1
            labels_path.write_bytes(labels.tobytes())
            row["hashes"]["labels"] = digest(labels_path)
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "targets disagree"):
                generate([root/"clean"], root/"wrong-truth", **config)
            self.assertFalse((root/"wrong-truth").exists())

    def test_targets_independent_bursts_pairing_and_split_ownership(self):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)

            def node(*args):
                subprocess.run(["node", str(ROOT/"scripts/corrector-dataset.mjs"), *map(str, args)],
                               check=True, capture_output=True, text=True)

            node("generate", "--out", root/"real", "--count", 1, "--phases", 2,
                 "--size", 64, "--colors", 16, "--seed", 903)
            path = root/"real/manifest.json"
            fixture = json.loads(path.read_text())
            for row in fixture["samples"]:
                # Canonical fixtures exercise the joined-real metadata path;
                # this test makes no camera-quality claim.
                row["source"] = "real"
                row["senderSessionId"] = row["sessionId"]
                row["sessionId"] = "target-fixture-receiver"
            path.write_text(json.dumps(fixture))
            original = path.read_bytes()
            node("targets", "--dataset", root/"real", "--out", root/"clean")
            real, clean = Dataset([root/"real"]), Dataset([root/"clean"])
            resolution = CaptureResolutionConfig(probability=1, min_scale=.25, max_scale=.25)
            photometric = PhotometricConfig(motion_probability=1, rgb_offset_probability=1)
            first = generate([root/"clean"], root/"corrupted", frames=2,
                             bursts_per_target=2, seed=71, severity=.4, severity_min=.25,
                             resolution=resolution, photometric=photometric)
            repeated = generate([root/"clean"], root/"repeat", frames=2,
                                bursts_per_target=2, seed=71, severity=.4, severity_min=.25,
                                resolution=resolution, photometric=photometric)
            self.assertEqual(len(first["samples"]), 8)
            self.assertEqual(first["samples"], repeated["samples"])
            self.assertEqual(first["generator"]["burstsPerTarget"], 2)
            self.assertEqual(first["generator"]["severityMin"], .25)
            self.assertEqual(first["generator"]["severity"], .4)
            self.assertEqual(first["generator"]["kind"], "correlated-calibrated-crop-v8")
            self.assertEqual(first["generator"]["photometric"]["glare_probability"], .65)
            self.assertEqual(first["generator"]["photometric"]["rgb_offset_probability"], 1)
            self.assertEqual(first["generator"]["photometric"]["rgb_offset_pixels"], .5)
            self.assertEqual(first["generator"]["captureResolution"],
                {"probability": 1, "min_scale": .25, "max_scale": .25, "blur_sigma": .35})
            by_target = {sample.id: [] for sample in clean.samples}
            for row in first["samples"]:
                by_target[row["synthetic"]["parentId"]].append(row)
                self.assertEqual(row["sessionId"], "target-fixture-receiver")
                self.assertEqual(row["split"], "train")
                self.assertGreaterEqual(row["synthetic"]["severity"], .25)
                self.assertLessEqual(row["synthetic"]["severity"], .4)
                self.assertEqual(row["synthetic"]["captureResolution"]["side"], 64)
            for rows in by_target.values():
                self.assertEqual(len({row["burstId"] for row in rows}), 2)
                self.assertEqual(len({row["synthetic"]["seed"] for row in rows}), 2)
                self.assertEqual(len({row["hashes"]["input"] for row in rows}), 4)
                self.assertEqual(len({row["hashes"]["labels"] for row in rows}), 1)
                self.assertEqual(len({row["synthetic"]["severity"] for row in rows}), 2)
                for burst_id in {row["burstId"] for row in rows}:
                    self.assertEqual(len({row["synthetic"]["severity"] for row in rows
                                          if row["burstId"] == burst_id}), 1)
                    self.assertEqual(len({json.dumps(row["synthetic"]["captureResolution"], sort_keys=True)
                                          for row in rows if row["burstId"] == burst_id}), 1)
                target = next(sample for sample in clean.samples if sample.id == rows[0]["synthetic"]["parentId"])
                for row in rows:
                    for field in ("labels", "mask", "expected"):
                        self.assertEqual(row["hashes"][field], target.metadata["hashes"][field])
            fixed = generate([root/"clean"], root/"fixed", frames=1,
                             bursts_per_target=1, seed=71, severity=.4)
            self.assertIsNone(fixed["generator"]["severityMin"])
            self.assertTrue(all(row["synthetic"]["severity"] == .4 for row in fixed["samples"]))
            self.assertTrue({row["id"] for row in fixed["samples"]}.isdisjoint(row["id"] for row in first["samples"]))
            # Strength selection is stable under changed glare settings; its RNG
            # does not belong to the glare trajectory or main sensor-noise stream.
            no_glare = generate([root/"clean"], root/"no-glare", frames=2, bursts_per_target=2,
                                seed=71, severity=.4, severity_min=.25,
                                resolution=resolution, photometric=PhotometricConfig(
                                    glare_probability=0, motion_probability=1, rgb_offset_probability=1))
            self.assertEqual([row["synthetic"]["severity"] for row in first["samples"]],
                             [row["synthetic"]["severity"] for row in no_glare["samples"]])
            self.assertTrue({row["id"] for row in no_glare["samples"]}.isdisjoint(row["id"] for row in first["samples"]))
            no_resolution = generate([root/"clean"], root/"no-resolution", frames=2,
                bursts_per_target=2, seed=71, severity=.4, severity_min=.25,
                resolution=CaptureResolutionConfig(probability=0))
            self.assertTrue(all(row["synthetic"]["captureResolution"] == {"side": 256, "blurSigma": 0}
                                for row in no_resolution["samples"]))
            self.assertEqual([row["synthetic"]["severity"] for row in first["samples"]],
                             [row["synthetic"]["severity"] for row in no_resolution["samples"]])
            self.assertTrue({row["id"] for row in no_resolution["samples"]}.isdisjoint(
                            row["id"] for row in first["samples"]))
            combined = Dataset([root/"real", root/"corrupted"], "train")
            sampler = BalancedSampler(combined.samples,
                {"target-fixture-receiver": "phone-to-pc"}, paired_targets=True)
            update = sum(sampler.draw_update(4, 2), [])
            for row in (sample for sample in update if sample.metadata["source"] == "real"):
                self.assertTrue(any(candidate.metadata["source"] == "synthetic"
                    and candidate.metadata["info"] == row.metadata["info"]
                    and candidate.metadata["hashes"]["labels"] == row.metadata["hashes"]["labels"]
                    for candidate in update))
            with self.assertRaisesRegex(ValueError, "session overlap"):
                assert_disjoint(real, Dataset([root/"corrupted"]))
            self.assertEqual(path.read_bytes(), original)
            for invalid in (0, -1, True, 1.5):
                with self.assertRaisesRegex(ValueError, "bursts-per-target"):
                    generate([root/"clean"], root/"invalid", bursts_per_target=invalid)
            for invalid in (-.1, .41, True, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "severity-min"):
                    generate([root/"clean"], root/"invalid", severity=.4, severity_min=invalid)
            with self.assertRaisesRegex(ValueError, "max-output-mib"):
                generate([root/"clean"], root/"over-budget", frames=1,
                         bursts_per_target=2, max_output_mib=2)


if __name__ == "__main__":
    unittest.main()
