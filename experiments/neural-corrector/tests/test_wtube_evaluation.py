"""Scoring-policy fixtures, not measurements of learned camera accuracy."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Dataset
from evaluate import evaluate_model
from reporting import compact_report
from test_multires_training import fixture

torch.set_num_threads(1)


def corpus(root, origin, size=16, symbol=0):
    fixture(root, "validation", size, seed=100+symbol)
    path = root/"manifest.json"
    manifest = json.loads(path.read_text())
    manifest["version"] = 2
    for row in manifest["samples"]:
        row["origin"] = origin
        for key in ("id", "sessionId", "senderSessionId", "burstId"):
            row[key] = origin+"/"+row[key]
        row["pixelsPerCell"] = row["info"]["pixelsPerCell"] = 4
        row["inputShape"] = [3, 4*size, 4*size]
        pixels = np.fromfile(root/row["input"], dtype="<f4").reshape(3, 2*size, 2*size)
        pixels = pixels.repeat(2, axis=1).repeat(2, axis=2)
        logical = np.full((size, size), symbol, dtype=np.uint8)
        mask = np.fromfile(root/row["mask"], dtype="u1").reshape(size, size).astype(bool)
        # The constant classifier deliberately gets reference cells wrong;
        # neither loss nor exact-matrix scoring may include those positions.
        logical[~mask] = 7
        labels = logical.copy()
        labels[mask] ^= (0, 15, 7, 8)[row["info"]["phase"]]
        for key, value in (("input", pixels), ("expected", logical), ("labels", labels)):
            raw = value.tobytes()
            (root/row[key]).write_bytes(raw)
            row["hashes"][key] = hashlib.sha256(raw).hexdigest()
        row["payloadId"] = row["hashes"]["expected"]
    path.write_text(json.dumps(manifest))
    return root


class ConstantCorrector(torch.nn.Module):
    profile = "W-tube"
    pixels_per_cell = 4
    supported_sizes = (16, 32, 64, 128)

    def forward(self, pixels):
        g = pixels.shape[-1]//4
        output = pixels.new_zeros((pixels.shape[0], 16, g, g))
        output[:, 0] = 4
        return output


class WTubeEvaluationTests(unittest.TestCase):
    def test_native_metrics_and_selection_do_not_pool_easy_or_hard_auxiliary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = corpus(root/"native", "native", 16, 0)
            resampled = corpus(root/"resampled", "resampled", 32, 1)
            data = Dataset([native, resampled], "validation")
            report = evaluate_model(ConstantCorrector(), data, detailed=True)
            lean = compact_report(report)
            self.assertEqual(lean, compact_report(lean))
            self.assertEqual(lean["modelProfile"], "W-tube")
            self.assertEqual(lean["pixelsPerCell"], 4)
            self.assertEqual(lean["datasetVersion"], 2)
            self.assertEqual(lean["evaluationPopulation"], "native")
            self.assertEqual(lean["matrixAccuracy"]["exactRate"], 1)
            self.assertEqual(lean["selectionScore"]["exactRate"], 1)
            self.assertEqual(lean["selectionScore"]["originScope"], "native")
            self.assertEqual(lean["primarySizes"], [16])
            self.assertEqual(lean["evaluatedSizes"], [16, 32])
            self.assertEqual(lean["samples"], 5)
            self.assertEqual(lean["evaluatedCases"], 10)
            self.assertEqual(lean["uniqueCandidates"], 1)
            self.assertEqual(lean["uniqueInputCaptures"], 8)
            self.assertEqual(lean["origins"]["resampled"]["matrixAccuracy"]["exactRate"], 0)
            self.assertEqual(lean["origins"]["resampled"]["sizes"], [32])
            self.assertNotIn("bySize", lean["origins"]["resampled"])
            self.assertNotIn("cases", lean)
            for origin in ("native", "resampled"):
                rows = {row["frames"]: row for row in report["cases"] if row["origin"] == origin}
                self.assertEqual(rows[4]["id"], rows[8]["id"])
                self.assertTrue(set(rows[4]["memberIds"]) < set(rows[8]["memberIds"]))
            with self.assertRaisesRegex(ValueError, "cap omits"):
                evaluate_model(ConstantCorrector(), data, max_samples=1)

    def test_auxiliary_only_results_are_explicit_and_unknown_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = corpus(Path(temp)/"aux", "header-g64", 64, 1)
            data = Dataset([root], "validation")
            report = evaluate_model(ConstantCorrector(), data)
            self.assertEqual(report["evaluationPopulation"], "auxiliary-only")
            self.assertFalse(report["nativeAvailable"])
            self.assertEqual(report["selectionScore"]["originScope"], "auxiliary-only")
            self.assertEqual(report["matrixAccuracy"]["exactRate"], 0)
            data.samples[0].metadata.pop("origin")
            with self.assertRaisesRegex(ValueError, "origin metadata"):
                evaluate_model(ConstantCorrector(), data)


if __name__ == "__main__":
    unittest.main()
