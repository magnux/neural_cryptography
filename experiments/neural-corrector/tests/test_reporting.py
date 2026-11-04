"""Summary size follows groups and presets, never session or capture counts."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reporting import compact_report


def report_with_detail(count):
    metrics = {"samples": 112, "dataCells": 4096, "wrongCells": 26,
               "loss": .0198, "cellErrorRate": .0064,
               "learnedExactFrames": 112, "learnedExactRate": 1.,
               "classicalExactFrames": 0, "classicalExactRate": 0.}
    recovery = {"frames": 112, "rawCRCFrames": 0, "hardCRCFrames": 112,
                "acceptedFrames": 112, "exactFrames": 112,
                "wrongAcceptedFrames": 0, "exactRate": 1.}
    single = {"samples": 112, "framesPerInput": 8, "frameCapacity": 8,
        "dataCells": 4096, "loss": .0198, "cellErrorRate": .0064,
        "expectedCalibrationError": .01,
        "groups": {"phone-to-pc": metrics}, "sources": {"real": metrics},
        "selectionScore": {"method": "group-balanced-real", "groups": ["phone-to-pc"],
                           "exactRate": 1., "loss": .0198, "cellErrorRate": .0064},
        "sessions": {f"private-session-{i}": copy.deepcopy(metrics) for i in range(count)},
        "recovery": {"summary": {"classical": recovery, "learned": recovery},
                     "inputCaptures": 896, "framesPerInput": 8,
                     "samples": [{"id": f"private-capture-{i}", "memberIds": [f"member-{i}-{j}" for j in range(8)]}
                                 for i in range(count)]},
        "confusion": [[i+j for i in range(16)] for j in range(16)],
        "calibrationBins": [{"count": 500, "accuracy": .9, "confidence": .95}]*15,
        "inferenceMs": {"p50": 10., "p95": 20., "samples": list(range(count))}}
    return {"format": "neural-corrector-evaluation", "version": 1, **single,
        "selection": {"seed": 7, "omittedSessions": [f"private-omitted-{i}" for i in range(count)]},
        "byFrameCount": {str(k): {**single, "framesPerInput": k} for k in range(4, 9)}}


class ReportTests(unittest.TestCase):
    def test_size_does_not_grow_with_repeated_capture_and_session_details(self):
        small = compact_report(report_with_detail(2))
        large = compact_report(report_with_detail(10000))
        small_text, large_text = json.dumps(small), json.dumps(large)
        self.assertLess(len(large_text), 16000)
        self.assertLess(abs(len(large_text)-len(small_text)), 10)
        self.assertNotIn("private-", large_text)
        self.assertNotIn("member-", large_text)
        self.assertEqual(large["selection"]["omittedSessionCount"], 10000)

    def test_projection_preserves_aggregates_and_does_not_mutate_diagnostics(self):
        original = report_with_detail(3)
        before = copy.deepcopy(original)
        compact = compact_report(original)
        self.assertEqual(original, before)
        self.assertEqual(compact_report(compact), compact)
        for left, right in [(original, compact), *[(original["byFrameCount"][str(k)],
                                                   compact["byFrameCount"][str(k)]) for k in range(4, 9)]]:
            for key in ("samples", "framesPerInput", "loss", "cellErrorRate", "groups", "sources", "selectionScore"):
                self.assertEqual(left[key], right[key])
            self.assertEqual(left["recovery"]["summary"], right["recovery"]["summary"])
            self.assertNotIn("sessions", right)
            self.assertNotIn("confusion", right)
            self.assertNotIn("calibrationBins", right)
            self.assertNotIn("samples", right["recovery"])
            self.assertNotIn("samples", right["inferenceMs"])

    def test_unknown_recovery_stays_unknown_and_order_is_stable(self):
        report = report_with_detail(0)
        report["recovery"] = None
        report["groups"] = {"z-direction": report["groups"]["phone-to-pc"],
                            "a-direction": report["groups"]["phone-to-pc"]}
        report["byFrameCount"] = dict(reversed(list(report["byFrameCount"].items())))
        compact = compact_report(report)
        self.assertIsNone(compact["recovery"])
        self.assertEqual(list(compact["groups"]), ["a-direction", "z-direction"])
        self.assertEqual(list(compact["byFrameCount"]), ["4", "5", "6", "7", "8"])


if __name__ == "__main__":
    unittest.main()
