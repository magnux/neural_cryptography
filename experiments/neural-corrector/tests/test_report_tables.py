"""Persist aggregate evidence once, without changing live selection arithmetic."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Dataset
from evaluate import evaluate_model, format_evaluation_summary
from multires_evaluation import _breakdown, _selection, counter
from reporting import (EVALUATION_COLUMNS, TRAIN_COLUMNS, compact_report,
                       evaluation_document, evaluation_step, evaluation_table,
                       lean_json, training_rows)
from test_reporting import report_with_detail
from test_wtube_evaluation import ConstantCorrector, corpus


def matrix_report():
    """Four sizes, two directions, three real origins plus synthetic coverage."""
    leaves = {}
    for grid in (16, 32, 64, 128):
        for source in ("real", "synthetic"):
            for group in (("pc-to-phone", "phone-to-pc") if source == "real" else ("source:synthetic",)):
                for origin in ("native", "resampled", "header-g64"):
                    for frames in range(4, 9):
                        samples = grid//16+frames
                        cells = samples*(grid*grid-16)
                        wrong = grid+frames if group == "phone-to-pc" else 0
                        row = counter()
                        row.update(samples=samples, lossSum=samples*.123456789123,
                                   wrongCells=wrong, dataCells=cells,
                                   exactMatrices=samples-int(wrong > 0), inputCaptures=samples*frames)
                        leaves[(grid, source, group, frames, origin)] = row
    primary = {key: value for key, value in leaves.items() if key[1] == "real" and key[4] == "native"}
    selection = _selection(primary)
    selection.update(originScope="native", nativeAvailable=True, origins=["native"])
    return {"format": "neural-corrector-evaluation", "version": 1,
            "modelProfile": "W-tube", "datasetFormat": "neural-qr-dataset", "datasetVersion": 2,
            "metric": "exact-masked-logical-matrix", "pixelsPerCell": 4,
            "actualFrameCounts": [4, 5, 6, 7, 8], "frameCapacity": 8,
            "evaluationPopulation": "native", "nativeAvailable": True,
            "primarySizes": [16, 32, 64, 128], "evaluatedSizes": [16, 32, 64, 128],
            "evaluatedCases": sum(row["samples"] for row in leaves.values()),
            "selectionScore": selection, "metricTable": evaluation_table(leaves),
            **_breakdown(primary),
            "cases": [{"id": "private-burst", "sessionId": "private-session"}],
            "inferenceMs": {"p50": 10., "p95": 20., "samples": [10., 20.]}}


class ReportTableTests(unittest.TestCase):
    def test_real_evaluation_table_retains_masked_counts_and_native_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = Dataset([corpus(root/"native", "native", 16, 0),
                            corpus(root/"auxiliary", "resampled", 32, 1)], "validation")
            detailed = evaluate_model(ConstantCorrector(), data, detailed=True)
            before = copy.deepcopy(detailed)
            doc = evaluation_document(compact_report(detailed))
            self.assertEqual(detailed, before)
            rows = [dict(zip(doc["tables"]["evaluation"], row)) for row in doc["rows"]]
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["origin"] for row in rows}, {"native", "resampled"})
            for row in rows:
                cases = [case for case in detailed["cases"]
                         if case["origin"] == row["origin"] and case["frames"] == row["frames"]]
                self.assertEqual(row["samples"], len(cases))
                self.assertEqual(row["dataCells"], sum(case["dataCells"] for case in cases))
                self.assertEqual(row["wrongCells"], sum(case["wrongCells"] for case in cases))
                self.assertEqual(row["lossSum"], sum(case["loss"] for case in cases))
                self.assertEqual(row["exactMatrices"], sum(case["exact"] for case in cases))
                self.assertEqual(row["inputCaptures"], sum(case["frames"] for case in cases))
            native = [row for row in rows if row["origin"] == "native"]
            samples = sum(row["samples"] for row in native)
            self.assertEqual(doc["headline"]["samples"], samples)
            self.assertAlmostEqual(doc["headline"]["loss"], sum(row["lossSum"] for row in native)/samples)
            self.assertEqual(doc["headline"]["exactMatrices"], sum(row["exactMatrices"] for row in native))
            self.assertEqual(doc["selection"]["exactRate"], detailed["selectionScore"]["exactRate"])
            self.assertEqual(doc["metadata"]["selectionPolicy"]["originScope"], "native")
            self.assertEqual(doc["coverage"]["primarySizes"], [16])
            self.assertEqual(doc["coverage"]["evaluatedSizes"], [16, 32])
            self.assertNotIn("private-session", lean_json(doc))
            self.assertNotIn("cases", doc)
            self.assertIn("cases", detailed)

    def test_rows_reconstruct_native_headline_and_balanced_selection(self):
        report = matrix_report()
        doc = evaluation_document(report)
        rows = [dict(zip(EVALUATION_COLUMNS, row)) for row in doc["rows"]]
        native = [row for row in rows if row["source"] == "real" and row["origin"] == "native"]
        total_samples, total_cells = sum(row["samples"] for row in native), sum(row["dataCells"] for row in native)
        self.assertEqual(doc["headline"]["samples"], total_samples)
        self.assertAlmostEqual(doc["headline"]["loss"], sum(row["lossSum"] for row in native)/total_samples)
        self.assertAlmostEqual(doc["headline"]["cellErrorRate"], sum(row["wrongCells"] for row in native)/total_cells)
        self.assertAlmostEqual(doc["headline"]["exactRate"], sum(row["exactMatrices"] for row in native)/total_samples)
        self.assertAlmostEqual(doc["selection"]["exactRate"],
                               sum(row["exactMatrices"]/row["samples"] for row in native)/len(native))
        self.assertAlmostEqual(doc["selection"]["loss"],
                               sum(row["lossSum"]/row["samples"] for row in native)/len(native))
        self.assertAlmostEqual(doc["selection"]["cellErrorRate"],
                               sum(row["wrongCells"]/row["dataCells"] for row in native)/len(native))
        self.assertEqual(doc["lossWeight"], "samples")
        self.assertNotIn("samples", doc["metadata"]["inferenceMs"])
        self.assertEqual(evaluation_document(doc), doc)
        self.assertEqual(lean_json(json.loads(lean_json(doc))), lean_json(doc))
        for key in ("bySize", "byFrameCount", "groups", "sources", "origins", "cases", "metricTable"):
            self.assertNotIn(key, doc)
        self.assertEqual(set(doc["selection"]), {"loss", "exactRate", "cellErrorRate"})

    def test_packet_table_preserves_cell_weighted_ce_and_both_recoveries(self):
        report = report_with_detail(300)
        doc = evaluation_document(report)
        self.assertEqual(doc["lossWeight"], "dataCells")
        self.assertEqual(len(doc["rows"]), 5)
        for values in doc["rows"]:
            row = dict(zip(EVALUATION_COLUMNS, values))
            self.assertIsNone(row["grid"])
            self.assertIsNone(row["origin"])
            self.assertEqual(row["lossSum"], .0198*4096)
            self.assertEqual(row["exactMatrices"], 112)
            self.assertEqual(row["classicalExactMatrices"], 0)
        self.assertEqual(doc["headline"]["recovery"], report["recovery"]["summary"])
        self.assertNotIn("private-", lean_json(doc))
        self.assertNotIn("confusion", lean_json(doc))

    def test_serialized_precision_is_bounded_and_does_not_change_live_values(self):
        source = {"selection": .123456789123, "count": 1234567890123456789, "unknown": None}
        before = copy.deepcopy(source)
        loaded = json.loads(lean_json(source))
        self.assertEqual(source, before)
        self.assertEqual(loaded["selection"], .12345679)
        self.assertEqual(loaded["count"], source["count"])
        self.assertIsInstance(loaded["count"], int)
        self.assertIsNone(loaded["unknown"])
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                lean_json({"bad": nonfinite})

    def test_console_is_aggregate_lines_with_stable_grid_order(self):
        report = matrix_report()
        report["bySize"] = {str(grid): {**report, "bySize": {}} for grid in (128, 64, 32, 16)}
        text = "\n".join(format_evaluation_summary(report))
        self.assertLess(text.index("  G16 |"), text.index("  G32 |"))
        self.assertLess(text.index("  G32 |"), text.index("  G64 |"))
        self.assertLess(text.index("  G64 |"), text.index("  G128 |"))
        self.assertIn("Population: native", text)
        self.assertIn("Direction phone-to-pc", text)
        self.assertNotIn("private-", text)
        self.assertNotIn("lossSum", text)
        self.assertLess(len(text), 1400)
        packet = "\n".join(format_evaluation_summary(report_with_detail(100)))
        self.assertIn("K4: 100.0%", packet)
        self.assertLess(len(packet), 700)

    def test_full_run_size_depends_on_group_table_not_detailed_cases(self):
        report = matrix_report()
        base = evaluation_document(report)
        counters = {}
        for row in report["metricTable"]["rows"]:
            values = dict(zip(EVALUATION_COLUMNS, row))
            key = tuple(values[name] for name in TRAIN_COLUMNS[:4])
            target = counters.setdefault(key, {name: 0 for name in TRAIN_COLUMNS[4:]})
            for name in target:
                target[name] += values[name]
        training = training_rows(counters)
        self.assertEqual(sorted({row[0] for row in training}), [16, 32, 64, 128])
        self.assertEqual([row[0] for row in training], sorted(row[0] for row in training))
        document = {"format": "neural-corrector-training-run", "version": 2,
                    "settings": {"model": "W-tube", "steps": 32000},
                    "validationMetadata": base["metadata"],
                    "tables": {"training": TRAIN_COLUMNS, "evaluation": EVALUATION_COLUMNS},
                    "history": [{"step": step*1000, "elapsedSeconds": step*1000.123456,
                        "lr": .0001, "selected": True,
                        "training": {"updates": 1000, "rows": training,
                                     "frameCounts": [[k, 500] for k in range(4, 9)]},
                        "validation": evaluation_step(report)} for step in range(33)]}
        encoded = lean_json(document)
        self.assertLess(len(encoded.encode()), 1024*1024)
        report["cases"] *= 10000
        self.assertEqual(evaluation_document(report), base)
        self.assertNotIn("private-", encoded)
        self.assertEqual(encoded.count('"selectionPolicy"'), 1)
        print(f"Canonical reporting fixture: validation={len(lean_json(base).encode())} bytes; "
              f"33-step history={len(encoded.encode())} bytes ({len(training)} train and {len(base['rows'])} evaluation rows/step)")


if __name__ == "__main__":
    unittest.main()
