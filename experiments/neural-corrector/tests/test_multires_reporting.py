"""Source-by-grid reports keep unlike populations and CER denominators explicit."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multires_evaluation import _breakdown, _selection, add, counter
from reporting import compact_report
from train import (format_multires_grids, format_training_metrics,
                   format_validation_metrics, size_macro_cer,
                   summarize_training_grids, training_source_summaries)


def training_row(wrong, cells, loss=1., samples=1):
    return {"lossSum": loss*samples, "samples": samples, "inputCaptures": 4*samples,
            "wrongCells": wrong, "dataCells": cells}


def combine_training(*rows):
    return {name: sum(row[name] for row in rows) for name in rows[0]}


class MultiresReportingTests(unittest.TestCase):
    def test_grid_sources_and_macro_cer_use_their_own_denominators(self):
        real8, synthetic8 = training_row(4, 40, 2.), training_row(0, 40, .2)
        real256, synthetic256 = training_row(900, 1000, 3.), training_row(100, 1000, .1)
        interval = {"updates": 2, "loss": 2.65, "wrongCells": 1004, "dataCells": 2080,
                    "groups": {},
                    "sources": {"synthetic": combine_training(synthetic8, synthetic256),
                                "real": combine_training(real8, real256)},
                    "sizes": {"256": combine_training(real256, synthetic256),
                              "8": combine_training(real8, synthetic8)},
                    "sizeSources": {"256": {"synthetic": synthetic256, "real": real256},
                                    "8": {"synthetic": synthetic8, "real": real8}}}
        grids = summarize_training_grids(interval["sizes"], interval["sizeSources"])
        self.assertEqual(list(grids), ["8", "256"])
        self.assertEqual(list(grids["256"]["sources"]), ["real", "synthetic"])
        self.assertEqual(grids["256"]["sources"]["real"]["cellErrorRate"], .9)
        self.assertEqual(grids["256"]["cellErrorRate"], .5)
        self.assertEqual(grids["8"]["meanCE"], 1.1)
        self.assertAlmostEqual(size_macro_cer(grids), .275)
        sources = training_source_summaries(interval)
        self.assertAlmostEqual(sources["real"]["cellErrorRate"], 904/1040)
        self.assertEqual(sources["real"]["sizeMacroCellErrorRate"], .5)
        self.assertEqual(sources["synthetic"]["sizeMacroCellErrorRate"], .05)
        text = format_training_metrics(interval)
        self.assertIn("CER 48.27% (cell) CER(size) 27.50%", text)
        self.assertLess(text.index("real CE"), text.index("synthetic CE"))
        self.assertNotIn("val", text.lower())
        self.assertIsNone(size_macro_cer({}))

    def test_grid_lines_use_real_validation_and_keep_empty_columns(self):
        training = summarize_training_grids(
            {"8": training_row(4, 40)}, {"8": {"real": training_row(4, 40)}})
        real = {"loss": .3, "cellErrorRate": .2,
                "matrixAccuracy": {"exactRate": .4}}
        validation = {"256": {"sources": {"real": real}},
                      "8": {"loss": 99., "cellErrorRate": .99, "sources": {"real": real}}}
        lines = format_multires_grids(training, validation)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("  G8 "))
        self.assertTrue(lines[1].startswith("  G256"))
        self.assertIn("train real CE 1.0000 CER 10.00%", lines[0])
        self.assertIn("train synthetic n/a", lines[0])
        self.assertIn("val real CE 0.3000 CER 20.00% exact 40.0%", lines[0])
        self.assertNotIn("99.0000", lines[0])
        for line in lines:
            self.assertLess(line.index("train real"), line.index("train synthetic"))
            self.assertLess(line.index("train synthetic"), line.index("val real"))
        self.assertIn("train real n/a | train synthetic n/a", lines[1])
        self.assertNotIn("nan", " ".join(format_multires_grids({}, validation)))

    def test_validation_macro_is_distinct_from_pooled_cer_and_selection(self):
        leaves = {}
        for size, source, wrong, cells, repetitions in (
                (8, "real", 36, 36, 20), (256, "real", 0, 65536, 1),
                (256, "synthetic", 65536, 65536, 1)):
            row = counter()
            for _ in range(repetitions):
                add(row, loss=1., wrong=wrong, cells=cells, captures=4)
            leaves[(size, source, "direction" if source == "real" else "source:synthetic", 4)] = row
        before = _selection(leaves)
        result = _breakdown(leaves)
        self.assertEqual(result["sizeMacroCellErrorRate"], .75)
        self.assertEqual(result["sources"]["real"]["sizeMacroCellErrorRate"], .5)
        self.assertAlmostEqual(result["sources"]["real"]["cellErrorRate"], 720/(720+65536))
        self.assertEqual(_selection(leaves), before)
        report = {**result, "format": "neural-corrector-evaluation", "version": 1,
                  "selectionScore": before, "cases": [{"secretCapture": str(i)} for i in range(10000)],
                  "bySize": {str(size): _breakdown({k: v for k, v in leaves.items() if k[0] == size})
                             for size in (8, 256)}}
        original = copy.deepcopy(report)
        lean = compact_report(report)
        self.assertEqual(report, original)
        self.assertEqual(compact_report(lean), lean)
        self.assertEqual(lean["sizeMacroCellErrorRate"], .75)
        self.assertEqual(lean["sources"]["real"]["sizeMacroCellErrorRate"], .5)
        self.assertEqual(list(lean["bySize"]["256"]["sources"]), ["real", "synthetic"])
        self.assertNotIn("secretCapture", json.dumps(lean))
        self.assertLess(len(json.dumps(lean)), 8000)
        text = format_validation_metrics(lean)
        self.assertIn("(cell) CER(size) 75.00%", text)


if __name__ == "__main__":
    unittest.main()
