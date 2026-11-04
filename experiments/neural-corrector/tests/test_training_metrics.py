"""Readable training summaries use stable fields and the actual supervised cells."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from train import (format_training_metrics, format_validation_metrics,
                       summarize_training_counters, training_cell_counts)


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for training metrics")
class TrainingMetricTests(unittest.TestCase):
    def test_training_cer_ignores_reference_cells_and_inactive_palette_logits(self):
        logits = torch.zeros((2, 16, 2, 2))
        logits[:, 15] = 1000  # Invalid palette class must not win the argmax.
        labels = torch.zeros((2, 2, 2), dtype=torch.long)
        labels[0, 0, 1] = 1
        labels[1] = 2  # Wrong predictions on pilots do not count.
        mask = torch.tensor([[[True, True], [False, False]],
                             [[False, False], [True, False]]])
        logits[1, 2, 1, 0] = 1
        wrong, counts = training_cell_counts(logits, labels, mask, 4)
        self.assertEqual(wrong.tolist(), [1, 0])
        self.assertEqual(counts.tolist(), [2, 1])

    def test_live_training_fields_are_stable_under_source_and_group_draw_order(self):
        row = {"lossSum": .4, "samples": 2, "inputCaptures": 8,
               "wrongCells": 1, "dataCells": 10}
        interval = {"loss": .4, "updates": 2, "wrongCells": 2, "dataCells": 20,
                    "sources": {"synthetic": row, "real": row},
                    "groups": {"phone-to-pc": row, "pc-to-phone": row}}
        first = format_training_metrics(interval)
        reversed_interval = {**interval,
            "sources": dict(reversed(list(interval["sources"].items()))),
            "groups": dict(reversed(list(interval["groups"].items())))}
        self.assertEqual(first, format_training_metrics(reversed_interval))
        self.assertLess(first.index("real CE"), first.index("synthetic CE"))
        self.assertLess(first.index("pc-to-phone CE"), first.index("phone-to-pc CE"))
        self.assertNotIn("val", first.lower())
        self.assertNotIn("exact", first.lower())
        self.assertIn("CER 10.00%", first)
        summary = summarize_training_counters(interval["sources"], sources=True)
        self.assertEqual(list(summary), ["real", "synthetic"])
        self.assertEqual(summary["real"]["cellErrorRate"], .1)

    def test_validation_distinguishes_raw_recovery_from_selection_objective(self):
        report = {"loss": .02, "cellErrorRate": .0064,
                  "selectionScore": {"exactRate": .9354, "method": "frame-count-group-balanced-real"},
                  "recovery": {"summary": {"learned": {
                      "exactFrames": 529, "frames": 560, "exactRate": 529/560}}},
                  "byFrameCount": {str(k): {"recovery": {"summary": {"learned": {"exactRate": .8}}}}
                                   for k in (8, 6, 4, 7, 5)}}
        text = format_validation_metrics(report)
        self.assertIn("529/560 (94.46%)", text)
        self.assertIn("selection 93.54%", text)
        self.assertEqual([text.index(f"K{k}:") for k in range(4, 9)],
                         sorted(text.index(f"K{k}:") for k in range(4, 9)))


if __name__ == "__main__":
    unittest.main()
