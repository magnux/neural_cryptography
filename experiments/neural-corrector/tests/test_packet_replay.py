"""Recorded-error replay must remain distinct from raw matrix correctness."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Dataset
from evaluate import evaluate_model, format_evaluation_summary
from packet_replay import ReplayBatch, pack_data_cells, replay_policy, validate_rs_parity
from reporting import compact_report, evaluation_document, evaluation_step, lean_json
from test_wtube_evaluation import ConstantCorrector, corpus

torch.set_num_threads(1)


class SparseErrorCorrector(ConstantCorrector):
    def __init__(self, coordinate):
        super().__init__()
        self.coordinate = coordinate

    def forward(self, pixels):
        output = super().forward(pixels)
        y, x = self.coordinate
        output[:, 1, y, x] = 8
        return output


class PacketReplayTests(unittest.TestCase):
    def test_packing_excludes_references_and_accounts_for_unpaired_nibble(self):
        symbols = np.array([[15, 1, 2], [3, 15, 15]], dtype=np.uint8)
        mask = np.array([[False, True, True], [True, False, False]])
        self.assertEqual(pack_data_cells(symbols, mask), b"\x12")
        with self.assertRaisesRegex(ValueError, "integer"):
            pack_data_cells(symbols.astype(float), mask)
        invalid = symbols.copy()
        invalid[0, 1] = 16
        with self.assertRaisesRegex(ValueError, "sixteen"):
            pack_data_cells(invalid, mask)
        for value in (-1, 1, 3, 65, True, 32.):
            with self.assertRaises(ValueError):
                validate_rs_parity(value)
        for value in (0, 2, 32, 64):
            self.assertEqual(validate_rs_parity(value), value)
        self.assertFalse(replay_policy(0)["enabled"])

    def test_canonical_bridge_handles_clean_correctable_and_ineligible_cases(self):
        symbols = np.arange(256, dtype=np.uint8).reshape(16, 16) % 16
        mask = np.ones_like(symbols, dtype=bool)
        corrupt = symbols.copy()
        corrupt.flat[12] ^= 7
        with tempfile.TemporaryDirectory() as temp:
            with ReplayBatch(32, work_dir=temp) as batch:
                self.assertEqual(batch.append(symbols, symbols, mask), 0)
                batch.append(symbols, corrupt, mask)
                batch.append(symbols[:2], symbols[:2], mask[:2])
                answers = list(batch.results())
                self.assertEqual([row["id"] for row in answers], ["0", "1", "2"])
                self.assertEqual([row["eligible"] for row in answers], [1, 1, 0])
                self.assertEqual([row["exact"] for row in answers], [1, 1, 0])
                self.assertEqual(answers[1]["crcAccepted"], 1)
                self.assertEqual(answers[1]["wrongAccepted"], 0)
                self.assertEqual(answers[1]["wireBytes"], 128)
                self.assertEqual(answers[1]["payloadBytes"], 92)
                self.assertEqual(answers[1]["parityBytes"], 32)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_raw_metrics_selection_and_capture_members_do_not_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = Dataset([corpus(root/"data", "native", 16, 0)], "validation")
            mask = dataset.samples[0].tensors()[2]
            model = SparseErrorCorrector(tuple(np.argwhere(mask)[0]))
            model.train()
            before = evaluate_model(model, dataset, rs_parity=0, detailed=True, work_dir=root)
            report = evaluate_model(model, dataset, rs_parity=32, detailed=True, work_dir=root)
            self.assertTrue(model.training)
            self.assertEqual(report["matrixAccuracy"], before["matrixAccuracy"])
            self.assertEqual(report["selectionScore"], before["selectionScore"])
            self.assertEqual(report["loss"], before["loss"])
            self.assertEqual(report["cellErrorRate"], before["cellErrorRate"])
            self.assertEqual([r["memberIds"] for r in report["cases"]],
                             [r["memberIds"] for r in before["cases"]])
            self.assertEqual(report["matrixAccuracy"]["exactRate"], 0)
            replay = report["rsReplay"]
            self.assertEqual(replay["cases"], 5)
            self.assertEqual(replay["eligibleCases"], 5)
            self.assertEqual(replay["rsDecodedCases"], 5)
            self.assertEqual(replay["crcAcceptedCases"], 5)
            self.assertEqual(replay["exactCases"], 5)
            self.assertEqual(replay["wrongAcceptedCases"], 0)
            for row in report["byFrameCount"].values():
                self.assertEqual(row["rsReplay"]["exactCases"], 1)
            self.assertEqual(report["bySize"]["16"]["rsReplay"], replay)
            self.assertEqual(report["origins"]["native"]["rsReplay"], replay)
            text = "\n".join(format_evaluation_summary(report))
            self.assertIn("RS+CRC replay 5/5", text)
            self.assertIn("not measured packet reception", text)
            self.assertFalse(before["rsReplayPolicy"]["enabled"])
            self.assertNotIn("rsReplay", before)

            compact = compact_report(report)
            self.assertEqual(compact_report(compact), compact)
            self.assertNotIn("cases", compact)
            document = evaluation_document(compact)
            self.assertEqual(document["metadata"]["rsReplayPolicy"]["metric"], "recorded-error-replay")
            self.assertEqual(document["headline"]["rsReplay"], replay)
            rows = [dict(zip(document["tables"]["evaluation"], row)) for row in document["rows"]]
            self.assertEqual(sum(row["rsExactCases"] for row in rows), 5)
            self.assertEqual(sum(row["rsPayloadBytes"] for row in rows), replay["payloadBytes"])
            self.assertEqual(sum(row["exactMatrices"] for row in rows), 0)
            self.assertNotIn("rsReplayPolicy", evaluation_step(report))
            self.assertEqual(evaluation_document(document), document)
            self.assertLess(len(lean_json(document)), 7000)
            self.assertEqual(json.loads(lean_json(document))["headline"]["rsReplay"]["exactCases"], 5)

    def test_disabled_does_not_spawn_decoder_and_missing_node_is_actionable(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Dataset([corpus(Path(temp)/"data", "native", 16, 0)], "validation")
            with patch("packet_replay.subprocess.run", side_effect=AssertionError("should not run")):
                report = evaluate_model(ConstantCorrector(), dataset, rs_parity=0)
            self.assertNotIn("rsReplay", report)
            with self.assertRaisesRegex(RuntimeError, "check --node or use --rs-parity 0"):
                evaluate_model(ConstantCorrector(), dataset, node="/missing-node", work_dir=temp)
            self.assertEqual([p.name for p in Path(temp).iterdir()], ["data"])


if __name__ == "__main__":
    unittest.main()
