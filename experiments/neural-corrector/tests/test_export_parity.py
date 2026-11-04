"""Float32 export agreement uses the deployed browser policy without ignoring outliers."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments' / 'neural-corrector'))
from contract import ATOL, RTOL, parity_metrics


class ExportParityTests(unittest.TestCase):
    def test_reported_near_zero_drift_passes_for_every_value(self):
        for count, maximum in ((4, 4.955381155014038e-5), (8, 5.91278076171875e-5)):
            with self.subTest(count=count):
                expected = np.zeros((1, 16, 64, 64), np.float32)
                actual = expected.copy()
                actual.reshape(-1)[-count:] = maximum
                self.assertEqual(int((np.abs(actual - expected) > 2e-5).sum()), count)
                report = parity_metrics(expected, actual)
                self.assertTrue(report['passed'])
                self.assertEqual(report['mismatchedValues'], 0)
                self.assertEqual(report['values'], 65536)
                self.assertEqual(report['decisions']['argmaxChangedCells'], count)
                self.assertEqual(report['decisions']['maxReferenceMarginAtChangedCells'], 0.0)

    def test_one_late_outlier_still_fails_even_if_decisions_are_unchanged(self):
        expected = np.zeros((1, 16, 64, 64), np.float32)
        expected[:, 0] = 1
        actual = expected.copy()
        actual.reshape(-1)[-1] = 1.1e-4
        report = parity_metrics(expected, actual)
        self.assertFalse(report['passed'])
        self.assertEqual(report['mismatchedValues'], 1)
        self.assertEqual(report['decisions']['argmaxChangedCells'], 0)

    def test_relative_tolerance_and_shape_finite_guards_remain_active(self):
        expected = np.array([100.0, -100.0, 0.0])
        self.assertTrue(parity_metrics(expected, expected + np.array([.05, -.05, 0]))['passed'])
        self.assertFalse(parity_metrics(expected, expected + np.array([.11, 0, 0]))['passed'])
        for invalid in (np.zeros(2), np.array([np.nan, 0, 0]), np.array([0, np.inf, 0])):
            with self.assertRaises(ValueError):
                parity_metrics(expected, invalid)
        with self.assertRaises(ValueError):
            parity_metrics(np.array([]), np.array([]))

    def test_near_tie_decision_changes_are_reported_without_becoming_a_new_gate(self):
        expected = np.full((1, 16, 1, 2), -3.0, dtype=np.float32)
        expected[:, 0] = 0
        expected[:, 1] = -2e-5
        actual = expected.copy()
        actual[0, 1, 0, 0] = 2e-5
        report = parity_metrics(expected, actual)
        self.assertTrue(report['passed'])
        self.assertEqual(report['decisions']['cells'], 2)
        self.assertEqual(report['decisions']['argmaxChangedCells'], 1)
        self.assertAlmostEqual(report['decisions']['maxReferenceMarginAtChangedCells'], 2e-5, places=10)
        self.assertGreater(report['decisions']['maxProbabilityError'], 0)
        self.assertLess(report['decisions']['maxProbabilityError'], 1e-4)
        self.assertEqual(parity_metrics(expected, expected)['decisions']['maxProbabilityError'], 0)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for the cross-language policy check')
    def test_python_and_browser_policies_and_gate_results_match(self):
        cases = [([0, 0], [0, 5.91278076171875e-5]), ([0, 0], [0, 1.1e-4]),
                 ([100, -.01], [100.05, -.01]), ([100, -.01], [100, -.011])]
        module = (ROOT / 'apps/color-optical-bench/dist/js/corrector-lab/runtime.js').as_uri()
        script = f"import {{PARITY_TOLERANCE,compareLogits}} from {json.dumps(module)}; "
        script += 'console.log(JSON.stringify({policy:PARITY_TOLERANCE,checks:'
        script += json.dumps(cases) + '.map(([expected,actual])=>compareLogits(actual,expected).passed)}));'
        value = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', script], text=True))
        self.assertEqual(value['policy'], {'atol': ATOL, 'rtol': RTOL})
        self.assertEqual(value['checks'], [parity_metrics(np.array(e), np.array(a))['passed'] for e, a in cases])


if __name__ == '__main__':
    unittest.main()
