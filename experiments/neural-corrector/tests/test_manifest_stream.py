"""Large offline manifests keep their format without a full-size JSON string."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"experiments/neural-corrector"))
from data import Dataset, load_manifest
from test_multires_synthetic import JOINED_FIXTURE


class ManifestStreamTests(unittest.TestCase):
    def test_chunk_boundaries_preserve_key_order_unicode_and_number_tokens(self):
        document = {"samples": [{"id": "camera-é-🌈", "value": 1.234e-19,
                                  "nested": {"quotes": '\\" brackets ] }', "flag": True}},
                                 {"id": "second", "value": -1234567890123456789}],
                    "format": "neural-qr-dataset", "version": 2,
                    "generator": {"seed": 71, "unknown": None}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"manifest.json"
            for indent in (None, 2):
                path.write_text(json.dumps(document, ensure_ascii=False, indent=indent), encoding="utf8")
                for chunk in (1, 3, 7, 64, 65536):
                    with self.subTest(indent=indent, chunk=chunk):
                        self.assertEqual(load_manifest(path, chunk_size=chunk), document)

    def test_malformed_duplicate_and_nonfinite_values_are_rejected(self):
        inputs = ['[]', '{} {}', '{"samples":[{}', '{"samples":[{},]}',
                  '{"samples":[],}', '{"samples":[],"samples":[]}',
                  '{"samples":[{"id":1,"id":2}]}', '{"samples":[NaN]}',
                  '{"samples":[Infinity]}', '{"samples":[1e]}',
                  '{"samples":[1 2]}', '{"samples" []}']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"manifest.json"
            for value in inputs:
                path.write_text(value)
                for chunk in (1, 17):
                    with self.subTest(value=value, chunk=chunk), self.assertRaises(ValueError):
                        load_manifest(path, chunk_size=chunk)

    def test_limit_applies_to_each_value_not_array_or_read_ahead(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"manifest.json"
            sample = {"id": "x"*113}
            self.assertLess(len(json.dumps(sample, separators=(",", ":"))), 128)
            document = {"samples": [sample]*100, "version": 2}
            path.write_text(json.dumps(document, separators=(",", ":")))
            with patch("data.MAX_MANIFEST_VALUE_CHARS", 128):
                self.assertEqual(load_manifest(path, chunk_size=64), document)
                path.write_text(json.dumps({"samples": [{"id": "x"*129}]}))
                with self.assertRaisesRegex(ValueError, "Individual manifest JSON value"):
                    load_manifest(path, chunk_size=64)

    def test_cli_reports_counts_and_progress_with_reloadable_compact_manifest(self):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            subprocess.run(['node', '--input-type=module', '-e', JOINED_FIXTURE, str(root), '16'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            result = subprocess.run([sys.executable, str(ROOT/'experiments/neural-corrector/synthetic.py'),
                '--clean', str(root/'clean'), '--out', str(root/'generated'),
                '--paired-real', str(root/'joined'), '--frames', '2', '--severity', '0'],
                cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads(result.stdout)
            self.assertEqual(summary['samples'], 8)
            self.assertEqual(summary['out'], str(root/'generated'))
            self.assertNotIn('hashes', summary)
            self.assertIn('Synthetic captures: 0/8', result.stderr)
            self.assertIn('Synthetic captures: 8/8', result.stderr)
            self.assertIn('Preparing canonical baselines', result.stderr)
            data = Dataset([root/'generated'], 'train')
            self.assertEqual(len(data.samples), summary['samples'])
            for sample in data.samples:
                sample.read('baseline', '<f4')
            text = (root/'generated/manifest.json').read_text()
            self.assertEqual(text.count('\n'), 1)


if __name__ == '__main__':
    unittest.main()
