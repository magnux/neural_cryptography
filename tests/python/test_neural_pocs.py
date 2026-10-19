"""Run from the repo root: python -m unittest discover -s tests/python -v.

NumPy tests always run; neural dynamics tests additionally require PyTorch.
"""
import base64
import contextlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
REPO = Path(__file__).resolve().parents[2]
for path in (REPO / 'shared/python', REPO / 'experiments/neural-dynamics', REPO / 'experiments/python-zkboo'):
    sys.path.insert(0, str(path))
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    import neural_key_experiment as base
import neural_zkboo_experiment as zk
import neural_optical as optical


class POCRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is not None:
            torch.set_num_threads(1)

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_base_invariants(self):
        base.self_test()

    def test_proof_and_transport_invariants(self):
        zk.zk_self_test()

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_sample_space_covariance_matches_full_covariance(self):
        bits = np.random.default_rng(22).integers(0, 2, size=(13, 2, 17, 19), dtype=np.uint8)
        bits[:, 0, 0, :4] = 0
        flat = bits.reshape(len(bits), -1).astype(float)
        centered = flat - flat.mean(axis=0)
        cov = centered.T @ centered / len(bits)
        expected = np.trace(cov)**2 / np.square(cov).sum()
        result = base.distribution_stats(bits)
        self.assertAlmostEqual(result['covariance_participation_ratio'], expected, places=12)
        self.assertTrue(result['pair_correlation_is_sampled'])
        self.assertEqual(result['pair_correlation_coordinates'], 512)

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_sequence_inputs_reach_other_response_frames(self):
        model = base.NeuralKey(size=8, colors=16, frames=3, architecture='feistel', init='orthogonal')
        original = base.sample_challenges(4, 8, 41, colors=16, frames=3)
        changed = original.copy()
        changed[:, 0, 0, 0] ^= 1
        a = base.evaluate(model, original, [8], 4, bits_only=True)[8]
        b = base.evaluate(model, changed, [8], 4, bits_only=True)[8]
        a = base.bits_to_symbols(a, 16, 3)
        b = base.bits_to_symbols(b, 16, 3)
        self.assertTrue(np.any(a[:, 1:] != b[:, 1:]))
        self.assertEqual(a.shape, original.shape)

    def test_png_and_viewer_preserve_symbols_and_remove_stale_frames(self):
        grids = np.random.default_rng(27).integers(0, 16, (8, 32, 32), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            optical.save_sequence(folder, 'challenge', grids, 16)
            optical.save_sequence(folder, 'challenge', grids[:2], 16)
            self.assertEqual(len(list(folder.glob('challenge_[0-9][0-9].png'))), 2)
            text = (folder / 'challenge_display.html').read_text()
            urls = json.loads(re.search(r'const frames=(\[.*?\]);const palette=', text).group(1))
            self.assertEqual(len(urls), 2)
            for index, url in enumerate(urls):
                content = base64.b64decode(url.split(',', 1)[1], validate=True)
                self.assertEqual(content, (folder / f'challenge_{index + 1:02d}.png').read_bytes())
                self.assertEqual(content[:8], b'\x89PNG\r\n\x1a\n')
                position = 8
                compressed = b''
                while position < len(content):
                    count = struct.unpack_from('>I', content, position)[0]
                    kind = content[position + 4:position + 8]
                    payload = content[position + 8:position + 8 + count]
                    if kind == b'PLTE':
                        self.assertEqual(payload, optical.palette(16).tobytes())
                    if kind == b'IDAT':
                        compressed += payload
                    position += count + 12
                scanlines = np.frombuffer(zlib.decompress(compressed), np.uint8).reshape(32, 33)
                self.assertTrue(np.all(scanlines[:, 0] == 0))
                np.testing.assert_array_equal(scanlines[:, 1:], grids[index])

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_cli_boundaries_and_memory_preflight(self):
        for arguments in (['--size', '513'], ['--frames', '9'], ['--colors', '32'],
                          ['--size', '9', '--colors', '2', '--architecture', 'feistel']):
            with patch.object(sys, 'argv', ['neural_key_experiment.py'] + arguments):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    base.parse_args()
        with patch.object(sys, 'argv', ['neural_key_experiment.py', '--size', '512', '--colors', '16', '--frames', '8']):
            args = base.parse_args()
        with self.assertRaisesRegex(ValueError, 'Estimated working arrays'):
            base.prepare_workload(args)

    def test_proof_script_has_no_torch_dependency(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; import neural_zkboo_experiment; assert "torch" not in sys.modules'],
            cwd=REPO / "experiments/python-zkboo", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_compact_cli_uses_canonical_engine(self):
        import neural_zkboo_reference as core
        self.assertIs(zk.zk_prove, core.zk_prove)
        self.assertIs(zk.zk_verify, core.zk_verify)
        self.assertIs(zk.ZKBinaryCircuit, core.ZKBinaryCircuit)
        self.assertEqual(core.ZK_MAGIC, b"NPZK0002")
        self.assertEqual(core.zk_proof_size(core.ZKBinaryCircuit(8, 16), 32), 17474)

    def test_cli_public_only_verification_and_repetition_policy(self):
        script = REPO / 'experiments/python-zkboo/neural_zkboo_experiment.py'
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            def run(*arguments, expected=0):
                result = subprocess.run([sys.executable, str(script), '--out', str(folder), *arguments],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return result
            run('--zk-action', 'keygen', '--state-size', '4', '--zk-rounds', '4',
                '--minimum-repetitions', '16')
            run('--zk-action', 'challenge', '--required-repetitions', '32', '--size', '64')
            public = json.loads((folder / 'public_key.json').read_text())
            challenge = json.loads((folder / 'challenge.json').read_text())
            self.assertEqual(public['minimum_repetitions'], 16)
            self.assertEqual(challenge['required_repetitions'], 32)
            # An explicit too-small count fails before generating a proof.
            run('--zk-action', 'prove', '--repetitions', '16', expected=1)
            self.assertFalse((folder / 'proof.bin').exists())
            run('--zk-action', 'prove', '--size', '64', '--camera', '--camera-noise', '0')
            proof = (folder / 'proof.bin').read_bytes()
            self.assertEqual(zk.ZK_HEADER.unpack_from(proof)[:2], (b'NPZK0002', 32))
            report = json.loads((folder / 'summary.json').read_text())
            self.assertEqual(report['repetitions'], 32)
            self.assertEqual(report['required_repetitions'], 32)
            self.assertEqual(report['minimum_repetitions'], 16)
            self.assertTrue(report['camera']['public_verification'])
            self.assertEqual(report['proof_bytes'], zk.zk_proof_size(zk.zk_validate_public(public), 32))
            (folder / 'private_key.json').unlink()
            run('--zk-action', 'verify', '--captured', str(folder / 'captured_frames.npz'))
            replay = run('--zk-action', 'verify', expected=1)
            self.assertIn('already consumed', replay.stdout)

    def test_cli_demo_uses_explicit_repetitions_for_new_key_and_challenge(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable,
                str(REPO / 'experiments/python-zkboo/neural_zkboo_experiment.py'),
                '--out', directory, '--state-size', '4', '--zk-rounds', '4', '--repetitions', '32'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual((report['minimum_repetitions'], report['required_repetitions'], report['repetitions']),
                             (32, 32, 32))
            self.assertTrue(all(report['negative_checks'].values()))


if __name__ == '__main__':
    unittest.main()
