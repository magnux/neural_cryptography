"""Neural-QR synthetic phase trajectories, bounds and target ownership."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import synthetic
    from data import Dataset
    from phases import canonical_labels
    from synthetic import (CaptureResolutionConfig, corrupt_phase_group, generate,
                           sample_capture_resolution)
    torch.set_num_threads(1)

# Canonical JS creates physical symbols; Python never reimplements palette/wire
# encoding. One observed phase deliberately exercises offline completion of the
# remaining clean phases by the shared targets command.
JOINED_FIXTURE = r"""
import {mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {createSender,floatBytes} from './apps/color-optical-bench/dist/js/dataset/contract.js';
import {gridLayout} from './apps/color-optical-bench/dist/js/grid.js';
import {targets} from './scripts/corrector-dataset.mjs';
const root=process.argv[1],size=Number(process.argv[2]),folder=join(root,'joined');
await mkdir(folder,{recursive:true});
const {manifest,frames}=await createSender({size,seed:71,sessionId:'synthetic-sender'});
const frame=frames[0],mask=new Uint8Array(size*size);
for(const cell of gridLayout(size,16,8).cells)mask[cell]=1;
const sample={id:'real-frame',sessionId:'synthetic-receiver',senderSessionId:manifest.sessionId,
 payloadId:manifest.payloadId,split:'train',source:'real',size,colors:16,pixelsPerCell:2,
 inputShape:[3,2*size,2*size],info:frame.info,burstId:'real-burst',timestamp:1,
 devicePair:{sender:'phone',receiver:'pc'},hashes:{}};
const data={input:floatBytes(new Float32Array(3*(2*size)**2)),labels:frame.symbols,
 mask,baseline:floatBytes(new Float32Array(16*size*size)),expected:frame.chunk};
for(const [key,value] of Object.entries(data)){
 sample[key]=key+'.bin';sample.hashes[key]=createHash('sha256').update(value).digest('hex');
 await writeFile(join(folder,sample[key]),value);
}
await writeFile(join(folder,'manifest.json'),JSON.stringify({format:'neural-qr-dataset',version:1,samples:[sample]}));
await targets({dataset:folder,out:join(root,'clean')});
"""


@unittest.skipUnless(HAS_TORCH, "PyTorch needed for multires synthetic generation")
class MultiresSyntheticTests(unittest.TestCase):
    def test_supported_grids_two_pixel_contract_and_one_pixel_capture_floor(self):
        for g in (8, 16, 32, 64, 128, 256):
            with self.subTest(g=g):
                clean = np.zeros((4, 3, 2*g, 2*g), dtype=np.float32)
                clean[:] = np.array([-.8, -.2, .4, .9], np.float32)[:, None, None, None]
                result = corrupt_phase_group(clean, 7, 2, 0, pixels_per_cell=2)
                np.testing.assert_array_equal(result, np.repeat(clean[:, None], 2, axis=1))
                config = CaptureResolutionConfig(probability=1, min_scale=.5, max_scale=.5)
                capture = sample_capture_resolution(2*g, 71, .5, config, pixels_per_cell=2)
                self.assertEqual(capture['side'], g)
        with self.assertRaisesRegex(ValueError, 'at least 0.5'):
            corrupt_phase_group(np.zeros((4, 3, 16, 16), np.float32), 7,
                                resolution=CaptureResolutionConfig(min_scale=.25), pixels_per_cell=2)

    def test_streamed_trajectory_matches_buffered_and_scales_pixel_offsets(self):
        clean = np.random.default_rng(8).uniform(-1, 1, (4, 3, 32, 32)).astype(np.float32)
        options = dict(pixels_per_cell=2, severity=.4, seed=9)
        with patch('synthetic.sample_photometric', wraps=synthetic.sample_photometric) as lighting, \
                patch('synthetic.rolling_shutter_maps', wraps=synthetic.rolling_shutter_maps) as shutter:
            observed = list(synthetic._iter_corrupt_sequence(clean, np.tile(np.arange(4), 2), **options))
        self.assertEqual(lighting.call_count, 1)
        self.assertEqual([call.args[2] for call in shutter.call_args_list], list(range(8)))
        for call in shutter.call_args_list:
            if call.args[1] is not None:
                self.assertLessEqual(call.args[1]['skew'], .5*.4*.5)
        expected = np.stack(observed).reshape(2, 4, 3, 32, 32).transpose(1, 0, 2, 3, 4)
        np.testing.assert_array_equal(expected, corrupt_phase_group(clean, frames=2, **options))
        self.assertTrue(np.isfinite(expected).all())
        self.assertLessEqual(float(np.abs(expected).max()), 1)
        self.assertFalse(np.array_equal(expected[0, 0], expected[0, 1]))
        # G256 exercises the full dense-crop size without allocating an entire
        # 32-capture output batch: only the first yielded view is requested.
        iterator = synthetic._iter_corrupt_sequence(np.zeros((4, 3, 512, 512), np.float32),
            np.tile(np.arange(4), 8), seed=9, severity=.4, pixels_per_cell=2)
        large = next(iterator)
        iterator.close()
        self.assertEqual(large.shape, (3, 512, 512))
        self.assertTrue(np.isfinite(large).all())

    def test_default_256_captures_pairing_determinism_and_clean_truth(self):
        base = ROOT/'validation/tmp'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            subprocess.run(['node', '--input-type=module', '-e', JOINED_FIXTURE, str(root), '8'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            with patch('synthetic._corrupt_sequence', side_effect=AssertionError('Must stream crops')):
                first = generate([root/'clean'], root/'generated', severity=0)
            self.assertEqual(first['format'], 'neural-qr-dataset')
            self.assertEqual(len(first['samples']), 256)
            self.assertEqual(first['generator']['frames'], 8)
            self.assertEqual(first['generator']['burstsPerTarget'], 8)
            self.assertEqual(first['generator']['captureResolution']['min_scale'], .5)
            groups = {}
            dataset = Dataset([root/'generated'])
            clean = Dataset([root/'clean'])
            for sample in dataset.samples:
                row = sample.metadata
                groups.setdefault(row['synthetic']['phaseGroupId'], []).append(row)
                self.assertEqual(row['sessionId'], 'synthetic-receiver')
                self.assertEqual(row['senderSessionId'], 'synthetic-sender')
                self.assertEqual(row['split'], 'train')
                self.assertEqual(row['devicePair'], {'sender':'phone', 'receiver':'pc'})
                pixels, labels, mask = sample.tensors()
                self.assertEqual(pixels.shape, (3, 16, 16))
                expected = sample.read('expected', 'u1').reshape(8, 8)
                np.testing.assert_array_equal(canonical_labels(labels, mask, row['info']['phase']), expected)
                source = next(s for s in clean.samples if s.id == row['synthetic']['parentId'])
                np.testing.assert_array_equal(pixels, source.tensors()[0])
            self.assertEqual(len(groups), 8)
            for group in groups.values():
                self.assertEqual(sorted(row['synthetic']['timelineFrame'] for row in group), list(range(32)))
                self.assertEqual(len({row['burstId'] for row in group}), 4)
            first = generate([root/'clean'], root/'small', frames=2, bursts_per_target=1, severity=.35)
            repeated = generate([root/'clean'], root/'repeat', frames=2, bursts_per_target=1, severity=.35)
            self.assertEqual(first['samples'], repeated['samples'])
            self.assertTrue(any(row['hashes']['baseline'] != next(s.metadata['hashes']['baseline']
                for s in clean.samples if s.id == row['synthetic']['parentId']) for row in first['samples']))
            with self.assertRaisesRegex(ValueError, 'max-output-mib'):
                generate([root/'clean'], root/'too-large', bursts_per_target=100000, max_output_mib=1)
            self.assertFalse((root/'too-large').exists())
            with self.assertRaisesRegex(ValueError, 'four coherent phases'):
                generate([root/'clean'], root/'ungrouped', phase_grouped=False)


if __name__ == '__main__':
    unittest.main()
