"""Synthetic kernels and the current collection/training format boundary."""

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
import {createSender,floatBytes,NEURAL_QR_VERSION} from './apps/color-optical-bench/dist/js/dataset/contract.js';
import {gridLayout} from './apps/color-optical-bench/dist/js/grid.js';
import {targets} from './scripts/corrector-dataset.mjs';
const root=process.argv[1],size=Number(process.argv[2]),folder=join(root,'joined');
await mkdir(folder,{recursive:true});
const {manifest,frames}=await createSender({size,seed:71,sessionId:'synthetic-sender'});
const frame=frames[0],mask=new Uint8Array(size*size);
for(const cell of gridLayout(size,16,8).cells)mask[cell]=1;
const sample={id:'real-frame',sessionId:'synthetic-receiver',senderSessionId:manifest.sessionId,
 payloadId:manifest.payloadId,split:'train',source:'real',size,colors:16,pixelsPerCell:4,
 inputShape:[3,4*size,4*size],info:frame.info,burstId:'real-burst',timestamp:1,
 devicePair:{sender:'phone',receiver:'pc'},hashes:{}};
const data={input:floatBytes(new Float32Array(3*(4*size)**2)),labels:frame.symbols,
 mask,baseline:floatBytes(new Float32Array(16*size*size)),expected:frame.chunk};
for(const [key,value] of Object.entries(data)){
 sample[key]=key+'.bin';sample.hashes[key]=createHash('sha256').update(value).digest('hex');
 await writeFile(join(folder,sample[key]),value);
}
await writeFile(join(folder,'manifest.json'),JSON.stringify({format:'neural-qr-dataset',version:NEURAL_QR_VERSION,samples:[sample]}));
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

    def test_four_pixel_pipeline_keeps_phase_targets_and_counts_actual_real_captures(self):
        base = ROOT/'validation/tmp'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            subprocess.run(['node', '--input-type=module', '-e', JOINED_FIXTURE, str(root), '16'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            joined = json.loads((root/'joined/manifest.json').read_text())
            clean = json.loads((root/'clean/manifest.json').read_text())
            self.assertEqual(joined['version'], 2)
            self.assertEqual(clean['version'], 2)
            self.assertEqual(len(clean['samples']), 4)
            self.assertEqual({row['info']['phase'] for row in clean['samples']}, {0, 1, 2, 3})
            for row in clean['samples']:
                self.assertEqual(row['pixelsPerCell'], 4)
                self.assertEqual(row['inputShape'], [3, 64, 64])
                self.assertEqual(row['sessionId'], 'synthetic-receiver')
                self.assertEqual(row['senderSessionId'], 'synthetic-sender')
                self.assertEqual(row['devicePair'], {'sender': 'phone', 'receiver': 'pc'})
                self.assertEqual((root/'clean'/row['input']).stat().st_size, 3*64*64*4)
                labels = np.fromfile(root/'clean'/row['labels'], dtype='u1').reshape(16, 16)
                mask = np.fromfile(root/'clean'/row['mask'], dtype='u1').reshape(16, 16)
                target = np.fromfile(root/'clean'/row['expected'], dtype='u1').reshape(16, 16)
                np.testing.assert_array_equal(canonical_labels(labels, mask, row['info']['phase']), target)
            for folder in ('joined', 'clean'):
                dataset = Dataset([root/folder])
                self.assertEqual(dataset.dataset_version, 2)
                self.assertEqual(dataset.pixels_per_cell, 4)
                for sample in dataset.samples:
                    self.assertEqual(sample.tensors()[0].shape, (3, 64, 64))
            plan = generate([root/'clean'], root/'generated', paired_real=[root/'joined'],
                            frames=2, recipe_name='w-tube', plan_only=True)
            self.assertFalse((root/'generated').exists())
            self.assertEqual((plan['targets'], plan['captures']), (1, 8))
            self.assertEqual(plan['recipe']['pairedCounts']['realCaptures'], 1)
            self.assertEqual(plan['recipe']['pairedCounts']['achievedRatio'], 8)
            self.assertEqual(plan['recipe']['severityMin'], .5)
            self.assertEqual(plan['recipe']['severity'], 1)
            self.assertEqual(plan['recipe']['cameraGeometry']['footprint_min'], 96.)
            self.assertEqual(plan['recipe']['cameraGeometry']['coverage']['targets'], 1)
            self.assertNotIn('captureResolution', plan['recipe'])
            cli = subprocess.run([sys.executable, str(EXPERIMENT/'synthetic.py'),
                '--clean', str(root/'clean'), '--paired-real', str(root/'joined'),
                '--out', str(root/'planned-only'), '--recipe', 'w-tube', '--frames', '2',
                '--white-balance-stops', '.1', '--plan'], check=True, capture_output=True, text=True)
            cli_plan = json.loads(cli.stdout)
            self.assertEqual(cli_plan['captures'], plan['captures'])
            self.assertEqual(cli_plan['recipe']['photometric']['white_balance_stops'], .1)
            self.assertEqual(cli_plan['recipe']['photometric']['glare_strength'], .8)
            self.assertEqual(cli_plan['recipe']['photometric']['structured_light_probability'], .65)
            self.assertFalse((root/'planned-only').exists())
            with self.assertRaisesRegex(ValueError, 'Choose paired-real'):
                generate([root/'clean'], root/'bad', paired_real=[root/'joined'],
                         bursts_per_target=2, plan_only=True)
            self.assertFalse((root/'bad').exists())
            generated = generate([root/'clean'], root/'generated', paired_real=[root/'joined'],
                                 frames=2, severity=0, seed=17)
            self.assertEqual(generated['version'], 2)
            self.assertEqual(len(generated['samples']), 8)
            self.assertEqual({s['info']['phase'] for s in generated['samples']}, {0, 1, 2, 3})
            self.assertEqual(len({s['synthetic']['phaseGroupId'] for s in generated['samples']}), 1)
            self.assertEqual(len({s['synthetic']['seed'] for s in generated['samples']}), 1)
            clean_by_phase = {s.metadata['info']['phase']: s for s in Dataset([root/'clean']).samples}
            for sample in Dataset([root/'generated']).samples:
                parent = clean_by_phase[sample.metadata['info']['phase']]
                for observed, expected in zip(sample.tensors(), parent.tensors()):
                    np.testing.assert_array_equal(observed, expected)
                self.assertEqual(sample.metadata['sessionId'], 'synthetic-receiver')
                self.assertEqual(sample.metadata['senderSessionId'], 'synthetic-sender')
            # Equal datasets with a different split may never supply counts.
            joined['samples'][0]['split'] = 'validation'
            (root/'joined/manifest.json').write_text(json.dumps(joined))
            with self.assertRaisesRegex(ValueError, 'must match exactly one'):
                generate([root/'clean'], root/'wrong-split', paired_real=[root/'joined'], plan_only=True)
            self.assertFalse((root/'wrong-split').exists())

    def test_four_pixel_supported_sizes_and_physics_recipe(self):
        for g in (16, 32, 64, 128):
            clean = np.zeros((4, 3, 4*g, 4*g), dtype=np.float32)
            iterator = synthetic._iter_corrupt_sequence(clean, np.arange(4), seed=1,
                                                        severity=0, pixels_per_cell=4)
            np.testing.assert_array_equal(next(iterator), clean[0])
            iterator.close()
        defaults = synthetic.recipe_photometric('standard')
        harder = synthetic.recipe_photometric('w-tube')
        self.assertGreater(harder.motion_probability, defaults.motion_probability)
        self.assertGreater(harder.rolling_shutter_stops, defaults.rolling_shutter_stops)
        self.assertGreater(harder.glare_strength, defaults.glare_strength)
        with self.assertRaisesRegex(ValueError, 'recipe must'):
            synthetic.recipe_photometric('unknown')



if __name__ == '__main__':
    unittest.main()

