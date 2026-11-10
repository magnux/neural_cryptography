"""RS-budget training weights preserve canonical bytes through augmentation."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses import rs_block_layout, rs_cell_weights, rs_loss_policy, rs_weighted_loss
from packet_replay import pack_data_cells

ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(1)


def error_case(counts, size=32, extra_mask=False):
    """Place distinct byte errors in chosen blocks of the canonical wire."""
    mask = torch.ones((size, size), dtype=torch.bool)
    if extra_mask:
        mask[0, 0] = False  # odd data count, shifted nibble pairing
    wire_bytes = int(mask.sum())//2
    lengths, eligible = rs_block_layout(wire_bytes)
    assert eligible and len(counts) == len(lengths)
    data_errors = torch.zeros(int(mask.sum()), dtype=torch.bool)
    for block, count in enumerate(counts):
        for column in range(count):
            byte = column*len(lengths)+block
            data_errors[2*byte] = True
    errors = torch.zeros_like(mask)
    errors[mask] = data_errors
    return errors, mask


class RSWeightedLossTests(unittest.TestCase):
    def test_packet_layout_and_interleave_match_canonical_js_encoder(self):
        script = """
import {packetLayout,encodePacket} from './scripts/lib/corrector-rs-replay.mjs';
import {rsEncode} from './apps/color-optical-bench/dist/js/fec.js';
import {crc32} from './apps/color-optical-bench/dist/js/format.js';
const rows=[];
for(const capacity of [17,128,255,256,511,512,1575,8192]){
 const layout=packetLayout(capacity,32);
 if(!layout.eligible){rows.push({layout});continue;}
 const payload=Uint8Array.from({length:layout.payloadBytes},(_,i)=>(i*13+7)%256);
 const data=new Uint8Array(payload.length+4);data.set(payload);
 new DataView(data.buffer).setUint32(payload.length,crc32(payload),false);
 let offset=0;const blocks=[];
 for(const length of layout.blockLengths){const n=length-32;blocks.push(Array.from(rsEncode(data.slice(offset,offset+n),32)));offset+=n;}
 rows.push({layout,wire:Array.from(encodePacket(payload,capacity,32)),blocks});
}
console.log(JSON.stringify(rows));
"""
        output = subprocess.run(["node", "--input-type=module", "-e", script],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        for row in json.loads(output.stdout):
            layout = row["layout"]
            lengths, eligible = rs_block_layout(layout["wireBytes"], 32)
            self.assertEqual(list(lengths), layout["blockLengths"])
            self.assertEqual(eligible, bool(layout["eligible"]))
            if eligible:
                count = len(lengths)
                expected = [row["blocks"][i % count][i//count] for i in range(layout["wireBytes"])]
                self.assertEqual(expected, row["wire"])

    def test_byte_pairing_excludes_pilots_and_counts_two_wrong_nibbles_once(self):
        errors, mask = error_case([12, 13, 0], size=32, extra_mask=True)
        weights = rs_cell_weights(errors, mask)
        self.assertEqual(weights[0].item(), 1.)
        self.assertEqual(weights[2].item(), 3.)
        # Add the other nibble in an already wrong byte: unchanged byte counts.
        data = errors[mask]
        data[1] = True
        errors[mask] = data
        self.assertTrue(torch.equal(rs_cell_weights(errors, mask), weights))
        # Reference-cell error and odd unpaired last data-cell error do not count.
        errors[~mask] = True
        errors.flatten()[-1] = True
        self.assertTrue(torch.equal(rs_cell_weights(errors, mask), weights))
        self.assertEqual(weights[-1].item(), 1.)
        symbols = np.arange(1024, dtype=np.int64).reshape(32, 32) % 16
        packed = pack_data_cells(symbols, mask.numpy())
        data_symbols = symbols[mask.numpy()]
        self.assertEqual(packed[0], (int(data_symbols[0]) << 4) | int(data_symbols[1]))

    def test_budget_thresholds_and_scaled_policy(self):
        for errors_count, expected in [(0, 1.), (12, 1.), (13, 3.), (16, 3.), (24, 3.), (25, 2.)]:
            errors, mask = error_case([errors_count, 0, 0])
            weights = rs_cell_weights(errors, mask)
            self.assertEqual(weights[0].item(), expected)
            self.assertEqual(weights[2].item(), 1.)
            self.assertFalse(weights.requires_grad)
        policy = rs_loss_policy(16, .3)
        self.assertEqual(policy["errorCountUpperBounds"], [6, 12])
        self.assertEqual(policy["unknownByteErrorBudget"], 8)
        self.assertEqual(policy["weightedFraction"], .3)

    def test_small_packet_falls_back_to_ce(self):
        logits = torch.randn(2, 16, 4, 4, requires_grad=True)
        labels = torch.zeros((2, 4, 4), dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        objective, raw = rs_weighted_loss(logits, labels, mask, 16)
        torch.testing.assert_close(objective, F.cross_entropy(logits, labels))
        torch.testing.assert_close(objective, raw.mean())
        objective.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighted_ce_normalizes_each_example_then_averages(self):
        errors, mask = error_case([13, 25, 0])
        labels = torch.zeros((2, 32, 32), dtype=torch.long)
        masks = torch.stack([mask, mask])
        # Different confidence by location makes the weighted and plain means differ.
        logits = torch.zeros((2, 16, 32, 32), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            logits[:, 0] = 2.
            logits[0, 1, errors] = 5.
            logits[1, 1, errors] = 8.
            masks[1, -1, -1] = False
        result, per_frame = rs_weighted_loss(logits, labels, masks, 16, mix=.25)
        cells = F.cross_entropy(logits, labels, reduction="none")
        expected_rows = []
        for i in range(2):
            wrong = logits[i].argmax(0) != labels[i]
            weights = rs_cell_weights(wrong, masks[i])
            base = cells[i][masks[i]].mean()
            weighted = (cells[i][masks[i]]*weights).sum()/weights.sum()
            expected_rows.append(.75*base+.25*weighted)
            torch.testing.assert_close(per_frame[i], base)
        torch.testing.assert_close(result, torch.stack(expected_rows).mean())
        self.assertNotAlmostEqual(result.item(), per_frame.mean().item(), places=4)
        result.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad.permute(0, 2, 3, 1)[~masks] == 0))

    def test_zero_mix_is_exact_plain_ce_and_gradient(self):
        torch.manual_seed(3)
        labels = torch.randint(0, 16, (2, 32, 32))
        masks = torch.rand(2, 32, 32) > .15
        logits = torch.randn(2, 16, 32, 32, dtype=torch.float64, requires_grad=True)
        objective, per_frame = rs_weighted_loss(logits, labels, masks, 16, mix=0)
        plain = (F.cross_entropy(logits, labels, reduction="none")*masks).flatten(1).sum(1)/masks.flatten(1).sum(1)
        torch.testing.assert_close(objective, plain.mean(), atol=0, rtol=0)
        torch.testing.assert_close(per_frame, plain, atol=0, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(objective, logits, retain_graph=True)[0],
                                   torch.autograd.grad(plain.mean(), logits)[0], atol=0, rtol=0)

    def test_uniform_block_category_normalizes_to_plain_ce(self):
        # There is no priority difference when all data cells get the same
        # weight: a single G16 block and uniformly severe G128 must reduce to CE.
        for size, error_count in ((16, 13), (128, 25)):
            with self.subTest(size=size):
                blocks, _ = rs_block_layout(size*size//2)
                errors, mask = error_case([error_count]*len(blocks), size=size)
                labels = torch.zeros((1, size, size), dtype=torch.long)
                logits = torch.zeros((1, 16, size, size), dtype=torch.float64)
                logits[:, 0] = 2.
                logits[:, 1, errors] = 5.
                logits.requires_grad_()
                objective, raw = rs_weighted_loss(logits, labels, mask[None], 16)
                plain = F.cross_entropy(logits, labels)
                weights = rs_cell_weights(errors, mask)
                self.assertEqual(weights.unique().tolist(), [3. if size == 16 else 2.])
                torch.testing.assert_close(objective, plain, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(raw.mean(), plain, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(torch.autograd.grad(objective, logits, retain_graph=True)[0],
                                           torch.autograd.grad(plain, logits)[0], atol=1e-12, rtol=1e-12)

    def test_loss_and_gradient_invariant_under_flips_and_phase_xor(self):
        errors, mask = error_case([13, 25, 0], extra_mask=True)
        canonical = torch.zeros((1, 16, 32, 32), dtype=torch.float64)
        canonical[:, 0] = 2.
        canonical[:, 1, errors] = 5.
        canonical.requires_grad_()
        labels = torch.zeros((1, 32, 32), dtype=torch.long)
        mask = mask[None]
        reference, _ = rs_weighted_loss(canonical, labels, mask, 16)
        expected_gradient, = torch.autograd.grad(reference, canonical)
        for horizontal in (False, True):
            for vertical in (False, True):
                dims = ([-1] if horizontal else [])+([-2] if vertical else [])
                for delta in (0, 15, 7, 8):
                    permutation = torch.arange(16) ^ delta
                    transformed = canonical.detach()[:, permutation]
                    target, transformed_mask = labels.clone(), mask.clone()
                    target[mask] ^= delta
                    if dims:
                        transformed, target, transformed_mask = (torch.flip(v, dims) for v in (transformed, target, transformed_mask))
                    transformed.requires_grad_()
                    actual, _ = rs_weighted_loss(transformed, target, transformed_mask, 16,
                                                  spatial_flips=np.array([[horizontal, vertical]], dtype=bool))
                    gradient, = torch.autograd.grad(actual, transformed)
                    if dims:
                        gradient = torch.flip(gradient, dims)
                    gradient = gradient[:, permutation]
                    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
                    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)

    def test_confident_wrong_prediction_still_has_finite_nonzero_gradient(self):
        errors, mask = error_case([13, 0, 0])
        logits = torch.zeros((1, 16, 32, 32), requires_grad=True)
        with torch.no_grad():
            logits[:, 0] = 10.
            logits[:, 1, errors] = 1000.
        loss, _ = rs_weighted_loss(logits, torch.zeros((1, 32, 32), dtype=torch.long), mask[None], 16)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[0, 1, errors] > 0))

    def test_invalid_inputs_rejected(self):
        for parity in (0, 1, 3, 66, True):
            with self.assertRaises(ValueError):
                rs_loss_policy(parity)
        for mix in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                rs_loss_policy(mix=mix)
        logits = torch.zeros((1, 16, 16, 16))
        labels = torch.zeros((1, 16, 16), dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "sixteen"):
            rs_weighted_loss(logits, labels, mask, 4)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            rs_weighted_loss(logits, labels, ~mask, 16)
        with self.assertRaisesRegex(ValueError, "spatial_flips"):
            rs_weighted_loss(logits, labels, mask, 16, spatial_flips=[[0, 0]])


if __name__ == "__main__":
    unittest.main()
