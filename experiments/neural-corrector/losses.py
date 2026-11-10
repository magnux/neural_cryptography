"""Training-only RS-budget weighting of ordinary masked cross-entropy.

Collection targets are random symbol matrices, not RS codewords. This objective
uses the diagnostic packet layout to prioritize blocks near the correction
budget; it does not enforce a syndrome or run a decoder. Hard prediction errors
select detached weights. Cross-entropy supplies all gradients.
"""

from functools import lru_cache
import math

import torch
from torch.nn import functional as F

from packet_replay import validate_rs_parity


def rs_loss_policy(parity=32, mix=.25):
    """Validate and describe the experimental loss, including scaled cutoffs."""
    validate_rs_parity(parity)
    if parity == 0:
        raise ValueError("RS-weighted CE requires nonzero rs-parity")
    if isinstance(mix, bool) or not isinstance(mix, (int, float)) or not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("RS loss mix must be a finite number from 0 to 1")
    budget = parity//2
    return {"name": "rs-weighted-ce", "weightedFraction": float(mix),
            "parityBytesPerBlock": parity, "unknownByteErrorBudget": budget,
            "errorCountUpperBounds": [3*budget//4, 3*budget//2],
            "blockWeights": [1., 3., 2.], "weightGradient": False,
            "normalization": "weighted-cell-mean-per-example/mean-over-examples",
            "coordinates": "undo-horizontal-and-vertical-flips-before-byte-grouping",
            "packing": "masked-data-row-major/high-nibble-first/drop-last-unpaired-cell",
            "blocks": "balanced-shortened/round-robin-interleaved",
            "unpairedCellWeight": 1., "ineligiblePacket": "ordinary-ce",
            "note": "RS-budget training heuristic on recorded symbol errors; not a differentiable decoder or a guarantee of packet recovery."}


@lru_cache(maxsize=128)
def rs_block_layout(wire_bytes, parity=32):
    """Mirror canonical JS packetLayout, returning immutable block lengths.

    Tests compare lengths and wire order with the canonical packet encoder.
    No encoding, decoding or codeword assumptions are introduced here.
    """
    validate_rs_parity(parity)
    if parity == 0:
        raise ValueError("RS-weighted CE requires nonzero rs-parity")
    if type(wire_bytes) is not int or not 1 <= wire_bytes <= 32768:
        raise ValueError("RS wire capacity must be 1..32768 bytes")
    count = (wire_bytes+254)//255
    short, extra = divmod(wire_bytes, count)
    lengths = tuple(short+int(i < extra) for i in range(count))
    eligible = wire_bytes-parity*count-4 > 0 and min(lengths) > parity
    return lengths, eligible


@lru_cache(maxsize=64)
def _wire_block_ids(wire_bytes, device):
    # Balanced lengths put the extra bytes in the first blocks, so canonical
    # column-wise round-robin interleaving is exactly offset % block_count.
    count = (wire_bytes+254)//255
    return torch.arange(wire_bytes, dtype=torch.long, device=device).remainder(count)


@torch.no_grad()
def rs_cell_weights(errors, mask, *, rs_parity=32):
    """Return canonical data-cell weights, excluding reference/pilot cells.

    Two adjacent *data* nibbles form one byte, even when a pilot lies between
    them spatially. Either wrong nibble makes that byte wrong, counted once.
    The odd last data nibble remains in CE at weight one but has no RS block.
    """
    policy = rs_loss_policy(rs_parity)
    if errors.ndim != 2 or errors.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("RS weights require a cell error matrix and matching boolean mask")
    data_errors = errors[mask].bool()
    wire_bytes = data_errors.numel()//2
    weights = torch.ones(data_errors.shape, dtype=torch.float32, device=errors.device)
    if wire_bytes == 0:
        return weights
    lengths, eligible = rs_block_layout(wire_bytes, rs_parity)
    if not eligible:
        return weights
    byte_errors = data_errors[:2*wire_bytes].reshape(wire_bytes, 2).any(dim=1)
    block_ids = _wire_block_ids(wire_bytes, str(errors.device))
    counts = torch.zeros(len(lengths), dtype=torch.long, device=errors.device)
    counts.scatter_add_(0, block_ids, byte_errors.long())
    low, high = policy["errorCountUpperBounds"]
    block_weights = torch.where(counts <= low, 1., torch.where(counts <= high, 3., 2.))
    weights[:2*wire_bytes] = block_weights[block_ids].repeat_interleave(2)
    return weights


def _flip_flags(spatial_flips, batch_size):
    if spatial_flips is None:
        return [(False, False)]*batch_size
    flags = torch.as_tensor(spatial_flips)
    if flags.dtype != torch.bool or tuple(flags.shape) != (batch_size, 2):
        raise ValueError("spatial_flips must be a boolean [batch,2] array: horizontal,vertical")
    return flags.tolist()


def rs_weighted_loss(logits, labels, mask, colors, *, rs_parity=32, mix=.25,
                     spatial_flips=None):
    """Return (optimized loss, unweighted CE per example).

    Spatial metadata restores only G×G CE/error/mask maps. RGB input and class
    logits stay untouched. A consistent phase XOR relabeling preserves correct
    versus incorrect predictions and therefore needs no inverse label mapping.
    """
    rs_loss_policy(rs_parity, mix)
    if (colors != 16 or logits.ndim != 4 or logits.shape[1] < colors
            or labels.shape != mask.shape or mask.dtype != torch.bool
            or tuple(labels.shape) != (logits.shape[0], *logits.shape[-2:])):
        raise ValueError("RS-weighted CE requires sixteen-color logits, labels and matching boolean masks")
    flags = _flip_flags(spatial_flips, logits.shape[0])
    counts = mask.flatten(1).sum(1)
    if not torch.all(counts > 0):
        raise ValueError("RS-weighted CE requires a nonempty data-cell mask per example")
    cells = F.cross_entropy(logits[:, :colors], labels, reduction="none")
    per_frame = (cells*mask).flatten(1).sum(1)/counts
    if mix == 0:
        return per_frame.mean(), per_frame
    errors = logits[:, :colors].detach().argmax(dim=1) != labels
    weighted = []
    for ce, wrong, data_mask, (horizontal, vertical) in zip(cells, errors, mask, flags):
        dimensions = ([1] if horizontal else [])+([0] if vertical else [])
        if dimensions:
            ce, wrong, data_mask = (torch.flip(value, dimensions) for value in (ce, wrong, data_mask))
        weights = rs_cell_weights(wrong, data_mask, rs_parity=rs_parity)
        weighted.append((ce[data_mask]*weights).sum()/weights.sum())
    weighted = torch.stack(weighted)
    return ((1-mix)*per_frame+mix*weighted).mean(), per_frame
