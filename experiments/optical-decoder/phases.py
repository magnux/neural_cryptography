"""The four-phase G64 decoder's label convention, shared by offline pipelines.

The canonical JavaScript wire codec uses these public XOR palette mappings for
the first four 16-color phases. Only data cells change color; pilot cells do not.
"""

import numpy as np


PHASE_MASKS = (0, 15, 7, 8)


def phase_neutral_info(info):
    """Validate the model's optical profile and remove only the phase index."""
    if (not isinstance(info, dict) or info.get("size") != 64 or info.get("colors") != 16
            or info.get("phases") != 4 or info.get("permute") is not True
            or type(info.get("phase")) is not int or info["phase"] not in range(4)):
        raise ValueError("Phase grouping requires G64, 16 colors, four permuted phases")
    return {key: value for key, value in info.items() if key != "phase"}


def canonical_labels(labels, mask, phase):
    """Undo one known palette assignment on data cells, retaining pilot labels."""
    labels, mask = np.asarray(labels), np.asarray(mask)
    if (type(phase) is not int or phase not in range(4) or labels.shape != mask.shape
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
            or np.any(labels >= 16) or np.any((mask != 0) & (mask != 1))):
        raise ValueError("Invalid phase labels or data-cell mask")
    result = labels.copy()
    result[mask.astype(bool)] ^= PHASE_MASKS[phase]
    return result
