"""The current optical tensor contract; no camera corrector or protocol implementation."""

from pathlib import Path
import hashlib

import numpy as np


FORMAT = "neural-corrector"
VERSION = 1
SIZES = (64, 128)
MULTIRES_SIZES = (8, 16, 32, 64, 128, 256)
FOUR_PIXEL_PHASE_SIZES = (16, 32, 64, 128)
PIXELS_PER_CELL = 4
CHANNELS = 3
MAX_FRAMES = 64
FRAME_COUNTS = tuple(range(1, MAX_FRAMES + 1))
CLASSES = 16
CAPTURE_INPUT_MIN = -1.5
CAPTURE_INPUT_MAX = 1.5
NETWORK_INPUT_MIN = -1.0
NETWORK_INPUT_MAX = 1.0
# Default single-frame full-resolution tube; exports record their actual count.
PARAMETER_COUNT = 140_976
INPUT_NAME = "pixels"
OUTPUT_NAME = "logits"
# One deployment policy for native ONNX Runtime, browser and WASM checks.
# Float32 backends may accumulate/reorder operations differently, especially
# through trained residual/attention blocks. Every logit must pass this bound.
ATOL = 1e-4
RTOL = 1e-3


def shapes(size: int, frames: int = 1, *, presence: bool = False,
           pixels_per_cell: int = PIXELS_PER_CELL) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(pixels_per_cell) is not int or pixels_per_cell not in (2, 4):
        raise ValueError("Use 2 pixels per cell for multires or 4 for U-tube")
    sizes = MULTIRES_SIZES if pixels_per_cell == 2 else (FOUR_PIXEL_PHASE_SIZES if presence else SIZES)
    if isinstance(size, bool) or not isinstance(size, int) or size not in sizes:
        raise ValueError(f"Unsupported grid size for {pixels_per_cell} pixels per cell: {size}")
    if type(frames) is not int or frames not in FRAME_COUNTS:
        raise ValueError(f"The corrector supports 1 to {MAX_FRAMES} input frames")
    if type(presence) is not bool or presence and frames != 8:
        raise ValueError("Presence-plane inputs require eight slots and a supported phase-grouped grid")
    if pixels_per_cell == 2 and not presence:
        raise ValueError("Multires requires phase-paired RGB and presence planes")
    channels = (CHANNELS + int(presence)) * frames
    return (1, channels, size * pixels_per_cell, size * pixels_per_cell), (1, CLASSES, size, size)


def normalize_rgb(calibrated_hwc: np.ndarray, *, pixels_per_cell: int = PIXELS_PER_CELL) -> np.ndarray:
    """Calibrated RGB [H,W,3] in 0..255 units -> float32 [1,3,H,W].

    Calibration may produce values outside the camera byte range. Preserve
    that overshoot in the stored crop, matching the receiver's [-1.5,1.5]
    capture bound. The model separately clips every source to [-1,1] immediately
    before its learned layers; that operation is also part of the ONNX graph.
    Camera homography and affine color fitting remain the receiver's job.
    Float64 intermediate arithmetic matches JavaScript Number operations;
    the final tensor is float32.
    """
    rgb = np.asarray(calibrated_hwc)
    if rgb.ndim != 3 or rgb.shape[2] != CHANNELS or rgb.shape[0] != rgb.shape[1]:
        raise ValueError("Expected a square calibrated HWC RGB crop")
    if type(pixels_per_cell) is not int or pixels_per_cell not in (2, 4):
        raise ValueError("Use 2 pixels per cell for multires or 4 for U-tube")
    sizes = MULTIRES_SIZES if pixels_per_cell == 2 else FOUR_PIXEL_PHASE_SIZES
    if rgb.shape[0] not in tuple(s * pixels_per_cell for s in sizes):
        raise ValueError(f"Unsupported calibrated crop for {pixels_per_cell} pixels per cell")
    if (not np.issubdtype(rgb.dtype, np.number) or np.iscomplexobj(rgb)
            or not np.isfinite(rgb).all()):
        raise ValueError("RGB samples must be finite real numeric values")
    normalized = np.clip(rgb.astype(np.float64) / 127.5 - 1.0,
                         CAPTURE_INPUT_MIN, CAPTURE_INPUT_MAX)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def fixture_input(size: int, seed: int, frames: int = 1, *, presence: bool = False,
                  available_frames: int | None = None,
                  pixels_per_cell: int = PIXELS_PER_CELL) -> np.ndarray:
    """Public diagnostic input with variation in space and all RGB channels.

    This input has no transmitted payload or symbol labels. It tests numerical
    inference agreement only; it cannot measure a corrector's recovery accuracy.
    Retained calibrated overshoot exercises the model's final [-1,1] input clip.
    """
    input_shape, _ = shapes(size, frames, presence=presence, pixels_per_cell=pixels_per_cell)
    available = (4 if presence else frames) if available_frames is None else available_frames
    if type(available) is not int or (presence and available not in (4, 5, 6, 7, 8)) or \
            (not presence and available != frames):
        raise ValueError("Fixture observation count must match the current input layout")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
    side = input_shape[2]
    y, x = np.mgrid[:side, :side]
    values = []
    for index in range(frames):
        texture = rng.uniform(-45.0, 45.0, (side, side, CHANNELS))
        rgb = texture + np.stack((255.0 * x / (side - 1), 255.0 * y / (side - 1),
                                  127.5 + 100.0 * np.sin((x + y) / 11.0 + index*.2)), axis=-1)
        values.append(normalize_rgb(rgb, pixels_per_cell=pixels_per_cell))
    rgb = np.ascontiguousarray(np.concatenate(values, axis=1), dtype=np.float32)
    if not presence:
        return rgb
    # Fixed phase pairs: one observation in every pair, then random extra
    # phases. The fixture tests packing/inference, not a known optical payload.
    packed = np.zeros(input_shape, dtype=np.float32)
    slots = [2*phase + int(rng.integers(2)) for phase in range(4)]
    for phase in rng.choice(4, size=available-4, replace=False):
        slots.append(slots[int(phase)] ^ 1)
    for slot in slots:
        packed[:, 3*slot:3*slot+3] = rgb[:, 3*slot:3*slot+3]
        packed[:, CHANNELS*frames+slot] = 1.
    return packed


def write_f32(path: Path, tensor: np.ndarray) -> dict:
    values = np.asarray(tensor)
    if not np.isfinite(values).all():
        raise ValueError("Refusing to serialize a non-finite tensor")
    path.write_bytes(np.ascontiguousarray(values, dtype="<f4").tobytes())
    return describe_file(path)


def describe_file(path: Path) -> dict:
    data = path.read_bytes()
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def parity_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if not expected.size or expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Inference outputs must be finite and have identical shapes")
    reference = expected.astype(np.float64)
    output = actual.astype(np.float64)
    delta = np.abs(reference - output)
    allowed = ATOL + RTOL * np.abs(reference)
    result = {
        "passed": bool(np.all(delta <= allowed)),
        "atol": ATOL,
        "rtol": RTOL,
        "maxAbsError": float(delta.max()),
        "meanAbsError": float(delta.mean()),
        "mismatchedValues": int(np.count_nonzero(delta > allowed)),
        "values": int(expected.size),
    }
    if expected.ndim == 4 and expected.shape[1] == CLASSES:
        # Diagnostic only: numerical agreement can flip an argmax at a near tie.
        # These public inputs contain no transmitted payload/known palette labels;
        # neither argmax agreement nor this check claims CRC recovery accuracy.
        changed = reference.argmax(axis=1) != output.argmax(axis=1)
        top_two = np.partition(reference, -2, axis=1)[:, -2:]
        margins = top_two[:, 1] - top_two[:, 0]
        def probabilities(logits):
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            return weights / weights.sum(axis=1, keepdims=True)
        result["decisions"] = {
            "cells": int(changed.size),
            "argmaxChangedCells": int(np.count_nonzero(changed)),
            "maxReferenceMarginAtChangedCells": float(margins[changed].max()) if changed.any() else 0.0,
            "maxProbabilityError": float(np.abs(probabilities(reference) - probabilities(output)).max()),
        }
    return result
