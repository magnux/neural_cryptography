"""The current M1 tensor contract; no camera decoder or protocol implementation."""

from pathlib import Path
import hashlib

import numpy as np


FORMAT = "optical-decoder-m1"
VERSION = 1
SIZES = (64, 128)
PIXELS_PER_CELL = 4
CHANNELS = 3
CLASSES = 16
PARAMETER_COUNT = 121_792
INPUT_NAME = "pixels"
OUTPUT_NAME = "logits"
# Native Python ONNX Runtime thresholds; the browser uses its documented,
# slightly wider thresholds to accommodate different execution backends.
ATOL = 2e-5
RTOL = 2e-4


def shapes(size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(size, bool) or not isinstance(size, int) or size not in SIZES:
        raise ValueError("M1 supports physical grid sizes 64 and 128 only")
    return (1, CHANNELS, size * PIXELS_PER_CELL, size * PIXELS_PER_CELL), (1, CLASSES, size, size)


def normalize_rgb(calibrated_hwc: np.ndarray) -> np.ndarray:
    """Calibrated RGB [H,W,3] in 0..255 units -> float32 [1,3,H,W].

    Calibration may produce values outside the camera byte range. Do not clip
    to 0..255 first: match the receiver's final [-1.5,1.5] clamp exactly. Camera
    homography and affine color fitting remain the JavaScript receiver's job.
    Float64 intermediate arithmetic matches JavaScript Number operations;
    the final tensor is float32.
    """
    rgb = np.asarray(calibrated_hwc)
    if rgb.ndim != 3 or rgb.shape[2] != CHANNELS or rgb.shape[0] != rgb.shape[1]:
        raise ValueError("Expected a square calibrated HWC RGB crop")
    if rgb.shape[0] not in tuple(s * PIXELS_PER_CELL for s in SIZES):
        raise ValueError("Expected a 256x256 or 512x512 crop")
    if (not np.issubdtype(rgb.dtype, np.number) or np.iscomplexobj(rgb)
            or not np.isfinite(rgb).all()):
        raise ValueError("RGB samples must be finite real numeric values")
    normalized = np.clip(rgb.astype(np.float64) / 127.5 - 1.0, -1.5, 1.5)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def fixture_input(size: int, seed: int) -> np.ndarray:
    """Public diagnostic input with variation in space and all RGB channels.

    This input has no transmitted payload or symbol labels. It tests numerical
    inference agreement only; it cannot measure a decoder's recovery accuracy.
    """
    input_shape, _ = shapes(size)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
    side = input_shape[2]
    y, x = np.mgrid[:side, :side]
    texture = rng.uniform(-45.0, 45.0, (side, side, CHANNELS))
    rgb = texture + np.stack((255.0 * x / (side - 1), 255.0 * y / (side - 1),
                              127.5 + 100.0 * np.sin((x + y) / 11.0)), axis=-1)
    return normalize_rgb(rgb)


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
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Inference outputs must be finite and have identical shapes")
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    allowed = ATOL + RTOL * np.abs(expected.astype(np.float64))
    return {
        "passed": bool(np.all(delta <= allowed)),
        "atol": ATOL,
        "rtol": RTOL,
        "maxAbsError": float(delta.max()),
        "meanAbsError": float(delta.mean()),
        "mismatchedValues": int(np.count_nonzero(delta > allowed)),
        "values": int(expected.size),
    }
