"""Camera-scale draws for synthetic rectified crops.

The stored raster is always four pixels per cell. Camera evidence is not: its
projected footprint is sampled independently of G, or from measurements of the
same real session/target/split. No labels or image contents enter these draws.
"""

from dataclasses import dataclass
import math

import numpy as np


MIN_CAMERA_SIDE = 32.
MAX_CAMERA_SIDE = 2048.
_GRIDS = (16, 32, 64, 128)
_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class CameraGeometryConfig:
    """Fallback footprint and camera-pixel effects, before rectification.

    Footprint bounds describe missing-metadata draws, not empirical filtering.
    The separate 32..2048 camera-pixel guard is recorded whenever it clips a draw.
    An easy draw increases the footprint and halves photometric/geometric effect
    strength; it does not replace the ordinary examples or their target pairing.
    """

    footprint_min: float = 96.
    footprint_max: float = 640.
    easy_fraction: float = .15
    blur_sigma_min: float = .25
    blur_sigma_max: float = .85
    registration_pixels: float = .35
    affine_pixels: float = .45

    def __post_init__(self):
        for name, low, high in (
                ("footprint_min", MIN_CAMERA_SIDE, MAX_CAMERA_SIDE),
                ("footprint_max", MIN_CAMERA_SIDE, MAX_CAMERA_SIDE),
                ("easy_fraction", 0., 1.), ("blur_sigma_min", 0., 3.),
                ("blur_sigma_max", 0., 3.), ("registration_pixels", 0., 2.),
                ("affine_pixels", 0., 1.)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be finite and in [{low:g}, {high:g}]")
        if self.footprint_min > self.footprint_max:
            raise ValueError("footprint_min must not exceed footprint_max")
        if self.blur_sigma_min > self.blur_sigma_max:
            raise ValueError("blur_sigma_min must not exceed blur_sigma_max")


def camera_target_key(row):
    """Exact paired identity; never merge measurements from another split."""
    key = tuple(row.get(name) for name in ("sessionId", "payloadId", "split"))
    if any(not isinstance(value, str) or not value for value in key) or key[2] not in _SPLITS:
        raise ValueError("Camera geometry requires session/payload/split identity")
    return key


def _positive_measurement(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def build_camera_pools(samples):
    """Index actual source-camera pixels/cell, preserving each observation.

    Missing or invalid optional statistics use the explicit footprint fallback.
    Recording multiplicity within a target is meaningful: it represents the
    measured working-distance distribution. Sorting makes draws file-order
    independent, while exact target keys prevent borrowing across sessions.
    """
    pools, grids, owners = {}, {}, {"sessionId": {}, "payloadId": {}}
    total, observed = 0, 0
    for sample in samples:
        row = sample.metadata
        if row.get("source") != "real":
            raise ValueError("Camera geometry pools require real capture metadata")
        key = camera_target_key(row)
        grid = row.get("size")
        if type(grid) is not int or grid not in _GRIDS:
            raise ValueError("Camera geometry requires a supported W-tube grid")
        if grids.setdefault(key, grid) != grid:
            raise ValueError("Paired camera target has conflicting grid sizes")
        for name in owners:
            previous = owners[name].setdefault(row[name], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Camera geometry {name} crosses dataset splits")
        values = pools.setdefault(key, [])
        provenance = row.get("provenance")
        stats = provenance.get("stats") if isinstance(provenance, dict) else None
        value = stats.get("pixelsPerCell") if isinstance(stats, dict) else None
        total += 1
        if _positive_measurement(value):
            values.append(float(value))
            observed += 1
    pools = {key: tuple(sorted(values)) for key, values in sorted(pools.items())}
    covered = sum(bool(values) for values in pools.values())
    return pools, {"captures": total, "observedScaleCaptures": observed,
                   "missingOrInvalidScaleCaptures": total-observed,
                   "targets": len(pools), "observedScaleTargets": covered,
                   "fallbackTargets": len(pools)-covered,
                   "measurement": "provenance.stats.pixelsPerCell",
                   "pooling": "same-session-target-split"}


def sample_camera_geometry(grid, seed, severity, config=None, *,
                           observed_pixels_per_cell=(), source=None):
    """Draw one reproducible physical geometry shared by a four-phase burst.

    ``cameraToCrop`` converts camera-pixel distances to stored-raster pixels.
    ``affineCamera`` is a row-major 2x2 displacement field evaluated at normalized
    crop coordinates [-1,1]; each coefficient has camera-pixel units. These are
    small residual rectification errors, not changes to the clean target grid.
    ``effectStrength`` is a multiplier (1 or .5), separate from severity.
    """
    config = config or CameraGeometryConfig()
    if not isinstance(config, CameraGeometryConfig):
        raise ValueError("Expected CameraGeometryConfig")
    if type(grid) is not int or grid not in _GRIDS:
        raise ValueError("Camera geometry requires G16/G32/G64/G128")
    if type(seed) is not int or not 0 <= seed <= 0xffffffff:
        raise ValueError("Camera geometry seed must be uint32")
    if (isinstance(severity, bool) or not isinstance(severity, (int, float))
            or not math.isfinite(severity) or not 0 <= severity <= 1):
        raise ValueError("Camera geometry severity must be finite and in [0, 1]")
    values = tuple(observed_pixels_per_cell)
    if any(not _positive_measurement(value) for value in values):
        raise ValueError("Observed camera pixels/cell must be finite and positive")
    if source is not None and (not isinstance(source, str) or not source or len(source) > 128):
        raise ValueError("Camera geometry source must be a short nonempty string")
    side = 4*grid
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x43414d45]))
    observed = float(rng.choice(sorted(values))) if values else None
    footprint = (observed*grid*float(rng.uniform(.85, 1.15)) if observed is not None
                 else float(np.exp(rng.uniform(np.log(config.footprint_min),
                                               np.log(config.footprint_max)))))
    easy = bool(rng.random() < config.easy_fraction)
    if easy:
        footprint *= float(rng.uniform(1.25, 1.75))
    unclamped = footprint
    footprint = float(np.clip(footprint, MIN_CAMERA_SIDE, MAX_CAMERA_SIDE))
    multiplier = .5 if easy else 1.
    strength = severity*multiplier
    result = {
        "cameraSide": footprint, "cameraPixelsPerCell": footprint/grid,
        "cameraToCrop": side/footprint,
        # More camera pixels than the stored crop need no larger output tensor.
        "samplingSide": max(8, min(side, round(footprint))),
        "blurSigmaCamera": float(rng.uniform(config.blur_sigma_min, config.blur_sigma_max)*strength),
        "samplingOffsetCamera": rng.uniform(-.5, .5, 2).tolist(),
        "registrationOffsetCamera": (rng.uniform(-config.registration_pixels,
                                                config.registration_pixels, 2)*strength).tolist(),
        "registrationDriftCamera": (rng.uniform(-.12, .12, 2)*strength).tolist(),
        "affineCamera": (rng.uniform(-config.affine_pixels, config.affine_pixels, 4)*strength).tolist(),
        "effectStrength": multiplier, "easy": easy,
        "source": (source or "paired-capture") if observed is not None else "fallback",
        "observedPixelsPerCell": observed, "unclampedCameraSide": unclamped,
        "footprintClamped": footprint != unclamped,
    }
    if severity == 0:
        result.update(cameraSide=float(side), cameraPixelsPerCell=4., cameraToCrop=1.,
                      samplingSide=side, blurSigmaCamera=0., samplingOffsetCamera=[0., 0.],
                      registrationOffsetCamera=[0., 0.], registrationDriftCamera=[0., 0.],
                      affineCamera=[0.]*4, effectStrength=0., easy=False, source="identity",
                      unclampedCameraSide=float(side), footprintClamped=False)
    return result
