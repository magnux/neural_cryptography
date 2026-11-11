#!/usr/bin/env python3
"""Generate correlated corrupted crops from canonical JavaScript clean datasets.

This is a calibrated-crop approximation of screen/camera interference, not a
physical sensor simulator. Real held-out sessions decide whether it is useful.
All frames in a burst retain the source session/payload split and physical labels.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import DATASET_FORMAT, Dataset, digest, local_path, load_manifest
from phases import canonical_labels, phase_neutral_info
from camera_geometry import (CameraGeometryConfig, build_camera_pools,
                             camera_target_key, sample_camera_geometry)


CORRUPTION_KIND = "correlated-calibrated-crop-v9"
MULTIRES_CORRUPTION_KIND = "correlated-calibrated-crop-multires-v2"
W_TUBE_CORRUPTION_KIND = "correlated-camera-scale-w-tube-v3"
NEURAL_QR_DATASET_FORMAT = "neural-qr-dataset"
MULTIRES_SIZES = (8, 16, 32, 64, 128, 256)


class _Progress:
    """Periodic stderr lines keep CLI stdout available for the result summary."""

    def __init__(self, total):
        self.total = total
        self.started = self.last = time.monotonic()
        print(f"Synthetic captures: 0/{total}", file=sys.stderr, flush=True)

    def update(self, completed):
        now = time.monotonic()
        if completed == self.total or now-self.last >= 5:
            print(f"Synthetic captures: {completed}/{self.total} "
                  f"({completed/self.total:.1%}), {now-self.started:.1f}s elapsed",
                  file=sys.stderr, flush=True)
            self.last = now


def _prepare_baselines(node, bridge, out):
    """Relay live progress while retaining only a bounded tail for errors."""
    command = [node, str(bridge), "baselines", "--dataset", str(out)]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, bufsize=1) as process:
        tail = ""
        for line in process.stderr:
            print(line, end="", file=sys.stderr, flush=True)
            tail = (tail+line)[-2000:]
        code = process.wait()
    if code:
        raise RuntimeError(f"Canonical synthetic baseline preparation failed: {tail.strip() or f'exit {code}'}")


@dataclass(frozen=True)
class PhotometricConfig:
    """Maximum residual color distortions at severity=1, before camera noise."""

    white_balance_stops: float = .6
    exposure_stops: float = 1.5
    shade_opacity: float = .45
    shade_probability: float = .7
    rolling_shutter_stops: float = .65
    rolling_shutter_probability: float = .7
    rolling_shutter_skew: float = .5
    motion_probability: float = .7
    linear_blur_pixels: float = 2.5
    rotation_blur_degrees: float = .35
    zoom_blur_fraction: float = .006
    motion_samples: int = 7
    glare_probability: float = .65
    glare_strength: float = .6
    rgb_offset_probability: float = .6
    rgb_offset_pixels: float = .5
    structured_light_probability: float = 0.

    def __post_init__(self):
        for name, limit in (("white_balance_stops", 4), ("exposure_stops", 4),
                            ("shade_opacity", .95), ("shade_probability", 1),
                            ("rolling_shutter_stops", 3), ("rolling_shutter_probability", 1),
                            ("rolling_shutter_skew", .75), ("motion_probability", 1),
                            ("linear_blur_pixels", 12), ("rotation_blur_degrees", 5),
                            ("zoom_blur_fraction", .05), ("glare_probability", 1),
                            ("glare_strength", 1), ("rgb_offset_probability", 1),
                            ("rgb_offset_pixels", 1), ("structured_light_probability", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= limit:
                raise ValueError(f"{name.replace('_', '-')} must be finite and in [0, {limit}]")
        if type(self.motion_samples) is not int or self.motion_samples not in (3, 5, 7, 9, 11, 13, 15):
            raise ValueError("motion-samples must be an odd integer from 3 to 15")


@dataclass(frozen=True)
class CaptureResolutionConfig:
    """Effective crop resolution as a fraction of the stored dense crop side."""

    probability: float = .75
    min_scale: float = .25
    max_scale: float = 1.
    blur_sigma: float = .35

    def __post_init__(self):
        for name, minimum, maximum in (("probability", 0, 1), ("min_scale", .25, 1),
                                        ("max_scale", .25, 1), ("blur_sigma", 0, 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"capture {name} must be finite and in [{minimum}, {maximum}]")
        if self.min_scale > self.max_scale:
            raise ValueError("capture min_scale must not exceed max_scale")


def recipe_photometric(recipe):
    """Explicit recipes keep existing experiment defaults reproducible.

    The harder recipe still draws every effect probabilistically, and each
    complete phase group shares its motion/lighting trajectory. Increasing
    difficulty does not mean independently destroying every observation.
    """
    if recipe not in ("standard", "w-tube"):
        raise ValueError("recipe must be standard or w-tube")
    if recipe == "standard":
        return PhotometricConfig()
    return PhotometricConfig(white_balance_stops=.8, exposure_stops=1.75,
        shade_opacity=.65, shade_probability=.8, rolling_shutter_stops=.8,
        rolling_shutter_probability=.8, rolling_shutter_skew=.65,
        motion_probability=.8, linear_blur_pixels=3., rotation_blur_degrees=.45,
        zoom_blur_fraction=.008, glare_probability=.75, glare_strength=.8,
        rgb_offset_probability=.7, rgb_offset_pixels=.65, structured_light_probability=.65)


def resolution_config(config, pixels_per_cell):
    """Never simulate a lower capture resolution than one pixel per matrix cell."""
    if type(pixels_per_cell) is not int or pixels_per_cell not in (2, 4):
        raise ValueError("pixels-per-cell must be 2 or 4")
    config = config or CaptureResolutionConfig(min_scale=1/pixels_per_cell)
    if config.min_scale < 1/pixels_per_cell:
        raise ValueError(f"capture-min-scale must be at least {1/pixels_per_cell:g} "
                         f"for {pixels_per_cell} pixels/cell")
    return config


def _validate_crop_shape(clean, pixels_per_cell, grouped=False):
    sizes = MULTIRES_SIZES if pixels_per_cell == 2 else (16, 32, 64, 128)
    expected_prefix = (4, 3) if grouped else (3,)
    if (type(pixels_per_cell) is not int or pixels_per_cell not in (2, 4)
            or clean.ndim != len(expected_prefix)+2
            or clean.shape[:-2] != expected_prefix
            or clean.shape[-2] != clean.shape[-1]
            or clean.shape[-1] not in tuple(g*pixels_per_cell for g in sizes)
            or not np.isfinite(clean).all() or np.max(np.abs(clean)) > 1.50001):
        kind = "four finite normalized RGB crops" if grouped else "a finite normalized RGB crop"
        raise ValueError(f"Expected {kind} with a supported grid and {pixels_per_cell} pixels/cell")


def sample_capture_resolution(side, seed, severity, config, *, pixels_per_cell=4):
    """Choose one effective pixel count and blur for a whole burst.

    Working distance is independent of photometric severity. Severity zero still
    means exact clean identity. The separate stream leaves all other effects and
    their frame-to-frame randomness unchanged when this simulation is toggled.
    """
    config = resolution_config(config, pixels_per_cell)
    identity = {"side": side, "blurSigma": 0.}
    if severity == 0 or config.probability == 0:
        return identity
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5245534f]))
    if rng.random() >= config.probability:
        return identity
    low, high = round(side*config.min_scale), round(side*config.max_scale)
    return {"side": int(rng.integers(low, high+1)),
            "blurSigma": float(config.blur_sigma*rng.uniform(.5, 1.))}


def sample_photometric(rng, frames, severity, config):
    """Draw a payload-independent trajectory and optional shade/reflection shapes.

    Per-channel log2 gains have zero mean, separating color cast from exposure.
    Keeping 20% of bursts free of each gain effect preserves easier examples.
    The trajectory stays within the severity-scaled configured stop limits.
    """
    times = np.arange(frames, dtype=np.float32)
    phases = rng.uniform(0, 2*np.pi, 3)
    wb_limit = severity * config.white_balance_stops
    temperature, tint = rng.uniform(-1, 1, 2)
    base = np.array([temperature, .35*tint, -temperature], dtype=np.float32)
    base -= base.mean()
    base /= max(1., float(np.max(np.abs(base))))
    drift = np.stack([np.sin(times*.35+p) for p in phases], axis=1)
    drift -= drift.mean(axis=1, keepdims=True)
    drift /= max(1., float(np.max(np.abs(drift))))
    wb_stops = wb_limit * (.85*base[None] + .15*drift)
    if rng.random() >= .8:
        wb_stops[:] = 0
    exposure = severity*config.exposure_stops * (
        .85*rng.uniform(-1, 1) + .15*np.sin(times*.3+rng.uniform(0, 2*np.pi)))
    if rng.random() >= .8:
        exposure[:] = 0
    overlays = []
    if config.shade_opacity > 0 and severity > 0 and rng.random() < config.shade_probability:
        # One broad reflection/shadow, occasionally a smaller overlapping shape.
        # Rotating, feathering and clipping at crop edges avoids a fixed box cue.
        for index in range(1+int(rng.random() < .25)):
            area = rng.uniform(.18, .32) if index == 0 else rng.uniform(.06, .16)
            aspect = rng.uniform(.5, 2.)
            kind = "shadow" if rng.random() < .5 else "reflection"
            color = rng.uniform(0, .18, 3) if kind == "shadow" else rng.uniform(.75, 1., 3)
            overlays.append({"cx": float(rng.uniform(.15, .85)), "cy": float(rng.uniform(.15, .85)),
                "width": float(np.sqrt(area*aspect)), "height": float(np.sqrt(area/aspect)),
                "shape": "rectangle" if rng.random() < .65 else "ellipse", "kind": kind,
                "color": color.tolist(), "angle": float(rng.uniform(-np.pi, np.pi)),
                "opacity": float(severity*config.shade_opacity*rng.uniform(.35, .9)),
                "feather": float(rng.uniform(.08, .45)), "gradient": float(rng.uniform(-.4, .4)),
                "driftX": float(rng.uniform(-.01, .01)), "driftY": float(rng.uniform(-.01, .01)),
                "phase": float(rng.uniform(0, 2*np.pi))})
            if config.structured_light_probability > 0 and rng.random() < config.structured_light_probability:
                # A hand/room edge or window reflection is a broad field, not a
                # second independently placed corruption. Its extent is a
                # fraction of the screen and therefore spans more cells at G128.
                overlay = overlays[-1]
                overlay.update(shape="edge" if kind == "shadow" else "window",
                    width=float(rng.uniform(.55, 1.1)), height=float(rng.uniform(.5, 1.1)),
                    feather=float(rng.uniform(.15, .5)),
                    driftX=float(rng.uniform(-.035, .035)), driftY=float(rng.uniform(-.035, .035)),
                    angleDrift=float(rng.uniform(-.08, .08)))
                if kind == "reflection":
                    overlay.update(columns=int(rng.integers(2, 5)), rows=int(rng.integers(1, 4)),
                                   barWidth=float(rng.uniform(.035, .09)))
    return {"whiteBalanceGains": np.exp2(wb_stops).astype(np.float32),
            "exposureStops": exposure.astype(np.float32), "overlays": overlays}


def apply_color_exposure(image, white_balance_gains, exposure_stops):
    """Residual RGB gains with highlight/shadow clipping, not a raw-sensor model."""
    gains = torch.as_tensor(white_balance_gains, dtype=image.dtype, device=image.device)[:, None, None]
    return (image*gains*float(2.**exposure_stops)).clamp(0, 1)


def overlay_alpha(side, overlay, frame):
    """Rotated soft shape with a spatial opacity gradient and correlated drift."""
    axis = (torch.arange(side, dtype=torch.float32)+.5)/side
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    movement = np.sin(frame*.35+overlay["phase"])-np.sin(overlay["phase"])
    x = x-overlay["cx"]-overlay["driftX"]*movement
    y = y-overlay["cy"]-overlay["driftY"]*movement
    angle = overlay["angle"]+overlay.get("angleDrift", 0.)*movement
    sine, cosine = np.sin(angle), np.cos(angle)
    u = (cosine*x+sine*y)/(overlay["width"]/2)
    v = (-sine*x+cosine*y)/(overlay["height"]/2)
    shape = overlay["shape"]
    if shape == "edge":
        # A smooth half-plane shade may run off any edge of the crop. It does
        # not move symbols, and remains partly transmissive at maximum opacity.
        feathered = (.5-u/(2*overlay["feather"])).clamp(0, 1)
    else:
        radius = (torch.sqrt(u*u+v*v) if shape == "ellipse"
                  else torch.maximum(u.abs(), v.abs()))
        feathered = ((1-radius)/overlay["feather"]).clamp(0, 1)
        if shape == "window":
            # Dim mullions inside a tinted reflected window. Smooth bars avoid
            # introducing a pixel-size texture unrelated to screen geometry.
            dx = torch.remainder((u+1)*overlay["columns"]/2+.5, 1)-.5
            dy = torch.remainder((v+1)*overlay["rows"]/2+.5, 1)-.5
            bars = torch.maximum(torch.exp(-.5*(dx/overlay["barWidth"]).square()),
                                 torch.exp(-.5*(dy/overlay["barWidth"]).square()))
            feathered = feathered*(1-.8*bars)
    gradient = (1+overlay["gradient"]*u.clamp(-1, 1))/(1+abs(overlay["gradient"]))
    opacity = overlay["opacity"]*(.95+.05*np.sin(frame*.3+overlay["phase"]))
    return opacity*feathered*gradient


def apply_overlays(image, overlays, frame=0):
    """Shadows attenuate transmitted light; tinted reflections add light and clip."""
    for overlay in overlays:
        alpha = overlay_alpha(image.shape[-1], overlay, frame).to(image)
        color = torch.tensor(overlay["color"], dtype=image.dtype, device=image.device)[:, None, None]
        if overlay["kind"] == "shadow":
            image = image*(1-alpha*(1-color))
        else:
            image = (image+alpha*color).clamp(0, 1)
    return image


def sample_glare(rng, severity, config):
    """A shared optical glare spot with smooth motion and a broad veiling halo.

    This is a crop-space approximation, independent of transmitted symbols.
    The narrow bright core can saturate locally; its wider tinted halo reduces
    contrast without erasing the entire crop. Some bursts have no glare at all.
    """
    if severity == 0 or config.glare_strength == 0 or rng.random() >= config.glare_probability:
        return None
    radius = float(rng.uniform(.055, .13))
    aspect = float(rng.uniform(.65, 1.7))
    strength = float(severity*config.glare_strength*rng.uniform(.35, 1))
    tint = rng.uniform(.88, 1, 3)
    tint /= tint.max()
    glare = {"cx": float(rng.uniform(.05, .95)), "cy": float(rng.uniform(.05, .95)),
            "radiusX": radius*np.sqrt(aspect), "radiusY": radius/np.sqrt(aspect),
            "angle": float(rng.uniform(-np.pi, np.pi)), "haloScale": float(rng.uniform(2.5, 4)),
            "coreStrength": strength, "veilStrength": strength*float(rng.uniform(.35, .65)),
            "color": tint.tolist(), "driftX": float(rng.uniform(-.035, .035)),
            "driftY": float(rng.uniform(-.035, .035)), "phase": float(rng.uniform(0, 2*np.pi)),
            "velocity": float(rng.uniform(.18, .4))}
    if config.structured_light_probability > 0 and rng.random() < config.structured_light_probability:
        # Elongated light-source reflection, e.g. a ceiling strip, plus the
        # existing veiling halo. This replaces the spot rather than stacking
        # another bright layer on it. The burst shares its orientation/motion.
        glare.update(radiusX=float(rng.uniform(.17, .38)), radiusY=float(rng.uniform(.025, .06)),
                     haloScale=float(rng.uniform(2., 3.)), shape="streak")
    return glare


def glare_maps(side, glare, frame, *, device=None, dtype=torch.float32):
    """Return bounded additive core and veiling-alpha maps, centered on one ellipse."""
    axis = (torch.arange(side, device=device, dtype=dtype)+.5)/side
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    phase = glare["phase"]+frame*glare["velocity"]
    movement = np.sin(phase)-np.sin(glare["phase"])
    x = x-glare["cx"]-glare["driftX"]*movement
    y = y-glare["cy"]-glare["driftY"]*movement
    sine, cosine = np.sin(glare["angle"]), np.cos(glare["angle"])
    u = (cosine*x+sine*y)/glare["radiusX"]
    v = (-sine*x+cosine*y)/glare["radiusY"]
    squared_radius = u*u+v*v
    modulation = .9+.1*np.sin(phase)
    core = glare["coreStrength"]*modulation*torch.exp(-.5*squared_radius)
    veil = glare["veilStrength"]*modulation*torch.exp(-.5*squared_radius/glare["haloScale"]**2)
    return core, veil


def apply_glare(image, glare, frame=0):
    """Blend in broad scattered light, then add the localized highlight and clip."""
    if glare is None:
        return image
    core, veil = glare_maps(image.shape[-1], glare, frame, device=image.device, dtype=image.dtype)
    color = torch.as_tensor(glare["color"], device=image.device, dtype=image.dtype)[:, None, None]
    return (image*(1-veil)+color*(veil+core)).clamp(0, 1)


def sample_rolling_shutter(rng, severity, config):
    """Approximate refresh/PWM interference and residual scanline motion.

    This models the visible crop artifact, not a calibrated sensor readout. A
    shared drifting band pattern correlates captures without freezing the noise.
    """
    if severity == 0 or (config.rolling_shutter_stops == 0 and config.rolling_shutter_skew == 0):
        return None
    if rng.random() >= config.rolling_shutter_probability:
        return None
    return {"stops": float(severity*config.rolling_shutter_stops*rng.uniform(.25, 1)),
            "skew": float(severity*config.rolling_shutter_skew*rng.uniform(.3, 1)),
            "cycles": float(rng.uniform(1.5, 12)), "tilt": float(rng.uniform(-.12, .12)),
            "phase": float(rng.uniform(0, 2*np.pi)), "velocity": float(rng.uniform(.2, .9)),
            "color": rng.uniform(.8, 1, 3).tolist()}


def rolling_shutter_maps(side, shutter, frame):
    """Return RGB exposure gains and horizontal offsets in dense-input pixels.

    Offset magnitude never exceeds configured skew (at most .75 dense pixels).
    Together with registration jitter this stays below half a four-pixel cell.
    Centered exposure blur can mix neighboring cells without moving the target.
    """
    if shutter is None:
        return torch.ones((3, side, side)), torch.zeros((side, side))
    axis = torch.linspace(-1, 1, side)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    phase = shutter["phase"]+frame*shutter["velocity"]
    scan = y+shutter["tilt"]*x
    wave = torch.sin(np.pi*shutter["cycles"]*scan+phase)
    # Smooth PWM-like dark/light plateaus plus a sinusoidal component.
    bands = .6*wave+.4*torch.tanh(3*wave)
    color = torch.tensor(shutter["color"], dtype=torch.float32)[:, None, None]
    gains = torch.exp2(shutter["stops"]*color*bands)
    offsets = shutter["skew"]*(.65*y+.35*torch.sin(np.pi*y+phase))
    return gains, offsets


def sample_motion_blur(rng, severity, config):
    """Draw one camera-motion trajectory, with correlated exposure-to-exposure drift.

    Strengths describe the full centered exposure sweep, not displacement of the
    labeled matrix. Rotational blur is tangential about a pivot; zoom blur is
    radial. They can occur independently or together with linear camera motion.
    """
    strengths = (config.linear_blur_pixels, config.rotation_blur_degrees, config.zoom_blur_fraction)
    if severity == 0 or not any(strengths) or rng.random() >= config.motion_probability:
        return None
    enabled = rng.random(3) < np.array([.8, .65, .35])
    if not any(enabled[i] and strengths[i] for i in range(3)):
        enabled[int(rng.choice(np.flatnonzero(strengths)))] = True
    amplitudes = [float(severity*strength*rng.uniform(.25, 1)*active)
                  for strength, active in zip(strengths, enabled)]
    return {"linearPixels": amplitudes[0], "rotationDegrees": amplitudes[1]*int(rng.choice([-1, 1])),
            "zoomFraction": amplitudes[2]*int(rng.choice([-1, 1])),
            "angle": float(rng.uniform(0, 2*np.pi)), "angleDrift": float(rng.uniform(-.2, .2)),
            "cx": float(rng.uniform(.25, .75)), "cy": float(rng.uniform(.25, .75)),
            "phase": float(rng.uniform(0, 2*np.pi)), "velocity": float(rng.uniform(.2, .5)),
            "samples": config.motion_samples}


def sample_rgb_offset(rng, severity, config, motion):
    """Small relative exposure timing offsets along an existing motion path.

    This approximates color-fringed camera/display motion artifacts; it does not
    assume that real cameras universally expose their RGB channels sequentially.
    There is no static channel misregistration when camera motion is absent.
    """
    if (motion is None or severity == 0 or config.rgb_offset_pixels == 0
            or not any(motion[name] for name in ("linearPixels", "rotationDegrees", "zoomFraction"))
            or rng.random() >= config.rgb_offset_probability):
        return None
    offsets = rng.uniform(-1, 1, 3)
    offsets -= offsets.mean()
    offsets /= max(1., float(np.max(np.abs(offsets))))
    return {"timeOffsets": (offsets*.25).tolist(),
            "maxPixels": float(severity*config.rgb_offset_pixels*rng.uniform(.4, 1.))}


def motion_blur_grids(side, motion, frame, *, device=None, dtype=torch.float32,
                      time_offset=0.):
    """Centered exposure grids [S,H,W,2], in grid_sample normalized coordinates.

    Linear travel is in dense-crop pixels, rotation in degrees and zoom as a
    fractional scale sweep. A symmetric sample interval keeps the middle pose
    registered; the blur intentionally integrates evidence from adjacent cells.
    """
    axis = torch.linspace(-1, 1, side, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    times = torch.linspace(-.5, .5, motion["samples"], device=device, dtype=dtype)[:, None, None] + time_offset
    phase = motion["phase"]+frame*motion["velocity"]
    strength = .85+.15*np.sin(phase)
    angle = motion["angle"]+motion["angleDrift"]*np.sin(phase)
    theta = times*float(np.deg2rad(motion["rotationDegrees"])*strength)
    scale = 1+times*float(motion["zoomFraction"]*strength)
    cx, cy = 2*motion["cx"]-1, 2*motion["cy"]-1
    u, v = x-cx, y-cy
    sine, cosine = torch.sin(theta), torch.cos(theta)
    translation = times*float(2*motion["linearPixels"]*strength/(side-1))
    gx = cx+scale*(cosine*u-sine*v)+translation*float(np.cos(angle))
    gy = cy+scale*(sine*u+cosine*v)+translation*float(np.sin(angle))
    return torch.stack((gx, gy), dim=-1)


def rgb_motion_blur_grids(side, motion, rgb_offset, frame, *, device=None, dtype=torch.float32):
    """Return [RGB,S,H,W,2] exposure grids with centered, bounded color offsets.

    Each channel follows the same path with a slightly shifted exposure window.
    Centering relative displacements prevents a common image translation. The
    cap applies after centering, in dense-crop pixels, even under rotation/zoom.
    """
    common = motion_blur_grids(side, motion, frame, device=device, dtype=dtype)
    if rgb_offset is None or rgb_offset["maxPixels"] == 0:
        return common[None].expand(3, -1, -1, -1, -1)
    shifted = torch.stack([motion_blur_grids(side, motion, frame, device=device,
        dtype=dtype, time_offset=offset) for offset in rgb_offset["timeOffsets"]])
    delta = shifted-common[None]
    delta = delta-delta.mean(dim=0, keepdim=True)
    largest = float(torch.linalg.vector_norm(delta, dim=-1).max())*(side-1)/2
    delta *= min(1., rgb_offset["maxPixels"]/max(largest, 1e-12))
    return common[None]+delta


def apply_motion_blur(image, motion, frame=0, *, rgb_offset=None):
    """Integrate common or color-offset exposure windows once, with border extension."""
    if motion is None:
        return image
    if rgb_offset is not None and rgb_offset["maxPixels"] > 0:
        grids = rgb_motion_blur_grids(image.shape[-1], motion, rgb_offset, frame,
                                      device=image.device, dtype=image.dtype)
        channels, samples, side, _, _ = grids.shape
        inputs = image[:, None, None].expand(channels, samples, 1, side, side)
        observed = F.grid_sample(inputs.reshape(channels*samples, 1, side, side),
            grids.reshape(channels*samples, side, side, 2), mode="bilinear",
            padding_mode="border", align_corners=True)
        return observed.reshape(channels, samples, side, side).mean(dim=1)
    grids = motion_blur_grids(image.shape[-1], motion, frame, device=image.device, dtype=image.dtype)
    observed = F.grid_sample(image[None].expand(grids.shape[0], -1, -1, -1), grids,
                             mode="bilinear", padding_mode="border", align_corners=True)
    return observed.mean(dim=0)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < .08:
        return image
    # Camera-space blur can span many rectified pixels at low sampling density.
    # A fixed five-pixel radius would silently truncate precisely those cases.
    radius = max(1, int(np.ceil(3*sigma)))
    axis = torch.arange(-radius, radius+1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-axis.square() / (2*sigma*sigma))
    kernel /= kernel.sum()
    channels = image.shape[1]
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                     kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"),
                    kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def apply_capture_resolution(image, capture):
    """CHW RGB -> smaller capture -> Gaussian diffusion -> original crop size.

    Area resampling approximates finite pixel footprints. Blur sigma is measured
    in the smaller image's pixels; bilinear enlargement supplies no new evidence.
    This is a calibrated-crop approximation, not a full camera sensor simulator.
    """
    side = image.shape[-1]
    capture_side = capture["side"]
    if capture_side == side and capture["blurSigma"] == 0:
        return image
    low = image[None]
    if capture_side != side:
        low = F.interpolate(low, size=(capture_side, capture_side), mode="area")
    low = gaussian_blur(low, capture["blurSigma"])
    if capture_side != side:
        low = F.interpolate(low, size=(side, side), mode="bilinear", align_corners=False)
    return low[0]


def moire_phase_field(side, camera_side, period_camera, angle, curve=0., chirp=0.):
    """Moiré wavelength in camera pixels; curvature in cycles across the crop."""
    axis = (torch.arange(side, dtype=torch.float32)+.5)*2/side-1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return 2*np.pi*(camera_side/(2*period_camera)*(x*np.cos(angle)+y*np.sin(angle))
                   + curve*x*y + chirp*y*y)


def camera_motion_samples(side, motion, camera_to_crop=1.):
    """Increase exposure integration density when physical motion covers more pixels."""
    radius = np.sqrt(2)*side
    travel = (abs(motion['linearPixels'])+radius*(abs(np.deg2rad(motion['rotationDegrees']))
                                               + abs(motion['zoomFraction'])))
    # Optical blur also grows with cameraToCrop. Integrating at half-camera-
    # pixel intervals avoids oversampling a broad PSF on a large rectified crop.
    travel /= max(1., camera_to_crop)
    return min(63, max(motion['samples'], 2*int(np.ceil(travel))+1))


def apply_camera_sampling(image, camera, rng, severity, *, add_noise=True):
    """Arbitrary sensor phase, finite footprints, sensor noise, then rectification.

    The sampling phase is undone during reconstruction. Residual alignment error
    is applied separately, so changing the sensor lattice does not shift labels.
    If the real camera has more pixels than the output raster, compute only the
    output raster and scale noise to represent its larger averaging footprint.
    """
    side, count = image.shape[-1], camera['samplingSide']
    shift = torch.as_tensor(camera['samplingOffsetCamera'], dtype=image.dtype,
                            device=image.device)*2/camera['cameraSide']
    axis = (torch.arange(side, dtype=image.dtype, device=image.device)+.5)*2/side-1
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    canonical = torch.stack((x, y), dim=-1)[None]
    low = image[None]
    if count < side:
        shifted = F.grid_sample(low, canonical+shift, mode='bilinear',
                                padding_mode='border', align_corners=False)
        low = F.interpolate(shifted, size=(count, count), mode='area')
    if add_noise:
        noise = torch.as_tensor(rng.normal(size=tuple(low.shape)), dtype=low.dtype, device=low.device)
        std = severity*(.008+.025*low.clamp_min(0).sqrt())
        # Noise is added AFTER sensor integration, not averaged away by it.
        low = (low+noise*std*min(1., count/camera['cameraSide'])).clamp(0, 1)
    if count < side:
        low = F.grid_sample(low, canonical-shift, mode='bilinear',
                            padding_mode='border', align_corners=False)
    return low[0]


def corrupt_burst(clean: np.ndarray, seed: int, frames: int = 7, severity: float = .65,
                  *, photometric: PhotometricConfig | None = None,
                  resolution: CaptureResolutionConfig | None = None,
                  pixels_per_cell: int = 4) -> np.ndarray:
    """Return [T,3,H,W]; a shared texture evolves under small motion/focus shifts.

    Labels are unaffected by less than half-cell registration jitter. Spatial
    aliasing, curved multi-channel bands, rolling-shutter exposure/skew,
    integrated linear/rotational/radial-zoom exposure blur with relative RGB timing,
    white balance, signed exposure, soft tinted shade/reflections, drifting
    highlight/veiling glare, effective capture resolution, and shot/read
    noise vary independently of payload symbols. Seed is public augmentation data.
    """
    clean = np.asarray(clean, dtype=np.float32)
    _validate_crop_shape(clean, pixels_per_cell)
    if not 1 <= frames <= 16 or not 0 <= severity <= 1 or not 0 <= seed <= 0xffffffff:
        raise ValueError("Frames 1–16, severity 0–1, uint32 seed required")
    return _corrupt_sequence(clean[None], np.zeros(frames, dtype=np.int64), seed, severity,
                             photometric=photometric, resolution=resolution,
                             pixels_per_cell=pixels_per_cell)


def corrupt_phase_group(clean: np.ndarray, seed: int, frames: int = 7, severity: float = .65,
                        *, photometric: PhotometricConfig | None = None,
                        resolution: CaptureResolutionConfig | None = None,
                        pixels_per_cell: int = 4) -> np.ndarray:
    """Return [4,T,3,H,W] from one coherent phase-cycling camera trajectory.

    Slow optical conditions are drawn once. Frame p of each four-phase cycle
    sees the corresponding clean palette assignment; motion, moire, exposure
    drift and sensor noise advance continuously rather than resetting per phase.
    `frames` retains its CLI meaning: captures per phase, not total inputs.
    """
    clean = np.asarray(clean, dtype=np.float32)
    _validate_crop_shape(clean, pixels_per_cell, grouped=True)
    if type(frames) is not int or not 2 <= frames <= 16:
        raise ValueError("Phase grouping requires 2–16 captures per phase")
    if not 0 <= severity <= 1 or not 0 <= seed <= 0xffffffff:
        raise ValueError("Severity 0–1 and uint32 seed required")
    phases = np.tile(np.arange(4, dtype=np.int64), frames)
    observed = _corrupt_sequence(clean, phases, seed, severity,
                                photometric=photometric, resolution=resolution,
                                pixels_per_cell=pixels_per_cell)
    side = clean.shape[-1]
    return np.ascontiguousarray(observed.reshape(frames, 4, 3, side, side).transpose(1, 0, 2, 3, 4))


def _corrupt_sequence(clean, views, seed, severity, **kwargs):
    """Small in-memory convenience API; dataset generation streams the iterator."""
    return np.stack(list(_iter_corrupt_sequence(clean, views, seed, severity, **kwargs)))


def _iter_corrupt_sequence(clean, views, seed, severity, *, photometric=None,
                           resolution=None, pixels_per_cell=4, camera=None):
    """Yield one crop at a time along a shared, scheduled optical trajectory.

    Memory is bounded by the clean phase set and one simulated capture. W-tube
    receives physical camera geometry; its lengths are mapped from camera pixels
    into the stored crop. The independent OBM/v1 experiments retain explicit
    cell-relative lengths. Angles and image fractions keep their own units.
    """
    frames = len(views)
    photometric = photometric or PhotometricConfig()
    if camera is None:
        resolution = resolution_config(resolution, pixels_per_cell)
    if severity == 0:
        for view in views:
            yield clean[int(view)].copy()
        return
    rng = np.random.default_rng(seed)
    side = clean.shape[-1]
    capture = (sample_capture_resolution(side, seed, severity, resolution,
                                        pixels_per_cell=pixels_per_cell) if camera is None else None)
    pixel_scale = pixels_per_cell/4 if camera is None else camera['cameraToCrop']
    if camera is not None:
        severity *= camera['effectStrength']
    rgb = torch.from_numpy(np.clip((clean+1)/2, 0, 1))
    high = F.interpolate(rgb, scale_factor=2, mode="nearest")
    hy, hx = torch.meshgrid(torch.arange(2*side, dtype=torch.float32),
                           torch.arange(2*side, dtype=torch.float32), indexing="ij")
    # A high-resolution RGB stripe carrier is optically blurred before sampling.
    screen_period = rng.uniform(1.6, 3.4)*pixel_scale*(2 if camera is not None else 1)
    # Do not numerically alias a camera carrier the working raster cannot resolve.
    # Broad residual moire below remains present at every camera scale.
    carrier_weight = min(1., max(0., (screen_period-2)/2)) if camera is not None else 1.
    carrier_period = max(2., screen_period) if camera is not None else screen_period
    carrier = torch.stack([torch.cos(2*np.pi*hx/carrier_period+c*2*np.pi/3) for c in range(3)])
    high = (high * (1 + severity*rng.uniform(.04, .16)*carrier_weight*carrier[None])).clamp(0, 1)
    y, x = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij")
    angle, angle2 = rng.uniform(0, np.pi, 2)
    period, period2 = rng.uniform(7, 35), rng.uniform(4, 18)
    curve, chirp = rng.uniform(-7, 7), rng.uniform(-5, 5)
    phase0 = rng.uniform(0, 2*np.pi)
    velocity = rng.uniform(.10, .5)
    amplitudes = torch.tensor(rng.uniform(.03, .22, 3), dtype=torch.float32)[:, None, None] * severity
    color_mix = torch.tensor(np.eye(3)+rng.normal(0, .035*severity, (3, 3)), dtype=torch.float32)
    gamma = rng.uniform(1-.25*severity, 1+.3*severity)
    illumination = 1 + severity * (rng.uniform(-.14, .14)*x + rng.uniform(-.14, .14)*y)
    base_sigma = rng.uniform(.15, 1.3) * severity*pixel_scale
    base_shift = rng.uniform(-.6, .6, 2) * severity*pixel_scale
    drift = rng.uniform(-.18, .18, 2) * severity*pixel_scale
    affine_x = affine_y = 0.
    if camera is not None:
        # Geometry fields already include severity; only convert their units.
        base_sigma = camera['blurSigmaCamera']*2*pixel_scale
        base_shift = np.array(camera['registrationOffsetCamera'])*pixel_scale
        drift = np.array(camera['registrationDriftCamera'])*pixel_scale
        a, b, c, d = camera['affineCamera']
        affine_x, affine_y = pixel_scale*(a*x+b*y), pixel_scale*(c*x+d*y)
    camera_span = side/pixel_scale
    curved = moire_phase_field(side, camera_span, period, angle, curve, chirp)
    secondary = moire_phase_field(side, camera_span, period2, angle2)
    # Separate stream: toggling photometric controls does not redraw moiré,
    # motion or sensor noise, making controlled ablations easier to interpret.
    lighting_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x50484f54]))
    lighting = sample_photometric(lighting_rng, frames, severity, photometric)
    shutter_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485554]))
    shutter = sample_rolling_shutter(shutter_rng, severity, photometric)
    if shutter is not None:
        shutter["skew"] *= pixel_scale
    motion_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4d4f544e]))
    motion = sample_motion_blur(motion_rng, severity, photometric)
    if motion is not None:
        motion["linearPixels"] *= pixel_scale
        if camera is not None:
            motion['samples'] = camera_motion_samples(side, motion, pixel_scale)
    rgb_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5247424f]))
    rgb_offset = sample_rgb_offset(rgb_rng, severity, photometric, motion)
    if rgb_offset is not None:
        rgb_offset["maxPixels"] *= pixel_scale
    glare_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x474c4152]))
    glare = sample_glare(glare_rng, severity, photometric)
    for t in range(frames):
        view = int(views[t])
        blurred = gaussian_blur(high[view:view+1], base_sigma + severity*.15*pixel_scale*np.sin(t*.6+phase0))
        shift = np.clip(base_shift + np.sin(t*.55)*drift + rng.normal(0, .04, 2)*severity*pixel_scale,
                        -.9*pixel_scale, .9*pixel_scale)
        shutter_gains, row_offsets = rolling_shutter_maps(side, shutter, t)
        modulation = 1+.12*np.sin(t*.17+phase0)
        grid = torch.stack((x+2*(float(shift[0])+row_offsets+affine_x*modulation)/side,
                            y+2*(float(shift[1])+affine_y*modulation)/side), dim=-1)[None]
        observed = F.grid_sample(blurred, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        phase = phase0 + t*velocity
        bands = torch.stack([torch.sin(curved+phase+c*.63) + .35*torch.sin(secondary-phase*.7+c)
                             for c in range(3)])
        observed = observed + amplitudes*bands
        observed = apply_motion_blur(observed, motion, t, rgb_offset=rgb_offset)
        observed = torch.einsum("cd,dhw->chw", color_mix, observed) * illumination
        observed = observed*shutter_gains
        observed = apply_color_exposure(observed, lighting["whiteBalanceGains"][t],
                                         lighting["exposureStops"][t]).pow(gamma)
        observed = apply_overlays(observed, lighting["overlays"], t)
        observed = apply_glare(observed, glare, t)
        if camera is not None:
            observed = apply_camera_sampling(observed, camera, rng, severity).clamp(0, 1)
        else:
            # Independent OBM/2-pixel experiment uses its explicit fractional sampler.
            std = severity*(.008 + .025*observed.sqrt())
            noise = torch.from_numpy(rng.normal(size=clean.shape[1:]).astype(np.float32))
            observed = (observed + noise*std).clamp(0, 1)
            observed = apply_capture_resolution(observed, capture).clamp(0, 1)
        yield (observed*2-1).numpy().astype(np.float32, copy=False)


def clean_phase_groups(samples):
    """Validate complete clean phase sets before creating any output files.

    Group identity is the receiver/sender session, split, payload and complete
    phase-neutral frame header. Equal checksums alone do not establish a group.
    """
    groups = {}
    for sample in samples:
        row = sample.metadata
        info = phase_neutral_info(row["info"])
        key = json.dumps([row["sessionId"], row.get("senderSessionId"), row["payloadId"],
                          row["split"], info], sort_keys=True, separators=(",", ":"))
        phase = row["info"]["phase"]
        phases = groups.setdefault(key, {})
        if phase in phases:
            raise ValueError("Duplicate clean phase in one physical session/frame")
        phases[phase] = sample
    ordered = []
    for key, phases in sorted(groups.items()):
        if set(phases) != set(range(4)):
            raise ValueError("Phase grouping requires a complete set of four clean phases per session/frame")
        members = tuple(phases[phase] for phase in range(4))
        reference = None
        for phase, sample in enumerate(members):
            _, labels, mask = sample.tensors()
            canonical = canonical_labels(labels, mask, phase)
            expected = sample.read("expected", "u1")
            sample.read("baseline", "<f4")
            if reference is not None and any(not np.array_equal(a, b) for a, b in
                    zip((canonical, mask, expected), reference)):
                raise ValueError("Clean phase targets disagree on canonical labels, mask or payload bytes")
            reference = canonical, mask, expected
        ordered.append((key, members))
    return ordered


def paired_burst_counts(groups, paired_real, ratio, frames, dataset):
    """Count actual real captures, not nominal collection capacities.

    One synthetic trajectory contains ``frames`` observations of all four
    phases. Round upward to whole trajectories and report that rounding. IDs
    and splits must match exactly: a train target cannot borrow validation's
    capture count, even when the logical matrix happens to be equal.
    """
    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not np.isfinite(ratio) or not 0 < ratio <= 1024):
        raise ValueError("synthetic-ratio must be finite and in (0, 1024]")
    real = Dataset(paired_real)
    if ((real.dataset_format, real.dataset_version) != (NEURAL_QR_DATASET_FORMAT, 2)
            or (dataset.dataset_format, dataset.dataset_version) != (real.dataset_format, real.dataset_version)
            or any(s.metadata["source"] != "real" for s in real.samples)):
        raise ValueError("paired-real requires joined real Neural-QR v2 data and matching clean targets")
    counts = {}
    for sample in real.samples:
        row = sample.metadata
        key = row["sessionId"], row["payloadId"], row["split"]
        counts[key] = counts.get(key, 0)+1
    result, used = [], set()
    for _, members in groups:
        row = members[0].metadata
        key = row["sessionId"], row["payloadId"], row["split"]
        if key not in counts or key in used:
            raise ValueError("Every clean target must match exactly one paired real session/target/split")
        used.add(key)
        result.append(math.ceil(ratio*counts[key]/(4*frames)))
    if used != set(counts):
        raise ValueError("paired-real contains targets missing from the clean dataset")
    camera_pools, camera_coverage = build_camera_pools(real.samples)
    return result, {"requestedRatio": ratio, "realCaptures": sum(counts.values()),
        "syntheticCaptures": sum(result)*4*frames,
        "achievedRatio": sum(result)*4*frames/sum(counts.values()),
        "rounding": f"up to complete four-phase trajectories ({4*frames} captures)",
        "realManifestSha256": real.manifest_hash}, camera_pools, camera_coverage


def generate(clean_directories, out: Path, *, seed=7, frames=None, severity=None,
             max_output_mib=2**20, node="node", photometric: PhotometricConfig | None = None,
             bursts_per_target=None, severity_min=None,
             resolution: CaptureResolutionConfig | None = None, phase_grouped=None,
             paired_real=None, synthetic_ratio=4, recipe_name="standard", plan_only=False,
             summary_only=False, camera_config: CameraGeometryConfig | None = None):
    """Materialize deterministic corruptions, streaming one capture to disk.

    Neural-QR uses eight independent trajectories with eight captures per phase
    by default (256 crops/target). OBM's independent experiment retains its
    single seven-capture burst default. Both paths preserve physical session and
    target ownership; corruption never invents a new split identity.
    """
    dataset = Dataset(clean_directories)
    neural_qr = dataset.dataset_format == NEURAL_QR_DATASET_FORMAT
    pixels_per_cell = dataset.pixels_per_cell
    default_photometric = recipe_photometric(recipe_name)
    severity = (1. if recipe_name == "w-tube" else .65) if severity is None else severity
    severity_min = (.5 if recipe_name == "w-tube" else None) if severity_min is None else severity_min
    frames = (8 if neural_qr else 7) if frames is None else frames
    if paired_real is not None and bursts_per_target is not None:
        raise ValueError("Choose paired-real ratio or bursts-per-target, not both")
    bursts_per_target = (8 if neural_qr else 1) if bursts_per_target is None else bursts_per_target
    phase_grouped = neural_qr if phase_grouped is None else phase_grouped
    if type(phase_grouped) is not bool or neural_qr and not phase_grouped:
        raise ValueError("phase-grouped must be boolean; Neural-QR requires four coherent phases")
    if type(frames) is not int or not (2 if phase_grouped else 1) <= frames <= 16:
        raise ValueError("frames must be 2–16 per phase for grouped data, otherwise 1–16")
    if type(seed) is not int or not 0 <= seed <= 0xffffffff:
        raise ValueError("seed must be uint32")
    if not isinstance(max_output_mib, int) or isinstance(max_output_mib, bool) or max_output_mib < 0:
        raise ValueError("max-output-mib must be a nonnegative integer; zero disables the estimate limit")
    if type(bursts_per_target) is not int or not 1 <= bursts_per_target <= 2**20:
        raise ValueError("bursts-per-target must be an integer in [1, 1048576]")
    if isinstance(severity, bool) or not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("severity must be finite and in [0, 1]")
    if severity_min is not None and (isinstance(severity_min, bool) or
            not np.isfinite(severity_min) or not 0 <= severity_min <= severity):
        raise ValueError("severity-min must be finite and in [0, severity]")
    photometric = photometric or default_photometric
    physical_camera = neural_qr and dataset.dataset_version == 2
    if physical_camera:
        if resolution is not None:
            raise ValueError("Neural-QR v2 uses camera footprint controls, not fractional capture-resolution controls")
        camera_config = camera_config or CameraGeometryConfig()
        if not isinstance(camera_config, CameraGeometryConfig):
            raise ValueError("Expected CameraGeometryConfig")
    else:
        if camera_config is not None:
            raise ValueError("Camera geometry controls require Neural-QR v2")
        resolution = resolution_config(resolution, pixels_per_cell)
    corruption_kind = (W_TUBE_CORRUPTION_KIND if neural_qr and dataset.dataset_version == 2
                       else MULTIRES_CORRUPTION_KIND if neural_qr else CORRUPTION_KIND)
    recipe = {"kind": corruption_kind, "recipe": recipe_name, "seed": seed, "severity": severity,
              "severityMin": severity_min,
              "frames": frames, "phaseGrouped": phase_grouped,
              "phaseSchedule": "round-robin-continuous" if phase_grouped else None,
              "burstsPerTarget": None if paired_real is not None else bursts_per_target,
              "photometric": asdict(photometric)}
    if physical_camera:
        recipe['cameraGeometry'] = {**asdict(camera_config), 'units': 'camera-pixels',
            'sampling': 'phase-compensated-area-footprints', 'noise': 'after-camera-integration',
            'carrier': 'attenuated-below-working-raster-Nyquist'}
    else:
        recipe['captureResolution'] = asdict(resolution)
    if neural_qr:
        recipe.update(pixelsPerCell=pixels_per_cell)
    camera_pools = {}
    if any(sample.metadata["source"] != "synthetic" or "synthetic" in sample.metadata for sample in dataset.samples):
        raise ValueError("Generate corruptions from canonical synthetic clean datasets only; real captures remain unchanged")
    groups = clean_phase_groups(dataset.samples) if phase_grouped else [(s.id, (s,)) for s in dataset.samples]
    if paired_real is not None:
        if not phase_grouped:
            raise ValueError("paired-real generation requires four phase groups")
        burst_counts, ratio_report, camera_pools, camera_coverage = paired_burst_counts(
            groups, paired_real, synthetic_ratio, frames, dataset)
        recipe["pairedCounts"] = ratio_report
        recipe['cameraGeometry']['coverage'] = camera_coverage
    else:
        burst_counts = [bursts_per_target]*len(groups)
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    estimated = sum(sum(local_path(s.root, s.metadata[field]).stat().st_size
                        for field in ("input", "labels", "mask", "baseline", "expected"))
                    *frames*count for (_, members), count in zip(groups, burst_counts) for s in members)
    plan = {"format": "neural-corrector-synthetic-plan", "version": 1,
            "datasetVersion": dataset.dataset_version, "pixelsPerCell": pixels_per_cell,
            "targets": len(groups), "captures": sum(len(members)*frames*count
                for (_, members), count in zip(groups, burst_counts)),
            "estimatedBinaryBytes": estimated, "recipe": recipe,
            "recipeSha256": recipe_hash, "sourceManifestSha256": dataset.manifest_hash}
    if plan_only:
        return plan
    if max_output_mib and estimated > max_output_mib*1024**2:
        raise ValueError("Requested synthetic dataset exceeds --max-output-mib; increase the budget or use 0 to disable this estimate limit")
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    disk_parent = out
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    if estimated > shutil.disk_usage(disk_parent).free:
        raise ValueError(f"Insufficient free space for approximately {estimated/1024**3:.2f} GiB of synthetic binaries")
    out.mkdir(parents=True, exist_ok=True)
    generator = {**recipe, "recipeSha256": recipe_hash,
                 "estimatedBinaryBytes": estimated,
                 "sourceManifestSha256": dataset.manifest_hash,
                 "note": "Correlated calibrated-crop approximation; real-device evaluation is required."}
    manifest_path = out/"manifest.json"
    completed = 0
    progress = _Progress(plan["captures"])
    with manifest_path.open("w", encoding="utf8") as manifest_stream:
        manifest_stream.write('{"format":')
        json.dump(dataset.dataset_format, manifest_stream)
        manifest_stream.write(',"version":'+str(dataset.dataset_version)+',"samples":[')
        for (group_key, members), target_bursts in zip(groups, burst_counts):
            pixels = np.stack([sample.tensors()[0] for sample in members])
            for sample in members:
                sample.read("baseline", "<f4")
                sample.read("expected", "u1")
            _validate_crop_shape(pixels if phase_grouped else pixels[0], pixels_per_cell, grouped=phase_grouped)
            views = np.tile(np.arange(len(members), dtype=np.int64), frames)
            for burst_index in range(target_bursts):
                derived = int.from_bytes(hashlib.sha256(
                    f"{seed}/{group_key}/{burst_index}".encode()).digest()[:4], "little")
                # Every phase of a group shares one severity, capture scale and
                # physical trajectory. Sensors still produce fresh noise per frame.
                severity_rng = np.random.default_rng(np.random.SeedSequence([derived, 0x53455652]))
                burst_severity = (severity if severity_min is None else
                                  float(severity_rng.uniform(severity_min, severity)))
                camera = (sample_camera_geometry(members[0].shape[0], derived, burst_severity, camera_config,
                    observed_pixels_per_cell=camera_pools.get(camera_target_key(members[0].metadata), ()))
                    if physical_camera else None)
                capture = (sample_capture_resolution(pixels.shape[-1], derived, burst_severity, resolution,
                                                     pixels_per_cell=pixels_per_cell) if camera is None else None)
                phase_group_id = hashlib.sha256(f"{group_key}/{recipe_hash}/{burst_index}".encode()).hexdigest()[:24]
                observations = _iter_corrupt_sequence(pixels, views, derived, burst_severity,
                    photometric=photometric, resolution=resolution, pixels_per_cell=pixels_per_cell, camera=camera)
                for timeline_index, value in enumerate(observations):
                    sample = members[int(views[timeline_index])]
                    index = timeline_index//len(members)
                    burst_id = hashlib.sha256(f"{sample.id}/{recipe_hash}/{burst_index}".encode()).hexdigest()[:24]
                    identifier = hashlib.sha256(f"{burst_id}/{index}".encode()).hexdigest()[:24]
                    directory = out / identifier
                    directory.mkdir()
                    record = {**sample.metadata, "id": identifier, "hashes": {},
                              "source": "synthetic", "burstId": f"synthetic-{burst_id}",
                              "synthetic": {"seed": derived, "frame": index, "burst": burst_index,
                              "frames": frames, "severity": burst_severity, "parentId": sample.id,
                                  "kind": corruption_kind, "recipeSha256": recipe_hash},
                              "baselineMode": "canonical-from-corrupted-input"}
                    record['synthetic']['cameraGeometry' if physical_camera else 'captureResolution'] = camera if physical_camera else capture
                    if phase_grouped:
                        record["synthetic"].update(phaseGroupId=f"phase-group-{phase_group_id}",
                            timelineFrame=timeline_index)
                    for field in ("input", "labels", "mask", "baseline", "expected"):
                        suffix = {"input": "f32", "baseline": "f32", "expected": "bin"}.get(field, "u8")
                        path = directory / f"{field}.{suffix}"
                        if field == "input":
                            path.write_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())
                        else:
                            shutil.copyfile(local_path(sample.root, sample.metadata[field]), path)
                        record[field] = path.relative_to(out).as_posix()
                        record["hashes"][field] = digest(path)
                    # Other source-only file paths would point outside this bundle.
                    record.pop("samples", None)
                    record.pop("original", None)
                    if completed:
                        manifest_stream.write(",")
                    json.dump(record, manifest_stream, separators=(",", ":"), allow_nan=False)
                    completed += 1
                    progress.update(completed)
        manifest_stream.write('],"generator":')
        json.dump(generator, manifest_stream, separators=(",", ":"), allow_nan=False)
        manifest_stream.write("}\n")
    # Keep the stored baseline truthful too. Both transports use the canonical
    # JavaScript crop scorer; the copied clean scores are never used as results.
    bridge = Path(__file__).resolve().parents[2]/"scripts/corrector-dataset.mjs"
    print(f"Preparing canonical baselines for {completed} synthetic captures…",
          file=sys.stderr, flush=True)
    _prepare_baselines(node, bridge, out)
    if summary_only:
        return {"samples": completed, "out": str(out), "generator": generator}
    # API callers requesting the manifest receive the refreshed baseline hashes.
    return load_manifest(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Individual effect defaults shown above describe the standard recipe. "
        "The w-tube recipe raises these defaults; explicit effect arguments override it. "
        "Use --plan to inspect the fully resolved recipe before generation."))
    parser.add_argument("--clean", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recipe", choices=("standard", "w-tube"), default="standard",
                        help="w-tube draws severity .5–1 and stronger motion, color, glare and shutter effects")
    parser.add_argument("--paired-real", nargs="+", type=Path,
                        help="joined v2 real datasets whose capture counts determine synthetic volume")
    parser.add_argument("--synthetic-ratio", type=float, default=4,
                        help="synthetic captures per actual paired real capture (default 4), rounded to complete trajectories")
    parser.add_argument("--plan", action="store_true", help="validate inputs and print counts/estimated binary bytes without writing")
    parser.add_argument("--frames", type=int, default=None, help="captures per phase burst (Neural-QR default 8; OBM 7), 1–16 (grouped: 2–16)")
    parser.add_argument("--phase-grouped", action="store_true", default=None,
                        help="joint four-phase trajectories; automatically enabled for Neural-QR")
    parser.add_argument("--bursts-per-target", type=int, default=None,
                        help="independent trajectories per target (Neural-QR default 8; OBM 1)")
    parser.add_argument("--severity", type=float, default=None,
                        help="upper severity: standard .65, w-tube 1")
    parser.add_argument("--severity-min", type=float, default=None,
                        help="sample severity uniformly per burst from this minimum to --severity; omitted uses fixed severity")
    parser.add_argument("--capture-probability", type=float, default=None,
                        help="fraction of bursts with reduced capture resolution/blur (default: .75); 0 disables")
    parser.add_argument("--capture-min-scale", type=float, default=None,
                        help="minimum capture side / crop side; defaults to one pixel/cell (.5 at 2ppc, .25 at 4ppc)")
    parser.add_argument("--capture-max-scale", type=float, default=None,
                        help="maximum capture side / crop side, min-scale–1 (default: 1)")
    parser.add_argument("--capture-blur-sigma", type=float, default=None,
                        help="max Gaussian sigma in reduced-image pixels, 0–1 (default: .35); each burst draws half to full strength")
    parser.add_argument("--camera-footprint-min", type=float, default=None,
                        help="Neural-QR v2 fallback camera payload width, pixels (default96)")
    parser.add_argument("--camera-footprint-max", type=float, default=None,
                        help="Neural-QR v2 fallback camera payload width, pixels (default640)")
    parser.add_argument("--camera-easy-fraction", type=float, default=None,
                        help="Neural-QR v2 fraction with a larger footprint and gentler effects (default.15)")
    parser.add_argument("--white-balance-stops", type=float, default=None,
                        help="max residual per-channel log2 gain at severity 1 (default: .6); 0 disables")
    parser.add_argument("--exposure-stops", type=float, default=None,
                        help="max absolute exposure log2 gain at severity 1 (default: 1.5); 0 disables")
    parser.add_argument("--shade-opacity", type=float, default=None,
                        help="max soft shadow/reflection strength at severity 1 (default: .45); 0 disables")
    parser.add_argument("--shade-probability", type=float, default=None,
                        help="fraction of bursts with soft shadows/reflections (default: .7)")
    parser.add_argument("--structured-light-probability", type=float, default=None,
                        help="fraction of shade/glare events using edge shadows, window reflections or light streaks (standard: 0; w-tube: .65)")
    parser.add_argument("--glare-probability", type=float, default=None,
                        help="fraction of bursts with drifting highlight/veiling glare, 0–1 (default: .65)")
    parser.add_argument("--glare-strength", type=float, default=None,
                        help="max localized glare strength at severity 1, 0–1 (default: .6); 0 disables")
    parser.add_argument("--rolling-shutter-stops", type=float, default=None,
                        help="max scanline exposure log2 gain at severity 1 (default: .65); 0 disables banding")
    parser.add_argument("--rolling-shutter-probability", type=float, default=None,
                        help="fraction of bursts with drifting scanline bands/skew (default: .7); 0 disables both")
    parser.add_argument("--rolling-shutter-skew", type=float, default=None,
                        help="max residual scanline horizontal motion in dense pixels, 0–.75 (default: .5); 0 disables skew")
    parser.add_argument("--motion-probability", type=float, default=None,
                        help="fraction of bursts with integrated exposure motion blur (default: .7); 0 disables")
    parser.add_argument("--linear-blur-pixels", type=float, default=None,
                        help="max full linear exposure travel in dense-crop pixels at severity 1 (default: 2.5)")
    parser.add_argument("--rotation-blur-degrees", type=float, default=None,
                        help="max full rotational exposure sweep at severity 1 (default: .35 degrees)")
    parser.add_argument("--zoom-blur-fraction", type=float, default=None,
                        help="max full radial zoom scale sweep at severity 1 (default: .006)")
    parser.add_argument("--motion-samples", type=int, default=None,
                        help="odd exposure integration samples, 3–15 (default: 7)")
    parser.add_argument("--rgb-offset-probability", type=float, default=None,
                        help="fraction of motion-blurred bursts with relative RGB exposure offsets (default: .6); 0 disables")
    parser.add_argument("--rgb-offset-pixels", type=float, default=None,
                        help="max per-channel offset from common motion path in dense-crop pixels at severity 1, 0–1 (default: .5)")
    parser.add_argument("--max-output-mib", type=int, default=2**20, help="estimated binary output budget in MiB (default: 1 TiB); 0 disables the limit")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or args.max_output_mib < 0:
        parser.error("threads must be 1–64 and max-output-mib nonnegative")
    torch.set_num_threads(args.threads)
    try:
        photometric_values = asdict(recipe_photometric(args.recipe))
        photometric_values.update({key: getattr(args, key) for key in photometric_values
                                   if getattr(args, key) is not None})
        photometric = PhotometricConfig(**photometric_values)
        dataset = Dataset(args.clean)
        physical_camera = dataset.dataset_format == NEURAL_QR_DATASET_FORMAT and dataset.dataset_version == 2
        fractional = (args.capture_probability, args.capture_min_scale, args.capture_max_scale, args.capture_blur_sigma)
        geometry_options = {name: getattr(args, 'camera_'+name) for name in
                            ('footprint_min', 'footprint_max', 'easy_fraction')
                            if getattr(args, 'camera_'+name) is not None}
        if physical_camera and any(value is not None for value in fractional):
            raise ValueError("Neural-QR v2 uses --camera-footprint-min/max and --camera-easy-fraction, not --capture-* controls")
        if not physical_camera and geometry_options:
            raise ValueError("--camera-* controls require Neural-QR v2")
        resolution = None if physical_camera else CaptureResolutionConfig(
            probability=args.capture_probability if args.capture_probability is not None else .75,
            min_scale=args.capture_min_scale if args.capture_min_scale is not None else 1/dataset.pixels_per_cell,
            max_scale=args.capture_max_scale if args.capture_max_scale is not None else 1.,
            blur_sigma=args.capture_blur_sigma if args.capture_blur_sigma is not None else .35)
        result = generate(args.clean, args.out, seed=args.seed, frames=args.frames,
                          severity=args.severity, max_output_mib=args.max_output_mib, node=args.node,
                          photometric=photometric, bursts_per_target=args.bursts_per_target,
                          severity_min=args.severity_min, phase_grouped=args.phase_grouped,
                          paired_real=args.paired_real, synthetic_ratio=args.synthetic_ratio,
                          recipe_name=args.recipe, plan_only=args.plan, summary_only=True,
                          resolution=resolution,
                          camera_config=CameraGeometryConfig(**geometry_options) if physical_camera else None)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Synthetic generation failed: {error}\n")
    if args.plan:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()

