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
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from data import DATASET_FORMAT, Dataset, digest, local_path


CORRUPTION_KIND = "correlated-calibrated-crop-v5"


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

    def __post_init__(self):
        for name, limit in (("white_balance_stops", 4), ("exposure_stops", 4),
                            ("shade_opacity", .95), ("shade_probability", 1),
                            ("rolling_shutter_stops", 3), ("rolling_shutter_probability", 1),
                            ("rolling_shutter_skew", .75), ("motion_probability", 1),
                            ("linear_blur_pixels", 12), ("rotation_blur_degrees", 5),
                            ("zoom_blur_fraction", .05), ("glare_probability", 1),
                            ("glare_strength", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= limit:
                raise ValueError(f"{name.replace('_', '-')} must be finite and in [0, {limit}]")
        if type(self.motion_samples) is not int or self.motion_samples not in (3, 5, 7, 9, 11, 13, 15):
            raise ValueError("motion-samples must be an odd integer from 3 to 15")


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
    sine, cosine = np.sin(overlay["angle"]), np.cos(overlay["angle"])
    u = (cosine*x+sine*y)/(overlay["width"]/2)
    v = (-sine*x+cosine*y)/(overlay["height"]/2)
    radius = torch.maximum(u.abs(), v.abs()) if overlay["shape"] == "rectangle" else torch.sqrt(u*u+v*v)
    feathered = ((1-radius)/overlay["feather"]).clamp(0, 1)
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
    return {"cx": float(rng.uniform(.05, .95)), "cy": float(rng.uniform(.05, .95)),
            "radiusX": radius*np.sqrt(aspect), "radiusY": radius/np.sqrt(aspect),
            "angle": float(rng.uniform(-np.pi, np.pi)), "haloScale": float(rng.uniform(2.5, 4)),
            "coreStrength": strength, "veilStrength": strength*float(rng.uniform(.35, .65)),
            "color": tint.tolist(), "driftX": float(rng.uniform(-.035, .035)),
            "driftY": float(rng.uniform(-.035, .035)), "phase": float(rng.uniform(0, 2*np.pi)),
            "velocity": float(rng.uniform(.18, .4))}


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


def motion_blur_grids(side, motion, frame, *, device=None, dtype=torch.float32):
    """Centered exposure grids [S,H,W,2], in grid_sample normalized coordinates.

    Linear travel is in dense-crop pixels, rotation in degrees and zoom as a
    fractional scale sweep. A symmetric sample interval keeps the middle pose
    registered; the blur intentionally integrates evidence from adjacent cells.
    """
    axis = torch.linspace(-1, 1, side, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    times = torch.linspace(-.5, .5, motion["samples"], device=device, dtype=dtype)[:, None, None]
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


def apply_motion_blur(image, motion, frame=0):
    """Integrate a short linear/rotational/zoom exposure, using border extension."""
    if motion is None:
        return image
    grids = motion_blur_grids(image.shape[-1], motion, frame, device=image.device, dtype=image.dtype)
    observed = F.grid_sample(image[None].expand(grids.shape[0], -1, -1, -1), grids,
                             mode="bilinear", padding_mode="border", align_corners=True)
    return observed.mean(dim=0)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < .08:
        return image
    radius = min(5, max(1, int(np.ceil(3*sigma))))
    axis = torch.arange(-radius, radius+1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-axis.square() / (2*sigma*sigma))
    kernel /= kernel.sum()
    channels = image.shape[1]
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                     kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"),
                    kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def corrupt_burst(clean: np.ndarray, seed: int, frames: int = 7, severity: float = .65,
                  *, photometric: PhotometricConfig | None = None) -> np.ndarray:
    """Return [T,3,H,W]; a shared texture evolves under small motion/focus shifts.

    Labels are unaffected by less than half-cell registration jitter. Spatial
    aliasing, curved multi-channel bands, rolling-shutter exposure/skew,
    integrated linear/rotational/radial-zoom exposure blur,
    white balance, signed exposure, soft tinted shade/reflections, drifting
    highlight/veiling glare, and shot/read
    noise vary independently of payload symbols. Seed is public augmentation data.
    """
    clean = np.asarray(clean, dtype=np.float32)
    if clean.ndim != 3 or clean.shape[0] != 3 or clean.shape[1:] not in ((256, 256), (512, 512)):
        raise ValueError("Expected normalized [3,256,256] or [3,512,512] input")
    if not np.isfinite(clean).all() or np.max(np.abs(clean)) > 1.50001:
        raise ValueError("Input violates the finite normalized crop contract")
    if not 1 <= frames <= 16 or not 0 <= severity <= 1 or not 0 <= seed <= 0xffffffff:
        raise ValueError("Frames 1–16, severity 0–1, uint32 seed required")
    photometric = photometric or PhotometricConfig()
    if severity == 0:
        return np.repeat(clean[None], frames, axis=0).copy()
    rng = np.random.default_rng(seed)
    side = clean.shape[-1]
    rgb = torch.from_numpy(np.clip((clean+1)/2, 0, 1))[None]
    high = F.interpolate(rgb, scale_factor=2, mode="nearest")
    hy, hx = torch.meshgrid(torch.arange(2*side, dtype=torch.float32),
                           torch.arange(2*side, dtype=torch.float32), indexing="ij")
    # A high-resolution RGB stripe carrier is optically blurred before sampling.
    screen_period = rng.uniform(1.6, 3.4)
    carrier = torch.stack([torch.cos(2*np.pi*hx/screen_period+c*2*np.pi/3) for c in range(3)])
    high = (high * (1 + severity*rng.uniform(.04, .16)*carrier[None])).clamp(0, 1)
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
    base_sigma = rng.uniform(.15, 1.3) * severity
    base_shift = rng.uniform(-.6, .6, 2) * severity
    drift = rng.uniform(-.18, .18, 2) * severity
    # Separate stream: toggling photometric controls does not redraw moiré,
    # motion or sensor noise, making controlled ablations easier to interpret.
    lighting_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x50484f54]))
    lighting = sample_photometric(lighting_rng, frames, severity, photometric)
    shutter_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485554]))
    shutter = sample_rolling_shutter(shutter_rng, severity, photometric)
    motion_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4d4f544e]))
    motion = sample_motion_blur(motion_rng, severity, photometric)
    glare_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x474c4152]))
    glare = sample_glare(glare_rng, severity, photometric)
    outputs = []
    for t in range(frames):
        blurred = gaussian_blur(high, base_sigma + severity*.15*np.sin(t*.6+phase0))
        shift = np.clip(base_shift + np.sin(t*.55)*drift + rng.normal(0, .04, 2)*severity, -.9, .9)
        shutter_gains, row_offsets = rolling_shutter_maps(side, shutter, t)
        grid = torch.stack((x+2*(float(shift[0])+row_offsets)/side,
                            y+2*float(shift[1])/side), dim=-1)[None]
        observed = F.grid_sample(blurred, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        phase = phase0 + t*velocity
        curved = 2*np.pi*(side/period*(x*np.cos(angle)+y*np.sin(angle)) + curve*x*y + chirp*y*y)
        secondary = 2*np.pi*side/period2*(x*np.cos(angle2)+y*np.sin(angle2))
        bands = torch.stack([torch.sin(curved+phase+c*.63) + .35*torch.sin(secondary-phase*.7+c)
                             for c in range(3)])
        observed = observed + amplitudes*bands
        observed = apply_motion_blur(observed, motion, t)
        observed = torch.einsum("cd,dhw->chw", color_mix, observed) * illumination
        observed = observed*shutter_gains
        observed = apply_color_exposure(observed, lighting["whiteBalanceGains"][t],
                                         lighting["exposureStops"][t]).pow(gamma)
        observed = apply_overlays(observed, lighting["overlays"], t)
        observed = apply_glare(observed, glare, t)
        # Shot variance follows intensity; read noise is independent per capture.
        std = severity*(.008 + .025*observed.sqrt())
        noise = torch.from_numpy(rng.normal(size=clean.shape).astype(np.float32))
        observed = (observed + noise*std).clamp(0, 1)
        outputs.append((observed*2-1).numpy())
    return np.stack(outputs).astype(np.float32, copy=False)


def generate(clean_directories, out: Path, *, seed=7, frames=7, severity=.65,
             max_output_mib=2**20, node="node", photometric: PhotometricConfig | None = None,
             bursts_per_target=1, severity_min=None):
    if not isinstance(max_output_mib, int) or isinstance(max_output_mib, bool) or max_output_mib < 0:
        raise ValueError("max-output-mib must be a nonnegative integer; zero disables the estimate limit")
    if type(bursts_per_target) is not int or not 1 <= bursts_per_target <= 2**20:
        raise ValueError("bursts-per-target must be an integer in [1, 1048576]")
    if isinstance(severity, bool) or not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("severity must be finite and in [0, 1]")
    if severity_min is not None and (isinstance(severity_min, bool) or
            not np.isfinite(severity_min) or not 0 <= severity_min <= severity):
        raise ValueError("severity-min must be finite and in [0, severity]")
    photometric = photometric or PhotometricConfig()
    recipe = {"kind": CORRUPTION_KIND, "seed": seed, "severity": severity,
              "severityMin": severity_min,
              "frames": frames, "burstsPerTarget": bursts_per_target, "photometric": asdict(photometric)}
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    dataset = Dataset(clean_directories)
    if any(sample.metadata["source"] != "synthetic" or "synthetic" in sample.metadata for sample in dataset.samples):
        raise ValueError("Generate corruptions from canonical synthetic clean datasets only; real captures remain unchanged")
    estimated = sum(sum(local_path(s.root, s.metadata[field]).stat().st_size
                        for field in ("input", "labels", "mask", "baseline", "expected"))
                    for s in dataset.samples)*frames*bursts_per_target
    if max_output_mib and estimated > max_output_mib*1024**2:
        raise ValueError("Requested synthetic dataset exceeds --max-output-mib; increase the budget or use 0 to disable this estimate limit")
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in dataset.samples:
        pixels, _, _ = sample.tensors()
        sample.read("baseline", "<f4")
        sample.read("expected", "u1")
        for burst_index in range(bursts_per_target):
            derived = int.from_bytes(hashlib.sha256(
                f"{seed}/{sample.id}/{burst_index}".encode()).digest()[:4], "little")
            # One strength per trajectory, not independently per frame. Its own
            # stream keeps selecting strength independent from all artifact RNGs.
            severity_rng = np.random.default_rng(np.random.SeedSequence([derived, 0x53455652]))
            burst_severity = (severity if severity_min is None else
                              float(severity_rng.uniform(severity_min, severity)))
            burst = corrupt_burst(pixels, derived, frames, burst_severity, photometric=photometric)
            burst_id = hashlib.sha256(f"{sample.id}/{recipe_hash}/{burst_index}".encode()).hexdigest()[:24]
            for index, value in enumerate(burst):
                identifier = hashlib.sha256(f"{burst_id}/{index}".encode()).hexdigest()[:24]
                directory = out / identifier
                directory.mkdir()
                record = {**sample.metadata, "id": identifier, "hashes": {},
                          "source": "synthetic", "burstId": f"synthetic-{burst_id}",
                          "synthetic": {"seed": derived, "frame": index, "burst": burst_index,
                          "frames": frames, "severity": burst_severity, "parentId": sample.id,
                          "kind": CORRUPTION_KIND, "recipeSha256": recipe_hash},
                          "baselineMode": "canonical-from-corrupted-input"}
                for field in ("input", "labels", "mask", "baseline", "expected"):
                    suffix = {"input": "f32", "baseline": "f32", "expected": "bin"}.get(field, "u8")
                    path = directory / f"{field}.{suffix}"
                    if field == "input":
                        path.write_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())
                    else:
                        shutil.copyfile(local_path(sample.root, sample.metadata[field]), path)
                    record[field] = path.relative_to(out).as_posix()
                    record["hashes"][field] = digest(path)
                # Other source-only file paths would point outside this new bundle.
                record.pop("samples", None)
                record.pop("original", None)
                records.append(record)
    manifest = {"format": DATASET_FORMAT, "version": 1, "samples": records,
                "generator": {**recipe, "recipeSha256": recipe_hash,
                              "sourceManifestSha256": dataset.manifest_hash,
                              "note": "Correlated calibrated-crop approximation; real-device evaluation is required."}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Keep the stored baseline truthful too. The evaluation bridge independently
    # recomputes synthetic scores from these pixels as an additional safeguard.
    bridge = Path(__file__).resolve().parents[2]/"scripts/decoder-dataset.mjs"
    process = subprocess.run([node, str(bridge), "baselines", "--dataset", str(out)],
                             capture_output=True, text=True, timeout=max(60, len(records)*3))
    if process.returncode:
        raise RuntimeError(f"Canonical synthetic baseline preparation failed: {process.stderr[-2000:]}")
    return json.loads((out/"manifest.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bursts-per-target", type=int, default=1,
                        help="independent corruption bursts per clean target (default: 1)")
    parser.add_argument("--severity", type=float, default=.65)
    parser.add_argument("--severity-min", type=float, default=None,
                        help="sample severity uniformly per burst from this minimum to --severity; omitted uses fixed severity")
    parser.add_argument("--white-balance-stops", type=float, default=.6,
                        help="max residual per-channel log2 gain at severity 1 (default: .6); 0 disables")
    parser.add_argument("--exposure-stops", type=float, default=1.5,
                        help="max absolute exposure log2 gain at severity 1 (default: 1.5); 0 disables")
    parser.add_argument("--shade-opacity", type=float, default=.45,
                        help="max soft shadow/reflection strength at severity 1 (default: .45); 0 disables")
    parser.add_argument("--shade-probability", type=float, default=.7,
                        help="fraction of bursts with rotated rectangles/ellipses (default: .7)")
    parser.add_argument("--glare-probability", type=float, default=.65,
                        help="fraction of bursts with drifting highlight/veiling glare, 0–1 (default: .65)")
    parser.add_argument("--glare-strength", type=float, default=.6,
                        help="max localized glare strength at severity 1, 0–1 (default: .6); 0 disables")
    parser.add_argument("--rolling-shutter-stops", type=float, default=.65,
                        help="max scanline exposure log2 gain at severity 1 (default: .65); 0 disables banding")
    parser.add_argument("--rolling-shutter-probability", type=float, default=.7,
                        help="fraction of bursts with drifting scanline bands/skew (default: .7); 0 disables both")
    parser.add_argument("--rolling-shutter-skew", type=float, default=.5,
                        help="max residual scanline horizontal motion in dense pixels, 0–.75 (default: .5); 0 disables skew")
    parser.add_argument("--motion-probability", type=float, default=.7,
                        help="fraction of bursts with integrated exposure motion blur (default: .7); 0 disables")
    parser.add_argument("--linear-blur-pixels", type=float, default=2.5,
                        help="max full linear exposure travel in dense-crop pixels at severity 1 (default: 2.5)")
    parser.add_argument("--rotation-blur-degrees", type=float, default=.35,
                        help="max full rotational exposure sweep at severity 1 (default: .35 degrees)")
    parser.add_argument("--zoom-blur-fraction", type=float, default=.006,
                        help="max full radial zoom scale sweep at severity 1 (default: .006)")
    parser.add_argument("--motion-samples", type=int, default=7,
                        help="odd exposure integration samples, 3–15 (default: 7)")
    parser.add_argument("--max-output-mib", type=int, default=2**20, help="estimated binary output budget in MiB (default: 1 TiB); 0 disables the limit")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or args.max_output_mib < 0:
        parser.error("threads must be 1–64 and max-output-mib nonnegative")
    torch.set_num_threads(args.threads)
    try:
        photometric = PhotometricConfig(args.white_balance_stops, args.exposure_stops,
                                         args.shade_opacity, args.shade_probability,
                                         args.rolling_shutter_stops, args.rolling_shutter_probability,
                                         args.rolling_shutter_skew, args.motion_probability,
                                         args.linear_blur_pixels, args.rotation_blur_degrees,
                                         args.zoom_blur_fraction, args.motion_samples,
                                         args.glare_probability, args.glare_strength)
        result = generate(args.clean, args.out, seed=args.seed, frames=args.frames,
                          severity=args.severity, max_output_mib=args.max_output_mib, node=args.node,
                          photometric=photometric, bursts_per_target=args.bursts_per_target,
                          severity_min=args.severity_min)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Synthetic generation failed: {error}\n")
    print(f"Wrote {len(result['samples'])} correlated synthetic crops to {args.out.resolve()}")


if __name__ == "__main__":
    main()
