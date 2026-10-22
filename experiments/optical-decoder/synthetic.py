#!/usr/bin/env python3
"""Generate correlated corrupted crops from canonical JavaScript clean datasets.

This is a calibrated-crop approximation of screen/camera interference, not a
physical sensor simulator. Real held-out sessions decide whether it is useful.
All frames in a burst retain the source session/payload split and physical labels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from data import DATASET_FORMAT, Dataset, digest, local_path


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


def corrupt_burst(clean: np.ndarray, seed: int, frames: int = 3, severity: float = .65) -> np.ndarray:
    """Return [T,3,H,W]; a shared texture evolves under small motion/focus shifts.

    Labels are unaffected by less than half-cell registration jitter. Spatial
    aliasing, curved multi-channel bands, blur, exposure changes, and shot/read
    noise vary independently of payload symbols. Seed is public augmentation data.
    """
    clean = np.asarray(clean, dtype=np.float32)
    if clean.ndim != 3 or clean.shape[0] != 3 or clean.shape[1:] not in ((256, 256), (512, 512)):
        raise ValueError("Expected normalized [3,256,256] or [3,512,512] input")
    if not np.isfinite(clean).all() or np.max(np.abs(clean)) > 1.50001:
        raise ValueError("Input violates the finite normalized crop contract")
    if not 1 <= frames <= 16 or not 0 <= severity <= 1 or not 0 <= seed <= 0xffffffff:
        raise ValueError("Frames 1–16, severity 0–1, uint32 seed required")
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
    outputs = []
    for t in range(frames):
        blurred = gaussian_blur(high, base_sigma + severity*.15*np.sin(t*.6+phase0))
        shift = np.clip(base_shift + np.sin(t*.55)*drift + rng.normal(0, .04, 2)*severity, -.9, .9)
        grid = torch.stack((x+2*float(shift[0])/side, y+2*float(shift[1])/side), dim=-1)[None]
        observed = F.grid_sample(blurred, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        # Short, directional camera motion; no integer-cell translations.
        if t % 3 == 1:
            motion_grid = grid.clone()
            motion_grid[..., 0] += float(rng.uniform(-.7, .7)*severity/side)
            observed = (observed+F.grid_sample(blurred, motion_grid, padding_mode="border", align_corners=True)[0])/2
        phase = phase0 + t*velocity
        curved = 2*np.pi*(side/period*(x*np.cos(angle)+y*np.sin(angle)) + curve*x*y + chirp*y*y)
        secondary = 2*np.pi*side/period2*(x*np.cos(angle2)+y*np.sin(angle2))
        bands = torch.stack([torch.sin(curved+phase+c*.63) + .35*torch.sin(secondary-phase*.7+c)
                             for c in range(3)])
        observed = observed + amplitudes*bands
        observed = torch.einsum("cd,dhw->chw", color_mix, observed) * illumination
        gain = 1 + severity * (.07*np.sin(phase*.6) + (.08 if t == frames//2 else 0))
        observed = (observed*gain).clamp(0, 1).pow(gamma)
        # Shot variance follows intensity; read noise is independent per capture.
        std = severity*(.008 + .025*observed.sqrt())
        noise = torch.from_numpy(rng.normal(size=clean.shape).astype(np.float32))
        observed = (observed + noise*std).clamp(0, 1)
        outputs.append((observed*2-1).numpy())
    return np.stack(outputs).astype(np.float32, copy=False)


def generate(clean_directories, out: Path, *, seed=7, frames=3, severity=.65, max_output_mib=1024, node="node"):
    dataset = Dataset(clean_directories)
    if any(sample.metadata["source"] != "synthetic" or "synthetic" in sample.metadata for sample in dataset.samples):
        raise ValueError("Generate corruptions from canonical synthetic clean datasets only; real captures remain unchanged")
    estimated = sum(sum(local_path(s.root, s.metadata[field]).stat().st_size
                        for field in ("input", "labels", "mask", "baseline", "expected"))
                    for s in dataset.samples)*frames
    if estimated > max_output_mib*1024**2:
        raise ValueError("Requested synthetic dataset exceeds --max-output-mib; use fewer source frames/burst captures")
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in dataset.samples:
        pixels, _, _ = sample.tensors()
        sample.read("baseline", "<f4")
        sample.read("expected", "u1")
        derived = int.from_bytes(hashlib.sha256(f"{seed}/{sample.id}".encode()).digest()[:4], "little")
        burst = corrupt_burst(pixels, derived, frames, severity)
        for index, value in enumerate(burst):
            identifier = hashlib.sha256(f"{sample.id}/{seed}/{index}".encode()).hexdigest()[:24]
            directory = out / identifier
            directory.mkdir()
            record = {**sample.metadata, "id": identifier, "hashes": {},
                      "source": "synthetic", "synthetic": {"seed": derived, "frame": index,
                      "frames": frames, "severity": severity, "parentId": sample.id,
                      "kind": "correlated-calibrated-crop-v1"},
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
            records.append(record)
    manifest = {"format": DATASET_FORMAT, "version": 1, "samples": records,
                "generator": {"seed": seed, "severity": severity, "frames": frames,
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
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--severity", type=float, default=.65)
    parser.add_argument("--max-output-mib", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or args.max_output_mib < 1:
        parser.error("threads must be 1–64 and max-output-mib positive")
    torch.set_num_threads(args.threads)
    try:
        result = generate(args.clean, args.out, seed=args.seed, frames=args.frames,
                          severity=args.severity, max_output_mib=args.max_output_mib, node=args.node)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Synthetic generation failed: {error}\n")
    print(f"Wrote {len(result['samples'])} correlated synthetic crops to {args.out.resolve()}")


if __name__ == "__main__":
    main()
