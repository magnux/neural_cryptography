#!/usr/bin/env python3
"""Base neural cryptography POC: recurrent CNN and invertible Feistel experiments.

    python neural_key_experiment.py --compare
    python neural_key_experiment.py --size 16 --colors 16 --frames 4 --camera
    python neural_key_experiment.py --architecture feistel --rotation bilinear
    python neural_key_experiment.py --self-test

Each challenge is a sequence of --frames matrices of --size x --size cells.
--colors controls the actual input/output alphabet (2, 4, 8, or 16).
Frames are stacked vertically for convolution; Feistel shuffling spans the whole
sequence. The model returns the same number of symbols as its input. For Feistel,
the total bit count must be even (use an even size or frame count for 2/8 colors).
--size/--matrix-size supports 8..512; --frames supports 1..8. Larger challenges
increase computation, unlike the transport-only dimensions in the ZKBoo POC.
Use small --samples/--batch-size and disable optional attacks/probes for large grids.

The keyed kernel can be Gaussian/orthogonal and fixed/bilinearly rotating.
Wave activations: linear, relu, tanh, silu, sine. Normalization: layer, instance,
mixed, none. Feistel uses the CNN as its round function and preserves all bits.
A public shuffle improves global mixing; --no-shuffle disables it for ablation.
The small surrogate is the only trained model. Test seeds are reproducible, not
secure credentials. Avalanche, bias, covariance and attack scores are diagnostics,
not security estimates. Float arithmetic can differ across hardware and precision.

Outputs include summary.json, examples.svg, bounded trajectory.npz, optional
avalanche.npz, and animated challenge/response HTML and PNG sequences. Raw matrices
contain no frame headers: order and completeness must be provided by capture.
--camera uses an aligned, calibrated cell-mean simulator shared with the ZKBoo POC.
It does not acquire real camera images or perform error correction/perspective recovery.
"""
import argparse
import base64
import copy
import csv
import json
import math
import os
from pathlib import Path
import time
import sys
import xml.etree.ElementTree as ET
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Shared utilities live once in the repository; direct script execution works from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from neural_optical import (palette, camera_capture, png, save_sequence,
                            add_optical_arguments, validate_optical_arguments)

def normalize_maps(x, mode="mixed", eps=1e-5):
    """Same groups as the notebook, with explicit population variances."""
    if mode == "none":
        return x

    def standardized(dims):
        centered = x - x.mean(dim=dims, keepdim=True)
        return centered * (centered.square().mean(dim=dims, keepdim=True) + eps).rsqrt()

    if mode == "instance":
        return standardized((2, 3))
    if mode == "layer":
        return standardized((1, 2, 3))
    if mode == "mixed":
        return 0.5 * (standardized((2, 3)) + standardized((1, 2, 3)))
    raise ValueError(f"Unknown normalization: {mode}")


def symbol_bits(symbols, colors=4):
    """[N,F,S,S] symbols -> [N,log2(colors),F*S,S], low-bit plane first.

    A three-dimensional input is accepted as a one-frame sequence.
    """
    if symbols.ndim == 3:
        symbols = symbols[:, None]
    if symbols.ndim != 4 or symbols.shape[-2] != symbols.shape[-1]:
        raise ValueError("expected [batch, frames, side, side] symbols")
    if colors not in (2, 4, 8, 16) or torch.any(symbols < 0) or torch.any(symbols >= colors):
        raise ValueError("symbols must belong to the configured palette")
    planes = torch.stack([(symbols >> i) & 1 for i in range(colors.bit_length() - 1)], dim=1)
    return planes.flatten(2, 3)


def bipolar(symbols, colors=4):
    return symbol_bits(symbols, colors).float().mul(2).sub(1)


def bits_to_symbols(bits, colors=4, frames=1):
    count = colors.bit_length() - 1
    if colors not in (2, 4, 8, 16) or bits.shape[1] != count:
        raise ValueError("bit-plane count does not match palette")
    symbols = sum(bits[:, i] * (1 << i) for i in range(count))
    side = symbols.shape[-1]
    return symbols.reshape(len(symbols), frames, side, side)


def rotate_kernel_bank(weight, angles):
    """Notebook-compatible bilinear rotation, independently per output filter."""
    angles = angles.to(weight)
    theta = weight.new_zeros((weight.shape[0], 2, 3))
    s, c = angles.sin(), angles.cos()
    theta[:, 0, 0], theta[:, 0, 1] = c, -s
    theta[:, 1, 0], theta[:, 1, 1] = s, c
    grid = F.affine_grid(theta, weight.size(), align_corners=False)
    return F.grid_sample(weight, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class NeuralKey(nn.Module):
    """Frozen keyed kernel and, when enabled, keyed angular increments."""

    def __init__(self, channels=8, key_seed=8, norm="layer", activation="relu", padding="zeros", *,
                 size=8, init="gaussian", gain=1., kernel_size=3, rotation="none",
                 rotation_period=64, rotation_renorm=False, activation_gain=3., phase=.5,
                 architecture="wave", shuffle=True, colors=4, frames=1):
        super().__init__()
        self.norm, self.activation, self.padding = norm, activation, padding
        self.architecture, self.size, self.shuffle = architecture, size, shuffle
        if colors not in (2, 4, 8, 16) or not 1 <= frames <= 8 or not 8 <= size <= 512:
            raise ValueError("invalid palette, frame count or matrix size")
        self.colors, self.frames = colors, frames
        self.bit_planes = colors.bit_length() - 1
        self.state_bits = self.bit_planes * frames * size * size
        if channels < max(2, self.bit_planes):
            raise ValueError("channels must cover all output bit planes and be at least two")
        if architecture == "feistel" and self.state_bits % 2:
            raise ValueError("Feistel requires an even total bit count: choose an even size or frame count")
        # Pack all sequence bits into exactly two spatial halves, without padding.
        # The default 4-color, one-frame layout stays [N,2,S,S].
        self.half_width = size if (self.bit_planes * frames) % 2 == 0 else size // 2
        self.half_height = self.state_bits // (2 * self.half_width)
        self.rotation, self.rotation_period, self.rotation_renorm = rotation, rotation_period, rotation_renorm
        self.activation_gain, self.phase = activation_gain, phase
        public_rng = torch.Generator().manual_seed(2026)
        key_rng = torch.Generator().manual_seed(key_seed)
        embedding = torch.randn(channels, self.bit_planes, 1, 1, generator=public_rng) / math.sqrt(self.bit_planes)
        # Same public embedding for every key. Complementing all input bits negates it.
        self.register_buffer("embedding", embedding)
        shape = (channels, channels, kernel_size, kernel_size)
        if init == "orthogonal":
            kernel = nn.init.orthogonal_(torch.empty(shape), gain=gain, generator=key_rng)
        else:
            kernel = gain * torch.randn(shape, generator=key_rng) / math.sqrt(kernel_size**2 * channels)
        self.register_buffer("kernel", kernel)
        # Separate RNG stream: enabling rotation leaves the base bank unchanged.
        angle_rng = torch.Generator().manual_seed(key_seed ^ 0x5A17)
        self.register_buffer("angular_increment", torch.randn(channels, generator=angle_rng) * (math.pi / rotation_period))
        public_perm_rng = torch.Generator().manual_seed(2027)
        permutation = (torch.randperm(self.state_bits, generator=public_perm_rng)
                       if architecture == "feistel" else torch.empty(0, dtype=torch.long))
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", permutation.argsort())

    @property
    def key_coefficients(self):
        return self.kernel.numel() + (self.angular_increment.numel() if self.rotation != "none" else 0)

    def weight_at(self, step_index):
        if self.rotation == "none":
            return self.kernel
        angle = self.angular_increment * (step_index % self.rotation_period)
        weight = rotate_kernel_bank(self.kernel, angle)
        if self.rotation_renorm:
            base_norm = self.kernel.flatten(1).norm(dim=1)
            rotated_norm = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
            weight = weight * (base_norm / rotated_norm)[:, None, None, None]
        return weight

    def schedule(self, steps):
        return [self.weight_at(t) for t in range(1, steps + 1)]

    def encode(self, symbols):
        if symbols.ndim == 3:
            symbols = symbols[:, None]
        if tuple(symbols.shape[1:]) != (self.frames, self.size, self.size):
            raise ValueError("input sequence dimensions must match the model")
        if self.architecture == "feistel":
            return symbol_bits(symbols, self.colors).to(torch.uint8).reshape(
                len(symbols), 2, self.half_height, self.half_width)
        return F.conv2d(bipolar(symbols, self.colors).to(self.embedding.dtype), self.embedding)

    def field_update(self, x, weight):
        pad = weight.shape[-1] // 2
        if self.padding == "circular":
            y = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="circular"), weight)
        else:
            y = F.conv2d(x, weight, padding=pad)
        if self.activation == "tanh":
            y = y.tanh()
        elif self.activation == "relu":
            y = F.relu(y)
        elif self.activation == "silu":
            y = F.silu(y)
        elif self.activation == "sine":
            y = torch.sin(self.activation_gain * y + self.phase)
        features = normalize_maps(torch.cat((x, y), dim=1), self.norm)
        a, b = features.chunk(2, dim=1)
        return 0.5 * (a + b)

    def round_function(self, right, weight):
        field = F.conv2d(right.to(self.embedding.dtype).mul(2).sub(1), self.embedding[:, :1])
        for _ in range(2):
            field = self.field_update(field, weight)
        # Channel contrast avoids relying on a single channel's DC preference.
        return (field[:, :1] - field[:, 1:2] >= 0).to(torch.uint8)

    def step(self, x, step_index=1, weight=None):
        weight = self.weight_at(step_index) if weight is None else weight
        if self.architecture == "wave":
            return self.field_update(x, weight)
        left, right = x.chunk(2, dim=1)
        result = torch.cat((right, left ^ self.round_function(right, weight)), dim=1)
        if self.shuffle:
            result = result.flatten(1)[:, self.permutation].reshape_as(result)
        return result

    def readout(self, state):
        if self.architecture == "feistel":
            return state.reshape(len(state), self.bit_planes, self.frames * self.size, self.size).to(
                self.embedding.dtype).mul(2).sub(1)
        return state[:, :self.bit_planes]

    def inverse_step(self, state, step_index, weight=None):
        if self.architecture != "feistel":
            raise ValueError("Only the Feistel bit architecture is invertible.")
        weight = self.weight_at(step_index) if weight is None else weight
        if self.shuffle:
            state = state.flatten(1)[:, self.inverse_permutation].reshape_as(state)
        right, transformed_left = state.chunk(2, dim=1)
        return torch.cat((transformed_left ^ self.round_function(right, weight), right), dim=1)


def sample_challenges(n, size, seed, forbidden=(), *, colors=4, frames=1):
    """Sample unique grids; explicitly keep attacker training out of evaluation."""
    rng = np.random.default_rng(seed)
    seen = {np.asarray(x, dtype=np.uint8).tobytes() for x in forbidden}
    result = []
    while len(result) < n:
        x = rng.integers(0, colors, size=(frames, size, size), dtype=np.uint8)
        key = x.tobytes()
        if key not in seen:
            seen.add(key)
            result.append(x)
    return np.stack(result)


def flip_one_bit(challenges, seed, colors=4):
    rng = np.random.default_rng(seed)
    changed = challenges.copy().reshape(len(challenges), -1)
    positions = rng.integers(changed.shape[1], size=len(challenges))
    masks = (1 << rng.integers(0, colors.bit_length() - 1, size=len(challenges))).astype(np.uint8)
    changed[np.arange(len(challenges)), positions] ^= masks
    return changed.reshape(challenges.shape)


@torch.inference_mode()
def evaluate(model, challenges, depths, batch_size, *, bits_only=False):
    """One rollout per batch; retain readouts only at requested depths."""
    device = model.kernel.device
    collected = {d: [] for d in depths}
    schedule = model.schedule(max(depths))
    for start in range(0, len(challenges), batch_size):
        symbols = torch.as_tensor(challenges[start:start + batch_size], device=device)
        state = model.encode(symbols)
        for step in range(1, max(depths) + 1):
            state = model.step(state, step, schedule[step - 1])
            if step in collected:
                readout = model.readout(state)
                if not torch.isfinite(state).all():
                    raise FloatingPointError(f"Non-finite state at depth {step}; try normalization.")
                result = (readout >= 0).to(torch.uint8) if bits_only else readout
                collected[step].append(result.cpu().numpy())
    return {d: np.concatenate(parts) for d, parts in collected.items()}


def response_bits(logits):
    # Zero is assigned bit 1; exact ties are explicitly measured below.
    return (logits >= 0).astype(np.uint8)


def comparison(predicted, actual):
    equal = (predicted == actual).reshape(len(actual), -1)
    exact = equal.all(axis=1)
    return {"bit_accuracy": float(equal.mean()), "exact_successes": int(exact.sum()),
            "challenges": len(actual), "exact_success_rate": float(exact.mean())}


def hamming(a, b):
    fractions = (a != b).reshape(len(a), -1).mean(axis=1)
    return {"mean": float(fractions.mean()), "min": float(fractions.min()),
            "p05": float(np.quantile(fractions, .05)), "median": float(np.median(fractions)),
            "p95": float(np.quantile(fractions, .95)), "max": float(fractions.max())}


def collision_stats(challenges, bits):
    # np.unique(axis=0) constructs enormous structured dtypes for large rows.
    # Bytes keys retain exact equality without a probabilistic hash shortcut.
    seen, counts = set(), {}
    for challenge, response in zip(challenges, bits):
        key = np.asarray(challenge, dtype=np.uint8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        packed = np.packbits(response.reshape(-1), bitorder="big").tobytes()
        counts[packed] = counts.get(packed, 0) + 1
    return {"distinct_challenges": len(seen), "distinct_responses": len(counts),
            "colliding_pairs_of_distinct_challenges": sum(c * (c - 1) // 2 for c in counts.values()),
            "largest_response_multiplicity": max(counts.values(), default=0)}


def distribution_stats(bits, correlation_bits=512):
    """Exact marginal statistics and covariance dimension, bounded pair correlations.

    tr((X.T X)^2) == tr((X X.T)^2): work in sample space instead of
    allocating a response_bits squared covariance matrix. Columns are streamed.
    Correlations use at most correlation_bits evenly spaced output coordinates.
    """
    flat = bits.reshape(len(bits), -1)
    count = flat.shape[1]
    gram = np.zeros((len(bits), len(bits)), dtype=np.float64)
    ones = bias = entropy_sum = constants = trace = 0.
    worst = 0.
    for start in range(0, count, 8192):
        chunk = flat[:, start:start + 8192].astype(np.float64)
        p = chunk.mean(axis=0)
        centered = chunk - p
        gram += centered @ centered.T / len(bits)
        variance = (centered * centered).mean(axis=0)
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
        entropy[(p == 0) | (p == 1)] = 0
        ones += p.sum()
        bias += np.abs(p - .5).sum()
        worst = max(worst, float(np.abs(p - .5).max()))
        entropy_sum += entropy.sum()
        constants += (variance <= 1e-12).sum()
        trace += variance.sum()
    selected = np.linspace(0, count - 1, min(count, correlation_bits), dtype=np.int64)
    columns = flat[:, selected].astype(np.float64)
    columns -= columns.mean(axis=0)
    covariance = columns.T @ columns / len(bits)
    variance = covariance.diagonal()
    varying = variance > 1e-12
    covariance = covariance[varying][:, varying]
    variance = variance[varying]
    correlation = covariance / np.sqrt(variance[:, None] * variance[None, :])
    off = np.abs(correlation[np.triu_indices(len(variance), 1)])
    denominator = float((gram * gram).sum())
    return {"mean_one_fraction": float(ones / count),
            "mean_absolute_per_bit_bias": float(bias / count),
            "worst_absolute_per_bit_bias": worst,
            "mean_marginal_bit_entropy": float(entropy_sum / count),
            "constant_bit_fraction": float(constants / count),
            "mean_absolute_pair_correlation": float(off.mean()) if off.size else None,
            "pair_correlation_coordinates": len(selected),
            "pair_correlation_is_sampled": len(selected) < count,
            "covariance_participation_ratio": float(trace * trace / denominator) if denominator > 0 else 0.}



def numerical_probe(model, challenges, raw, depths, samples):
    """Separate batch-shape effects from an arithmetic-precision perturbation."""
    if samples == 0:
        return None
    data = challenges[:samples]
    single = evaluate(model, data, depths, 1)
    double = evaluate(copy.deepcopy(model).double(), data, depths, 1)
    rows = []
    for depth in depths:
        single_bits = response_bits(single[depth])
        rows.append({"depth": depth,
                     "sweep_vs_single_float32": comparison(single_bits, response_bits(raw[depth][:len(data)])),
                     "single_float32_vs_float64": comparison(response_bits(double[depth]), single_bits)})
    return {"challenges": len(data), "depth_sweep": rows,
            "interpretation": "Numerical agreement tests on the same device, not a cross-device guarantee. "
                              "Float64 uses the same initially float32-valued key and recomputes the dynamics."}


def avalanche_probe(model, args, forbidden):
    """Probe input bits one at a time; cap stored output coordinates for large grids."""
    if args.avalanche_inputs == 0:
        return None
    nbits = model.state_bits
    rng = np.random.default_rng(args.seed + 51)
    indices = rng.choice(nbits, size=min(args.avalanche_inputs, nbits), replace=False)
    output_indices = (np.arange(nbits) if nbits <= args.avalanche_output_bits else
                      np.sort(rng.choice(nbits, args.avalanche_output_bits, replace=False)))
    base = sample_challenges(args.avalanche_samples, args.size, args.seed + 52,
                             forbidden=forbidden, colors=args.colors, frames=args.frames)
    depth = max(args.depths)
    # Paired calls have identical batch positions and shapes.
    reference = evaluate(model, base, [depth], args.batch_size, bits_only=True)[depth].reshape(len(base), -1)
    reference = reference[:, output_indices]
    rows = []
    for index in indices:
        plane, position = divmod(int(index), args.frames * args.size**2)
        altered = base.copy().reshape(len(base), -1)
        altered[:, position] ^= 1 << plane
        changed = evaluate(model, altered.reshape(base.shape), [depth], args.batch_size, bits_only=True)[depth]
        rows.append((changed.reshape(len(base), -1)[:, output_indices] != reference).mean(axis=0))
    matrix = np.stack(rows)
    ideal = rng.binomial(len(base), .5, size=matrix.shape) / len(base)
    return {"matrix": matrix.astype(np.float32), "input_indices": indices, "output_indices": output_indices,
            "stats": {"tested_input_bits": len(indices), "tested_output_bits": len(output_indices),
                      "total_output_bits": nbits, "samples_per_input_bit": len(base), "depth": depth,
                      "mean_flip_probability": float(matrix.mean()),
                      "mean_absolute_deviation_from_half": float(np.abs(matrix - .5).mean()),
                      "random_reference_deviation": float(np.abs(ideal - .5).mean()),
                      "observed_zero_probability_fraction": float((matrix == 0).mean()),
                      "observed_one_probability_fraction": float((matrix == 1).mean()),
                      "bit_index_order": "low-bit plane first, then frame, row, column"}}



class Surrogate(nn.Module):
    """A deliberately small black-box attacker; it never receives teacher weights."""

    def __init__(self, width=16, colors=4):
        super().__init__()
        self.colors = colors
        planes = colors.bit_length() - 1
        self.layers = nn.Sequential(
            nn.Conv2d(planes, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, planes, 1),
        )

    def forward(self, symbols):
        return self.layers(bipolar(symbols, self.colors))


def fit_surrogate(train_x, train_bits, test_x, test_bits, device, epochs, batch_size, seed, colors=4):
    torch.manual_seed(seed)
    model = Surrogate(colors=colors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.as_tensor(train_x)
    y = torch.as_tensor(train_bits, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed + 1)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=rng)
        loss_sum = torch.zeros((), device=device)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[indices].to(device)), y[indices].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(indices)
        losses.append(float((loss_sum / len(x)).item()))
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(test_x), batch_size):
            batch = torch.as_tensor(test_x[start:start + batch_size], device=device)
            predictions.append((model(batch) >= 0).cpu().numpy().astype(np.uint8))
    majority = (train_bits.mean(axis=0, keepdims=True) >= .5).astype(np.uint8)
    return {"training_queries": len(train_x), "epochs": epochs,
            "attacker_parameters": sum(p.numel() for p in model.parameters()),
            "epoch_bce_losses": losses,
            "surrogate": comparison(np.concatenate(predictions), test_bits),
            "training_majority_baseline": comparison(np.broadcast_to(majority, test_bits.shape), test_bits)}


def simulate_capture(challenges, seed, noise_std, cell_pixels=8, colors=4):
    """Decode each sequence through the shared aligned cell-mean camera model."""
    return np.stack([camera_capture(sequence, colors, seed + i, noise_std, cell_pixels)
                     for i, sequence in enumerate(challenges)])



def save_svg(path, challenges, outputs, colors=4):
    """Compact SVG overview with lossless embedded PNGs for every sequence frame."""
    frames = challenges.shape[1]
    tile, gap = 112, 20
    depths = list(outputs)
    samples = min(4, len(challenges))
    columns = [challenges] + [bits_to_symbols(b, colors, frames) for b in outputs.values()]
    width = gap + (len(columns) + 1) * (tile + gap)
    height = 48 + samples * frames * (tile + 28)
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(width), height=str(height),
                     viewBox=f"0 0 {width} {height}")
    ET.SubElement(svg, "rect", width="100%", height="100%", fill="#f4f6f8")
    def label(x, y, text):
        ET.SubElement(svg, "text", x=str(x), y=str(y), fill="#18202d",
                      **{"font-family": "sans-serif", "font-size": "12"}).text = text
    for c, text in enumerate(["Challenge"] + [f"Response t={d}" for d in depths]):
        label(gap + (c + 1) * (tile + gap), 25, text)
    for sample in range(samples):
        for frame in range(frames):
            y = 40 + (sample * frames + frame) * (tile + 28)
            label(gap, y + 20, f"Sample {sample + 1}, frame {frame + 1}")
            for c, grids in enumerate(columns):
                data = png(grids[sample, frame].astype(np.uint8), palette(colors))
                ET.SubElement(svg, "image", x=str(gap + (c + 1) * (tile + gap)), y=str(y),
                              width=str(tile), height=str(tile), style="image-rendering:pixelated",
                              href="data:image/png;base64," + base64.b64encode(data).decode("ascii"))
    ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)



@torch.inference_mode()
def save_trajectory(path, model, challenge, steps, memory_mb=32):
    """Record evenly spaced full states within an explicit uncompressed byte budget."""
    if memory_mb <= 0:
        if path.exists():
            path.unlink()
        return {"saved": False, "reason": "disabled"}
    x = model.encode(torch.as_tensor(challenge[None], device=model.kernel.device))
    def display_state(state):
        return model.readout(state) if model.architecture == "feistel" else state
    value = display_state(x)
    snapshot_bytes = value.numel() * value.element_size()
    count = min(steps + 1, int(memory_mb * 1024**2) // snapshot_bytes)
    if count < 2:
        if path.exists():
            path.unlink()
        return {"saved": False, "reason": "budget cannot hold initial and final states",
                "snapshot_bytes": snapshot_bytes}
    selected = np.unique(np.linspace(0, steps, count, dtype=int))
    states = [value[0].cpu().numpy().copy()]
    for step, weight in enumerate(model.schedule(steps), 1):
        x = model.step(x, step, weight)
        if step in selected:
            states.append(display_state(x)[0].cpu().numpy().copy())
    np.savez_compressed(path, challenge=challenge, states=np.stack(states), steps=selected,
                        colors=model.colors, frames=model.frames, palette=palette(model.colors))
    return {"saved": True, "steps": selected.tolist(), "uncompressed_state_bytes": count * snapshot_bytes}



@torch.inference_mode()
def check_inverse(model, challenges, depth):
    symbols = torch.as_tensor(challenges, device=model.kernel.device)
    original = model.encode(symbols)
    state = original.clone()
    weights = model.schedule(depth)
    for t, weight in enumerate(weights, 1):
        state = model.step(state, t, weight)
    for t in range(depth, 0, -1):
        state = model.inverse_step(state, t, weights[t - 1])
    success = (state == original).flatten(1).all(1)
    if not success.all():
        raise AssertionError("Feistel inversion failed in this execution environment.")
    return {"challenges": len(challenges), "exact_reconstructions": int(success.sum()), "depth": depth}


def self_test():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(13)
    x = torch.randn(3, 6, 8, 8, generator=rng, dtype=torch.float64)
    torch.testing.assert_close(normalize_maps(x, "instance"), F.instance_norm(x), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(normalize_maps(x, "layer"), F.layer_norm(x, x.shape[1:]), rtol=1e-10, atol=1e-10)
    for mode in ("instance", "layer", "mixed"):
        assert normalize_maps(torch.ones_like(x), mode).count_nonzero() == 0
    symbols = torch.arange(4).reshape(1, 1, 2, 2)
    assert torch.equal(bits_to_symbols(symbol_bits(symbols)), symbols)
    data = sample_challenges(8, 8, 10)
    changed = flip_one_bit(data, 11)
    assert np.all(np.unpackbits((data ^ changed).reshape(8, -1), axis=1).sum(1) == 1)
    for activation in ("linear", "tanh"):
        model = NeuralKey(activation=activation)
        a = evaluate(model, data, [1, 4], 8)
        b = evaluate(model, data ^ 3, [1, 4], 8)
        split = evaluate(model, data, [1, 4], 1)
        for d in a:
            np.testing.assert_allclose(a[d], -b[d], rtol=1e-5, atol=2e-6)
            np.testing.assert_allclose(a[d], split[d], rtol=1e-5, atol=2e-6)
    circular = NeuralKey(padding="circular")
    shifted = evaluate(circular, np.roll(data, 1, axis=-1), [4], 8)[4]
    expected = np.roll(evaluate(circular, data, [4], 8)[4], 1, axis=3)
    np.testing.assert_allclose(shifted, expected, rtol=1e-5, atol=2e-6)
    duplicate_inputs = np.concatenate((data[:1], data[:2]))
    zero_bits = np.zeros((3, 2, 8, 8), dtype=np.uint8)
    assert collision_stats(duplicate_inputs, zero_bits)["colliding_pairs_of_distinct_challenges"] == 1
    capture = simulate_capture(data, 123, 0.)
    assert np.array_equal(capture, data)
    orthogonal = NeuralKey(init="orthogonal", gain=1.3, rotation="bilinear", rotation_period=4)
    flat = orthogonal.kernel.flatten(1)
    torch.testing.assert_close(flat @ flat.T, 1.3**2 * torch.eye(8), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(orthogonal.weight_at(4), orthogonal.kernel, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(orthogonal.weight_at(1), orthogonal.weight_at(5))
    assert not torch.allclose(orthogonal.weight_at(1), orthogonal.kernel)
    orthogonal.rotation_renorm = True
    torch.testing.assert_close(orthogonal.weight_at(1).flatten(1).norm(dim=1), flat.norm(dim=1))
    full = evaluate(orthogonal, data, [4], 8)[4]
    split = evaluate(orthogonal, data, [4], 1)[4]
    np.testing.assert_allclose(full, split, rtol=1e-5, atol=2e-6)
    for size, frames in ((8, 1), (16, 3), (9, 2), (8, 8)):
        for colors in (2, 4, 8, 16):
            sequence = sample_challenges(2, size, 19, colors=colors, frames=frames)
            tensor = torch.as_tensor(sequence)
            assert torch.equal(bits_to_symbols(symbol_bits(tensor, colors), colors, frames), tensor)
            assert np.all(np.unpackbits((sequence ^ flip_one_bit(sequence, 31, colors)).reshape(2, -1), axis=1).sum(1) == 1)
            for shuffle in (False, True):
                reversible = NeuralKey(size=size, colors=colors, frames=frames, architecture="feistel", shuffle=shuffle,
                                       init="orthogonal", rotation="bilinear", rotation_period=4)
                check_inverse(reversible, sequence, 5)
            capture = simulate_capture(sequence, 123, 0., colors=colors)
            assert np.array_equal(capture, sequence)
    repeated = np.tile(np.arange(16) % 2, (8, 1)).T.reshape(16, 2, 2, 2).astype(np.uint8)
    assert distribution_stats(repeated)["covariance_participation_ratio"] == 1.
    assert distribution_stats(np.zeros_like(repeated))["mean_marginal_bit_entropy"] == 0.
    print("Self-test passed: normalization, bit mapping, odd/shift symmetry, batch independence, "
          "collision counting, capture decoding, orthogonal rows, rotation/wrapping/gain, "
          "all palettes, multi-frame Feistel inversion and dependence diagnostics.")


def run(args):
    started = time.perf_counter()
    args = copy.deepcopy(args)
    prepare_workload(args)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable. Use --device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    def make_model(key_seed):
        return NeuralKey(args.channels, key_seed, args.norm, args.activation, args.padding,
                         size=args.size, colors=args.colors, frames=args.frames,
                         architecture=args.architecture, init=args.init, gain=args.gain,
                         kernel_size=args.kernel_size, rotation=args.rotation,
                         rotation_period=args.rotation_period, rotation_renorm=args.rotation_renorm,
                         activation_gain=args.activation_gain, phase=args.phase, shuffle=args.shuffle).to(device)

    model = make_model(args.key_seed)
    altered = copy.deepcopy(model)
    # A single scalar change, kept distinct from the independent-key comparison.
    center = args.kernel_size // 2
    altered.kernel[0, 0, center, center] += args.key_delta
    actual_delta = float((altered.kernel[0, 0, center, center] - model.kernel[0, 0, center, center]).item())
    if actual_delta == 0:
        raise SystemExit("key-delta rounds to zero in float32; choose a larger value.")
    independent = make_model(args.key_seed + 1)
    challenges = sample_challenges(args.samples, args.size, args.seed + 10, colors=args.colors, frames=args.frames)
    print(f"{device} | {args.frames} frame(s) x {args.size}x{args.size}, {args.colors} colors, {args.channels} channels | "
          f"{model.key_coefficients} key coefficients | {args.architecture}/{args.init}/{args.rotation} | "
          f"{args.norm}/{args.activation}/{args.padding}", flush=True)
    print("Evaluating depth sweep and input/key/symmetry probes...", flush=True)
    raw = evaluate(model, challenges, args.depths, args.batch_size)
    responses = {d: response_bits(v) for d, v in raw.items()}
    changed = evaluate(model, flip_one_bit(challenges, args.seed + 11, args.colors), args.depths, args.batch_size, bits_only=True)
    key_changed = evaluate(altered, challenges, args.depths, args.batch_size, bits_only=True)
    other_key = evaluate(independent, challenges, args.depths, args.batch_size, bits_only=True)
    complement = evaluate(model, challenges ^ (args.colors - 1), args.depths, args.batch_size, bits_only=True)
    shifted = evaluate(model, np.roll(challenges, 1, axis=-1), args.depths, args.batch_size, bits_only=True)
    rows = []
    print("depth   input flip   other input   key delta   new key   bit bias   bit H   cov dim   unique")
    for d in args.depths:
        bits = responses[d]
        row = {"depth": d,
               "input_bit_flip_hamming": hamming(bits, changed[d]),
               "independent_input_hamming": hamming(bits, np.roll(bits, 1, axis=0)),
               "one_weight_change_hamming": hamming(bits, key_changed[d]),
               "independent_key_hamming": hamming(bits, other_key[d]),
               "complement_prediction": comparison(1 - bits, complement[d]),
               "one_cell_shift_prediction": comparison(np.roll(bits, 1, axis=3), shifted[d]),
               "collisions": collision_stats(challenges, bits),
               **distribution_stats(bits),
               "zero_logit_fraction": float((raw[d] == 0).mean()),
               "absolute_logit_p01": float(np.quantile(np.abs(raw[d]), .01))}
        rows.append(row)
        means = [row[k]["mean"] * 100 for k in ("input_bit_flip_hamming", "independent_input_hamming",
                                                "one_weight_change_hamming", "independent_key_hamming")]
        print(f"{d:5d} {means[0]:11.2f}% {means[1]:12.2f}% {means[2]:10.2f}% {means[3]:8.2f}% "
              f"{100 * row['mean_absolute_per_bit_bias']:8.2f}% "
              f"{row['mean_marginal_bit_entropy']:7.3f} {row['covariance_participation_ratio']:9.1f} "
              f"{row['collisions']['distinct_responses']:6d}", flush=True)
    final_depth = max(args.depths)
    del changed, key_changed, other_key, complement, shifted, altered, independent
    constants = np.stack([np.full((args.frames, args.size, args.size), c, dtype=np.uint8) for c in range(args.colors)])
    constant_bits = evaluate(model, constants, [final_depth], args.batch_size, bits_only=True)[final_depth]
    random_bits = np.random.default_rng(args.seed + 60).integers(0, 2, size=responses[final_depth].shape, dtype=np.uint8)
    random_reference = distribution_stats(random_bits)
    print(f"Independent random-bit reference ({args.samples} samples): bias "
          f"{100 * random_reference['mean_absolute_per_bit_bias']:.2f}%, marginal H "
          f"{random_reference['mean_marginal_bit_entropy']:.3f}, covariance dimension "
          f"{random_reference['covariance_participation_ratio']:.1f}. "
          "These are not joint entropy or security bits.", flush=True)
    print(f"Structural probes at t={final_depth}: complement exact "
          f"{100 * rows[-1]['complement_prediction']['exact_success_rate']:.2f}%, "
          f"shift exact {100 * rows[-1]['one_cell_shift_prediction']['exact_success_rate']:.2f}%.", flush=True)
    probe = avalanche_probe(model, args, challenges)
    if probe is not None:
        s = probe["stats"]
        print(f"Avalanche matrix: mean flip {100 * s['mean_flip_probability']:.2f}%, "
              f"mean deviation from 50% {100 * s['mean_absolute_deviation_from_half']:.2f} points "
              f"(random sampling reference {100 * s['random_reference_deviation']:.2f}).", flush=True)
    with torch.inference_mode():
        norms = model.kernel.flatten(1).norm(dim=1)
        ratios = torch.stack([w.flatten(1).norm(dim=1) / norms.clamp_min(1e-12)
                              for w in model.schedule(final_depth)])
        normalized_rows = model.kernel.flatten(1) / norms[:, None].clamp_min(1e-12)
        gram = normalized_rows @ normalized_rows.T
        gram.fill_diagonal_(0)
    report = {"script_version": 4, "poc": "base_neural_cryptography", "config": {**vars(args), "out": str(args.out)},
              "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                              "dtype": "float32", "threads": torch.get_num_threads()},
              "key_coefficients": model.key_coefficients, "response_bits": model.state_bits,
              "sequence": {"frames": args.frames, "size": args.size, "colors": args.colors,
                           "bit_planes": model.bit_planes, "input_bits": model.state_bits,
                           "packing": "bit-plane, frame, row, column; frames stacked vertically",
                           "delivery": "ordered raw symbols; no packet headers or error correction"},
              "bijection_by_construction": args.architecture == "feistel",
              "readout_kind": "binary" if args.architecture == "feistel" else "continuous_thresholded",
              "initialization_diagnostics": {"max_absolute_filter_row_cosine": float(gram.abs().max().item()),
                                             "rotation_filter_norm_ratio_min": float(ratios.min().item()),
                                             "rotation_filter_norm_ratio_max": float(ratios.max().item())},
              "actual_one_weight_delta": actual_delta, "depth_sweep": rows,
              "uniform_color_probe": collision_stats(constants, constant_bits),
              "random_response_reference": random_reference,
              "interpretation": "Research diagnostics only. Seeds are test keys. Marginal entropy/covariance ignore higher-order dependence. "
                                "No observed collision/forgery is not a security proof. "
                                "Complement/shift predictions are structural probes, not attacks on a deployed protocol."}
    if probe is not None:
        report["avalanche_probe"] = probe["stats"]
    numerics = numerical_probe(model, challenges, raw, args.depths, args.numerics_samples)
    if numerics is not None:
        report["numerical_agreement"] = numerics
        last = numerics["depth_sweep"][-1]
        batch_agreement = last["sweep_vs_single_float32"]["exact_successes"]
        precision_agreement = last["single_float32_vs_float64"]["exact_successes"]
        print(f"Numerical agreement at t={final_depth}: batch vs single {batch_agreement}/{numerics['challenges']} exact; "
              f"float32 vs float64 {precision_agreement}/{numerics['challenges']} exact.", flush=True)
    if args.architecture == "feistel":
        report["inverse_check"] = check_inverse(model, challenges[:min(16, args.batch_size)], final_depth)
    if args.attack_epochs:
        print(f"Training small surrogate: {args.attack_samples} oracle queries, {args.attack_epochs} epochs...", flush=True)
        train_x = sample_challenges(args.attack_samples, args.size, args.seed + 20, forbidden=challenges,
                                    colors=args.colors, frames=args.frames)
        train_bits = evaluate(model, train_x, [final_depth], args.batch_size, bits_only=True)[final_depth]
        report["learned_attack"] = fit_surrogate(train_x, train_bits, challenges, responses[final_depth], device,
                                                 args.attack_epochs, args.batch_size, args.seed + 30, args.colors)
        attack = report["learned_attack"]["surrogate"]
        baseline = report["learned_attack"]["training_majority_baseline"]
        print(f"Surrogate: {100 * attack['bit_accuracy']:.2f}% bits, "
              f"{attack['exact_successes']}/{args.samples} exact responses. "
              f"Majority baseline: {100 * baseline['bit_accuracy']:.2f}% bits, "
              f"{baseline['exact_successes']}/{args.samples} exact.", flush=True)
    if args.camera:
        camera_x = challenges[:min(64, len(challenges))]
        decoded = simulate_capture(camera_x, args.seed + 40, args.camera_noise, args.cell_pixels, args.colors)
        camera_bits = evaluate(model, decoded, [final_depth], args.batch_size, bits_only=True)[final_depth]
        displayed_response = bits_to_symbols(camera_bits, args.colors, args.frames)
        captured_response = simulate_capture(displayed_response, args.seed + 140, args.camera_noise, args.cell_pixels, args.colors)
        received_bits = symbol_bits(torch.as_tensor(captured_response), args.colors).numpy()
        exact_decode = (decoded == camera_x).reshape(len(camera_x), -1).all(1)
        response_comparison = comparison(camera_bits, responses[final_depth][:len(camera_x)])
        report["camera"] = {"assumption": "Aligned, calibrated cell means; Gaussian RGB noise, gain and offset; no ECC/perspective.",
                            "symbol_accuracy": float((decoded == camera_x).mean()),
                            "exact_grid_decodes": int(exact_decode.sum()), "challenges": len(camera_x),
                            "response_agreement": response_comparison,
                            "response_symbol_accuracy": float((captured_response == displayed_response).mean()),
                            "round_trip": comparison(received_bits, responses[final_depth][:len(camera_x)])}
        print(f"Camera simulation: {int(exact_decode.sum())}/{len(camera_x)} exact challenge sequences, "
              f"{report['camera']['round_trip']['exact_successes']}/{len(camera_x)} exact round-trip responses.", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    save_svg(args.out / "examples.svg", challenges, responses, args.colors)
    save_sequence(args.out, "challenge", challenges[0], args.colors, args.fps)
    save_sequence(args.out, "response", bits_to_symbols(responses[final_depth][:1], args.colors, args.frames)[0], args.colors, args.fps)
    report["trajectory"] = save_trajectory(args.out / "trajectory.npz", model, challenges[0], final_depth, args.trajectory_mb)
    if probe is not None:
        np.savez_compressed(args.out / "avalanche.npz", probabilities=probe["matrix"],
                            input_bit_indices=probe["input_indices"], output_bit_indices=probe["output_indices"],
                            samples_per_input_bit=args.avalanche_samples)
    elif (args.out / "avalanche.npz").exists():
        (args.out / "avalanche.npz").unlink()
    report["elapsed_seconds"] = time.perf_counter() - started
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Finished in {report['elapsed_seconds']:.2f}s. Results: {args.out.resolve()}")
    return report


def run_comparison(args):
    """Paired architecture comparisons: all conditions share challenge datasets."""
    conditions = [
        ("01_gaussian_fixed", dict(init="gaussian", rotation="none")),
        ("02_orthogonal_fixed", dict(init="orthogonal", rotation="none")),
        ("03_orthogonal_rotating", dict(init="orthogonal", rotation="bilinear")),
        ("04_rotating_rescaled", dict(init="orthogonal", rotation="bilinear", rotation_renorm=True)),
        ("05_rotating_instance", dict(init="orthogonal", rotation="bilinear", norm="instance")),
        ("06_rotating_sine", dict(init="orthogonal", rotation="bilinear", norm="instance", activation="sine")),
        ("07_feistel_rotating", dict(init="orthogonal", rotation="bilinear", architecture="feistel")),
    ]
    all_rows = []
    summaries = []
    for name, overrides in conditions:
        trial = copy.deepcopy(args)
        trial.compare = False
        trial.architecture = "wave"
        trial.rotation_renorm = False
        for key, value in overrides.items():
            setattr(trial, key, value)
        trial.out = args.out / name
        print(f"\n{name}", flush=True)
        result = run(trial)
        summaries.append({"condition": name, "config": result["config"],
                          "final_depth": result["depth_sweep"][-1],
                          "random_response_reference": result["random_response_reference"],
                          "numerical_agreement": result.get("numerical_agreement"),
                          "elapsed_seconds": result["elapsed_seconds"]})
        for row in result["depth_sweep"]:
            all_rows.append({"condition": name, "depth": row["depth"],
                             "input_flip_probability": row["input_bit_flip_hamming"]["mean"],
                             "independent_input_distance": row["independent_input_hamming"]["mean"],
                             "key_delta_flip_probability": row["one_weight_change_hamming"]["mean"],
                             "independent_key_distance": row["independent_key_hamming"]["mean"],
                             "mean_absolute_bit_bias": row["mean_absolute_per_bit_bias"],
                             "mean_marginal_bit_entropy": row["mean_marginal_bit_entropy"],
                             "mean_absolute_pair_correlation": row["mean_absolute_pair_correlation"],
                             "covariance_participation_ratio": row["covariance_participation_ratio"],
                             "distinct_responses": row["collisions"]["distinct_responses"],
                             "complement_exact_prediction": row["complement_prediction"]["exact_success_rate"]})
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out / "comparison.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nFinal-depth comparison:                  input flip    bit bias    cov dim   batch exact", flush=True)
    for summary in summaries:
        row = summary["final_depth"]
        numerics = summary["numerical_agreement"]
        agreement = numerics["depth_sweep"][-1]["sweep_vs_single_float32"] if numerics else None
        exact = f"{agreement['exact_successes']}/{agreement['challenges']}" if agreement else "disabled"
        print(f"{summary['condition']:38s} {100 * row['input_bit_flip_hamming']['mean']:8.2f}% "
              f"{100 * row['mean_absolute_per_bit_bias']:10.2f}% {row['covariance_participation_ratio']:10.1f} {exact:>13s}")


def prepare_workload(args):
    """Preflight major arrays; reduce batches, never sample/depth/attack counts.

    This is an approximate working-array budget, not a process RSS limit. Torch,
    BLAS, convolution workspaces and the Python runtime add their own overhead.
    """
    bits = args.frames * args.size**2 * (args.colors.bit_length() - 1)
    ndepths = len(args.depths)
    extra = (6 * args.attack_samples if args.attack_epochs else 0)
    extra += 6 * args.avalanche_samples if args.avalanche_inputs else 0
    extra += 10 * ndepths * min(args.numerics_samples, args.samples)
    retained = bits * (args.samples * (10 * ndepths + 8) + extra)
    # Up to three model copies, each with a permutation and its inverse.
    retained += 48 * bits if args.architecture == 'feistel' else 0
    retained += min(args.trajectory_mb * 1024**2, (max(args.depths) + 1) * args.channels * args.frames * args.size**2 * 4)
    # The uniform-color probe has one challenge for every palette entry.
    retained += bits * args.colors * 3
    per_example = (args.channels * args.frames * args.size**2 * 4 *
                   max(1., (args.colors.bit_length() - 1) / 2 if args.architecture == 'feistel' else 1.) * 12)
    budget = args.max_memory_mb * 1024**2
    maximum_batch = int((budget - retained) // max(1, per_example))
    if maximum_batch < 1:
        raise ValueError(
            f'Estimated working arrays exceed --max-memory-mb {args.max_memory_mb:g}. '
            'Reduce --samples, --depths, --channels, --attack-samples; optional probes can be disabled with '
            '--attack-epochs 0 --avalanche-inputs 0 --numerics-samples 0 --trajectory-mb 0. '
            'Raise the budget only if your machine has enough memory.')
    args.requested_batch_size = args.batch_size
    args.batch_size = min(args.batch_size, maximum_batch)
    args.estimated_working_mb = (retained + args.batch_size * per_example) / 1024**2
    if args.batch_size != args.requested_batch_size:
        print(f'Memory preflight: batch size {args.requested_batch_size} -> {args.batch_size}; '
              'sample counts and depths unchanged.', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_optical_arguments(p, size=8, colors=4, frames=1, frame_help='matrices per challenge/response sequence, 1..8')
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--depths', default='1,4,8,16,32')
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--norm', choices=('instance', 'layer', 'mixed', 'none'), default='layer')
    p.add_argument('--activation', choices=('linear', 'relu', 'tanh', 'silu', 'sine'), default='relu')
    p.add_argument('--padding', choices=('zeros', 'circular'), default='zeros')
    p.add_argument('--architecture', choices=('wave', 'feistel'), default='wave')
    p.add_argument('--init', choices=('gaussian', 'orthogonal'), default='gaussian')
    p.add_argument('--gain', type=float, default=1.)
    p.add_argument('--kernel-size', type=int, default=3)
    p.add_argument('--rotation', choices=('none', 'bilinear'), default='none')
    p.add_argument('--rotation-period', type=int, default=64)
    p.add_argument('--rotation-renorm', action='store_true')
    p.add_argument('--activation-gain', type=float, default=3.)
    p.add_argument('--phase', type=float, default=.5)
    p.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--key-delta', type=float, default=1e-3)
    p.add_argument('--attack-samples', type=int, default=1024)
    p.add_argument('--attack-epochs', type=int, default=15, help='0 disables surrogate training')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--key-seed', type=int)
    p.add_argument('--avalanche-inputs', type=int, default=16, help='0 disables the matrix probe')
    p.add_argument('--avalanche-samples', type=int, default=16)
    p.add_argument('--avalanche-output-bits', type=int, default=4096, help='maximum sampled columns in the avalanche matrix')
    p.add_argument('--numerics-samples', type=int, default=16, help='0 disables precision/batch agreement probes')
    p.add_argument('--trajectory-mb', type=float, default=32., help='maximum uncompressed trajectory states; 0 disables')
    p.add_argument('--max-memory-mb', type=float, default=1024., help='approximate working-array budget, excluding runtime overhead')
    p.add_argument('--out', type=Path, default=Path('neural_key_results'))
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    validate_optical_arguments(p, args)
    try:
        args.depths = sorted(set(int(d) for d in args.depths.split(',')))
    except ValueError:
        p.error('depths must be comma-separated integers')
    if not args.depths or min(args.depths) < 1:
        p.error('depths must be positive')
    if args.channels < max(2, args.colors.bit_length() - 1) or args.samples < 2:
        p.error('channels must be at least max(2, log2(colors)); samples must be >=2')
    if min(args.batch_size, args.threads, args.attack_samples) < 1 or args.attack_epochs < 0:
        p.error('batch-size/threads/attack-samples must be positive; attack-epochs must be nonnegative')
    if args.architecture == 'feistel' or args.compare:
        if (args.colors.bit_length() - 1) * args.frames * args.size**2 % 2:
            p.error('Feistel requires an even total bit count: use even size or frames, or 4/16 colors')
    if args.kernel_size < 1 or args.kernel_size % 2 != 1 or args.rotation_period < 1:
        p.error('kernel-size must be positive/odd; rotation-period must be positive')
    min_width = args.size // 2 if ((args.colors.bit_length() - 1) * args.frames) % 2 else args.size
    if args.padding == 'circular' and args.kernel_size // 2 > min_width:
        p.error('circular padding may not wrap more than once; reduce kernel-size')
    if any(not math.isfinite(v) or v <= 0 for v in (args.gain, args.activation_gain, args.key_delta, args.max_memory_mb)):
        p.error('gain, activation-gain, key-delta and max-memory-mb must be finite and positive')
    if not math.isfinite(args.phase) or not math.isfinite(args.trajectory_mb) or args.trajectory_mb < 0:
        p.error('phase must be finite; trajectory-mb must be finite/nonnegative')
    if args.avalanche_inputs < 0 or args.avalanche_samples < 2 or args.avalanche_output_bits < 1 or args.numerics_samples < 0:
        p.error('invalid avalanche/numerical probe settings')
    if not 0 <= args.seed < 2**63 - 1000:
        p.error('seed must be in [0,2**63-1000)')
    if args.key_seed is None:
        args.key_seed = args.seed + 1
    if not 0 <= args.key_seed < 2**63 - 2:
        p.error('key-seed must be in [0,2**63-2)')
    return args


def main():
    args = parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.compare:
            run_comparison(args)
        else:
            run(args)
    except (ValueError, OSError, FloatingPointError) as exc:
        raise SystemExit(f'Base neural cryptography POC: {exc}') from exc


if __name__ == '__main__':
    main()
