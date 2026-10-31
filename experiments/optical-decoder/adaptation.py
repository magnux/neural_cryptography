"""Offline adaptation objective, mild real-camera augmentation, and scheduling.

No model or recording contract changes live here. Validation continues to use
ordinary data-cell cross-entropy and canonical exact recovery.
"""

import math
import numbers

import torch
from torch.nn import functional as F


def _finite_number(value, name, *, minimum=0., maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (positive and value <= minimum) \
            or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}")
    return value


def adaptation_loss(logits, labels, mask, colors, hard_weight=.1, hard_fraction=.25):
    """Return mean per-frame blended CE and the ordinary CE for each frame.

    Each frame contributes equally, irrespective of its valid-cell count. The
    hardest term selects ceil(hard_fraction * valid_cells) within each frame;
    reference cells and inactive palette classes receive no gradient.
    """
    hard_weight = _finite_number(hard_weight, "hard_weight", maximum=1)
    hard_fraction = _finite_number(hard_fraction, "hard_fraction", maximum=1, positive=True)
    if isinstance(colors, bool) or colors not in (4, 16):
        raise ValueError("A 4- or 16-color palette is required")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[0] < 1 \
            or logits.shape[1] < colors or not logits.is_floating_point():
        raise ValueError("Expected floating NCHW logits with the active palette")
    target_shape = (logits.shape[0], logits.shape[2], logits.shape[3])
    if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != target_shape \
            or labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("Labels must be matching int64 NHW tensors on the logits device")
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != target_shape \
            or mask.dtype != torch.bool or mask.device != logits.device:
        raise ValueError("Mask must be a matching boolean NHW tensor on the logits device")
    counts = mask.flatten(1).sum(1)
    if bool(torch.any(counts == 0)):
        raise ValueError("Every frame needs at least one valid data cell")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Logits must be finite")
    valid_labels = labels[mask]
    if bool(torch.any((valid_labels < 0) | (valid_labels >= colors))):
        raise ValueError("Valid labels must belong to the active palette")
    # Masked cells may carry a sentinel label; they never enter the objective.
    losses = F.cross_entropy(logits[:, :colors], labels.masked_fill(~mask, 0), reduction="none")
    ordinary, blended = [], []
    for row, valid in zip(losses, mask):
        cells = row[valid]
        mean = cells.mean()
        ordinary.append(mean)
        if hard_weight:
            count = max(1, math.ceil(cells.numel()*hard_fraction))
            hard = cells.topk(count, sorted=False).values.mean()
            blended.append((1-hard_weight)*mean + hard_weight*hard)
        else:
            blended.append(mean)
    return torch.stack(blended).mean(), torch.stack(ordinary)


class OnlineRealAugment:
    """Mild photometric changes on CPU real captures, using an isolated RNG.

    Input/output are float32 NCHW in calibrated [-1.5,1.5] units. RGB values
    outside nominal camera [0,1] survive unless the final contract bound clips
    them. Exposure/white-balance draws are per frame; noise is per pixel. No
    geometry, labels, synthetic samples, or global RNG state are modified.
    """

    def __init__(self, seed=7, exposure_stops=.15, white_balance_stops=.1, noise_std=.005):
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed <= 0xffffffff:
            raise ValueError("Augmentation seed must be a uint32 integer")
        self.exposure_stops = _finite_number(exposure_stops, "exposure_stops", maximum=4)
        self.white_balance_stops = _finite_number(white_balance_stops, "white_balance_stops", maximum=4)
        self.noise_std = _finite_number(noise_std, "noise_std", maximum=1)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def __call__(self, pixels, real_mask):
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[1] != 3 \
                or pixels.dtype != torch.float32 or pixels.device.type != "cpu" \
                or pixels.shape[0] < 1 or min(pixels.shape[2:]) < 1:
            raise ValueError("Augmentation expects nonempty CPU float32 NCHW RGB tensors")
        if not bool(torch.isfinite(pixels).all()) or bool(torch.any(pixels.abs() > 1.50001)):
            raise ValueError("Augmentation input violates the normalized calibrated range")
        if isinstance(real_mask, torch.Tensor):
            if real_mask.dtype != torch.bool or real_mask.device.type != "cpu":
                raise ValueError("Real-sample mask must be boolean and on CPU")
            real = real_mask
        elif isinstance(real_mask, (list, tuple)) and all(isinstance(x, bool) for x in real_mask):
            real = torch.tensor(real_mask, dtype=torch.bool)
        else:
            raise ValueError("Real-sample mask must contain booleans")
        if tuple(real.shape) != (pixels.shape[0],):
            raise ValueError("Real-sample mask must have one entry per frame")
        output = pixels.clone()
        count = int(real.sum())
        if not count or not (self.exposure_stops or self.white_balance_stops or self.noise_std):
            return output
        rgb = (pixels[real]+1)/2
        # Draw all gain terms even when their strength is zero, supporting paired
        # comparisons without changing the subsequent sensor-noise realization.
        exposure = (2*torch.rand((count, 1, 1, 1), generator=self.generator)-1)*self.exposure_stops
        wb = 2*torch.rand((count, 3, 1, 1), generator=self.generator)-1
        wb -= wb.mean(dim=1, keepdim=True)
        wb /= wb.abs().amax(dim=1, keepdim=True).clamp_min(1)
        wb *= self.white_balance_stops
        rgb = rgb*torch.exp2(exposure+wb)
        if self.noise_std:
            rgb = rgb + self.noise_std*torch.randn(rgb.shape, generator=self.generator)
        output[real] = (2*rgb-1).clamp(-1.5, 1.5)
        return output


class PlateauController:
    """Track meaningful validation progress independently of checkpoint saving.

    The parent trainer saves any strict lexicographic best (exact rate, -CE).
    Patience here resets only for a larger exact rate, or equal exact rate with
    CE improved by more than min_delta relative to the last meaningful best.
    LR cuts reset their own counter, never the early-stopping counter.
    """

    def __init__(self, lr, min_lr=2e-6, factor=.5, lr_patience=4,
                 stop_patience=12, min_delta=.001):
        self.lr = _finite_number(lr, "lr", positive=True)
        minimum = _finite_number(min_lr, "min_lr", positive=True)
        if minimum > self.lr:
            raise ValueError("min_lr must not exceed lr")
        self.min_lr = minimum
        self.factor = _finite_number(factor, "factor", positive=True)
        if self.factor >= 1:
            raise ValueError("factor must be smaller than one")
        for name, value in (("lr_patience", lr_patience), ("stop_patience", stop_patience)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer; zero disables the control")
        self.lr_patience, self.stop_patience = int(lr_patience), int(stop_patience)
        self.min_delta = _finite_number(min_delta, "min_delta", maximum=1)
        self.best_exact = self.best_ce = None
        self.bad_checks = self.lr_bad_checks = self.reductions = 0

    def update(self, exact, ce):
        exact = _finite_number(exact, "exact", maximum=1)
        ce = _finite_number(ce, "ce")
        improved = self.best_exact is None or exact > self.best_exact or (
            exact == self.best_exact and self.best_ce-ce > self.min_delta*self.best_ce)
        reduced = False
        if improved:
            self.best_exact, self.best_ce = exact, ce
            self.bad_checks = self.lr_bad_checks = 0
        else:
            self.bad_checks += 1
            self.lr_bad_checks += 1
            if self.lr_patience and self.lr_bad_checks >= self.lr_patience:
                next_lr = max(self.min_lr, self.lr*self.factor)
                reduced = next_lr < self.lr
                self.lr = next_lr
                self.reductions += int(reduced)
                self.lr_bad_checks = 0
        return {"improved": improved, "lr": self.lr, "reduced": reduced,
                "shouldStop": bool(self.stop_patience and self.bad_checks >= self.stop_patience),
                "badChecks": self.bad_checks, "lrBadChecks": self.lr_bad_checks,
                "reductions": self.reductions}
