"""Training-only early-to-late feature distillation for W-tube and VW.

The supervised student keeps its ordinary deployment graph. A residual predictor
maps its first U to detached features of the EMA teacher's last U. The two views
share captures, slots, phase permutation and geometry; only additional mild
photometric/noise corruption distinguishes the student view.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from contract import NETWORK_INPUT_MIN, NETWORK_INPUT_MAX
from model import ResidualBlock, W_TUBE_PROFILES


def distillation_weight(update: int, maximum: float = .05, ramp_updates: int = 1000) -> float:
    """Linear warm-up by optimizer update, independent of gradient accumulation."""
    if (type(update) is not int or update < 0 or type(ramp_updates) is not int
            or ramp_updates < 0 or isinstance(maximum, bool)
            or not math.isfinite(maximum) or maximum < 0):
        raise ValueError("Distillation weight and update/ramp counts must be finite and nonnegative")
    return float(maximum) * (min(1., update/ramp_updates) if ramp_updates else 1.)


class StudentAugmentation:
    """Extra corruption after the shared weak view, preserving packed-slot identity.

    Noise standard deviation is in RGB01 units. Tint and exposure gains are
    shared by the whole burst; pixel noise is independent across frames. A
    separate device RNG leaves sampling and geometry-augmentation draws alone.
    This does not add flips, phase changes, frame dropping, blur or occlusions.
    """

    def __init__(self, *, seed: int, noise_std: float = .01,
                 tint_stops: float = .04, exposure_stops: float = .04):
        if type(seed) is not int or seed < 0:
            raise ValueError("Student augmentation seed must be a nonnegative integer")
        for name, value, maximum in (("noise_std", noise_std, 1.),
                                     ("tint_stops", tint_stops, 4.),
                                     ("exposure_stops", exposure_stops, 4.)):
            if (isinstance(value, bool) or not math.isfinite(value)
                    or not 0 <= value <= maximum):
                raise ValueError(f"Student augmentation {name} must be finite and within 0..{maximum:g}")
        self.seed = seed
        self.noise_std = float(noise_std)
        self.tint_stops = float(tint_stops)
        self.exposure_stops = float(exposure_stops)
        self.generators = {}

    def settings(self):
        return {"scope": "student-only-real-and-synthetic", "noiseStdRGB01": self.noise_std,
                "tintStops": self.tint_stops, "exposureStops": self.exposure_stops,
                "seed": self.seed, "geometryAndPhase": "shared-with-teacher",
                "gainSharing": "one-exposure-and-rgb-tint-per-stack",
                "noiseSharing": "independent-per-pixel-and-exposure",
                "outputClip": [NETWORK_INPUT_MIN, NETWORK_INPUT_MAX]}

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        if (pixels.ndim != 4 or pixels.shape[1] != 32
                or not pixels.is_floating_point()):
            raise ValueError("Student view requires floating packed [N,32,H,W] input")
        device_key = str(pixels.device)
        if device_key not in self.generators:
            self.generators[device_key] = torch.Generator(device=pixels.device).manual_seed(self.seed)
        generator = self.generators[device_key]
        batch, _, height, width = pixels.shape
        rgb = pixels[:, :24].reshape(batch, 8, 3, height, width)
        rgb = (rgb + 1.)*.5
        if self.tint_stops or self.exposure_stops:
            exposure = (torch.rand((batch, 1, 1, 1, 1), device=pixels.device,
                                   dtype=pixels.dtype, generator=generator)*2.-1.) * self.exposure_stops
            tint = (torch.rand((batch, 1, 3, 1, 1), device=pixels.device,
                               dtype=pixels.dtype, generator=generator)*2.-1.) * self.tint_stops
            rgb = rgb * torch.exp2(exposure+tint)
        if self.noise_std:
            rgb = rgb + self.noise_std * torch.randn(rgb.shape, device=pixels.device,
                                                     dtype=pixels.dtype, generator=generator)
        rgb = (rgb*2.-1.).clamp(NETWORK_INPUT_MIN, NETWORK_INPUT_MAX)
        presence = pixels[:, 24:, :1, :1].clamp(0., 1.).unsqueeze(2)
        rgb = (rgb*presence).reshape(batch, 24, height, width)
        return torch.cat((rgb, pixels[:, 24:]), dim=1)


def normalized_feature_loss(prediction: torch.Tensor, target: torch.Tensor,
                            data_mask: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance of unit channel vectors, averaged per example.

    The mask is the same already augmented logical-grid mask used by CE. Nearest
    expansion makes every subpixel of a data cell count equally and excludes
    pilots. Per-example normalization prevents larger grids dominating the
    scalar. The target is detached defensively even outside the EMA wrapper.
    """
    if (prediction.ndim != 4 or target.shape != prediction.shape
            or data_mask.ndim != 3 or data_mask.shape[0] != prediction.shape[0]
            or data_mask.dtype != torch.bool or prediction.shape[1] != 32
            or prediction.shape[-2:] != tuple(4*size for size in data_mask.shape[-2:])
            or not bool(data_mask.flatten(1).any(1).all())):
        raise ValueError("Feature loss requires [N,32,4G,4G] features and nonempty boolean [N,G,G] masks")

    # Sum, not channel mean: squared distance between unit vectors is 2-2*cos
    # for nonzero features, retaining a useful scale independent of width.
    cells = (F.normalize(prediction, p=2, dim=1, eps=1e-6)
             - F.normalize(target.detach(), p=2, dim=1, eps=1e-6)).square().sum(dim=1)
    mask = F.interpolate(data_mask[:, None].to(dtype=cells.dtype), size=cells.shape[-2:],
                         mode="nearest")[:, 0]
    per_example = (cells*mask).flatten(1).sum(1)/mask.flatten(1).sum(1)
    return per_example.mean()


class EMADistillation(nn.Module):
    """Frozen last-U teacher plus a trainable first-U residual predictor.

    Add only ``projector.parameters()`` to the student's optimizer. Call
    ``update(student)`` exactly once after each successful optimizer step, not
    after individual accumulation microbatches. This module is never attached
    to the deployment model or stored in its modelState.
    """

    def __init__(self, student, *, decay: float = .995):
        super().__init__()
        if student.profile not in W_TUBE_PROFILES:
            raise ValueError("EMA feature distillation requires W-tube or VW")
        if isinstance(decay, bool) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.decay = float(decay)
        self.teacher = copy.deepcopy(student).requires_grad_(False).eval()
        parameter = next(student.parameters())
        self.projector = ResidualBlock(32, pairs=1).to(device=parameter.device, dtype=parameter.dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def settings(self):
        return {"method": "ema-early-to-late-features", "emaDecay": self.decay,
                "studentTap": "first-u-post-fusion", "teacherTap": "last-u-post-fusion",
                "featureScale": "4G", "projector": "two-3x3-conv-residual-block",
                "projectorChannels": 32, "projectorNormalization": "one-channel-layer-norm",
                "distance": "squared-l2-between-unit-channel-vectors",
                "mask": "nearest-expanded-data-cell-mask", "reduction": "equal-per-example",
                "teacherGradient": False, "emaUpdate": "after-optimizer-step",
                "checkpoint": "student-only-teacher-and-projector-reinitialized-on-continuation"}

    def target(self, weak_pixels: torch.Tensor) -> torch.Tensor:
        """Compute before the student forward to reduce peak activation memory."""
        with torch.no_grad():
            return self.teacher.final_stage_features(weak_pixels)

    def loss(self, first_features: torch.Tensor, target: torch.Tensor,
             data_mask: torch.Tensor) -> torch.Tensor:
        return normalized_feature_loss(self.projector(first_features), target, data_mask)

    @torch.no_grad()
    def update(self, student) -> None:
        if student.profile != self.teacher.profile:
            raise ValueError("EMA teacher and student profiles must match")
        student_parameters = dict(student.named_parameters())
        for name, target in self.teacher.named_parameters():
            target.lerp_(student_parameters[name].detach(), 1.-self.decay)
        # Current correctors have only derived geometry buffers, with no running
        # batch statistics. Copy all buffers so the teacher follows exact state.
        student_buffers = dict(student.named_buffers())
        for name, target in self.teacher.named_buffers():
            target.copy_(student_buffers[name].detach())
        self.teacher.eval()

