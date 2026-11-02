"""Multiframe optical classifiers with a shared browser tensor contract.

All profiles accept stacked RGB, [N,3*T,4G,4G], and predict physical palette
logits, [N,16,G,G]. None receives keys, labels, or expected payload bytes.
"""

import torch
from torch import nn
from torch.nn import functional as F

from contract import CHANNELS, FRAME_COUNTS, MAX_FRAMES, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX


MODEL_PROFILES = ("compact", "wide", "normalized-compact", "normalized-tube")
DEFAULT_MODEL_PROFILE = "normalized-compact"


class ChannelLayerNorm(nn.Module):
    """Normalize learned channels at each spatial position, not camera RGB.

    Explicit arithmetic exports as basic ONNX reductions/elementwise operators.
    There are no running statistics and no batch or image-size dependence.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=1, keepdim=True)
        variance = (centered * centered).mean(dim=1, keepdim=True)
        return centered / torch.sqrt(variance + self.eps) * self.weight + self.bias


def conv(in_channels: int, out_channels: int, stride: int = 1,
         normalize: bool = False) -> nn.Sequential:
    layers = [nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)]
    if normalize:
        layers.append(ChannelLayerNorm(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OpticalDecoder(nn.Module):
    """Baseline, wide, and fully feature-normalized compact/tubular U-Nets.

    Normalized-compact uses a 16-channel inlet, a 16/32/64 backbone, and channel
    LayerNorm after every hidden convolution, with no RGB-to-logits branch.
    Normalized-tube keeps all hidden stage outputs at 32 channels; skip
    concatenations are reduced from 64 to 32 by their next convolution.
    Compact and wide remain explicit experimental profiles. The constructor's
    compact default identifies the original baseline; fresh training/export use
    DEFAULT_MODEL_PROFILE. All profiles clip input only at the model boundary.
    """

    def __init__(self, profile: str = "compact", frames: int = 1) -> None:
        super().__init__()
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Choose model profile from {MODEL_PROFILES}")
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Choose 1 to {MAX_FRAMES} input frames")
        self.profile = profile
        self.frames = frames
        width = 32 if profile in ("wide", "normalized-tube") else 16
        inlet_width = 32 if profile == "normalized-tube" else (8 if profile == "compact" else 16)
        middle_width = width if profile == "normalized-tube" else 2 * width
        context_width = width if profile == "normalized-tube" else 4 * width
        normalize = profile != "compact"
        normalize_inlet = profile in ("normalized-compact", "normalized-tube")
        self.inlet = nn.Sequential(conv(3 * frames, inlet_width, 2, normalize_inlet),
                                   conv(inlet_width, width, 2, normalize_inlet))
        self.e0 = nn.Sequential(conv(width, width, normalize=normalize), conv(width, width, normalize=normalize))
        self.e1 = nn.Sequential(conv(width, middle_width, 2, normalize), conv(middle_width, middle_width, normalize=normalize))
        self.bottleneck = nn.Sequential(conv(middle_width, context_width, 2, normalize), conv(context_width, context_width, normalize=normalize))
        self.d1 = nn.Sequential(conv(context_width + middle_width, middle_width, normalize=normalize), conv(middle_width, middle_width, normalize=normalize))
        self.d0 = nn.Sequential(conv(middle_width + width, width, normalize=normalize), conv(width, width, normalize=normalize))
        self.head = nn.Conv2d(width, 16, 1, bias=True)
        if profile == "wide":
            self.local_color = nn.Sequential(nn.Conv2d(3 * frames, 16, 1), nn.ReLU(), nn.Conv2d(16, 16, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # Preserve calibrated overshoot in recordings/preprocessing; every
        # learned layer consumes the same bounded range for real and synthetic
        # inputs. ONNX exports this operation, including browser camera inference.
        pixels = torch.clamp(pixels, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX)
        e0 = self.e0(self.inlet(pixels))
        e1 = self.e1(e0)
        context = self.bottleneck(e1)
        d1 = self.d1(torch.cat((F.interpolate(context, scale_factor=2, mode="nearest"), e1), dim=1))
        d0 = self.d0(torch.cat((F.interpolate(d1, scale_factor=2, mode="nearest"), e0), dim=1))
        logits = self.head(d0)
        if self.profile == "wide":
            logits = logits + self.local_color(F.avg_pool2d(pixels, kernel_size=4, stride=4))
        return logits


def profile_from_state(state: dict) -> str:
    """Identify one supported architecture; the checkpoint loader checks every tensor.

    Profiles share the current checkpoint representation. Loading is exact;
    intentional single-frame warm starts use initialize_frames separately.
    """
    if not isinstance(state, dict) or not isinstance(state.get("head.weight"), torch.Tensor):
        raise ValueError("Checkpoint has no valid decoder head")
    inlet = state.get("inlet.0.0.weight")
    if not isinstance(inlet, torch.Tensor) or inlet.ndim != 4:
        raise ValueError("Checkpoint has no valid decoder inlet")
    signature = (tuple(state["head.weight"].shape), inlet.shape[0],
                 "inlet.0.1.weight" in state, "e0.0.1.weight" in state,
                 "local_color.0.weight" in state)
    if signature == ((16, 16, 1, 1), 8, False, False, False):
        return "compact"
    if signature == ((16, 32, 1, 1), 16, False, True, True):
        return "wide"
    if signature == ((16, 16, 1, 1), 16, True, True, False):
        return "normalized-compact"
    if signature == ((16, 32, 1, 1), 32, True, True, False):
        return "normalized-tube"
    raise ValueError("Checkpoint model does not match a supported decoder profile")


def frames_from_state(state: dict) -> int:
    """Read the input width; remaining tensor shapes are checked by the loader."""
    weight = state.get("inlet.0.0.weight") if isinstance(state, dict) else None
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 4
            or weight.shape[1] % CHANNELS != 0 or weight.shape[1] // CHANNELS not in FRAME_COUNTS):
        raise ValueError("Checkpoint does not have a supported input frame count")
    return int(weight.shape[1] // CHANNELS)


def initialize_frames(model: OpticalDecoder, frames: int) -> OpticalDecoder:
    """Explicitly initialize a multiframe model from a trained single-frame model.

    Only the two RGB-facing convolutions change: every frame gets 1/T of
    the original weights. Repeating an identical input across all T slots
    therefore preserves the initial function, apart from floating-point rounding.
    All other parameters, including biases, are copied exactly. This is an
    intentional training initialization, never a permissive checkpoint reader.
    """
    if type(frames) is not int or frames not in FRAME_COUNTS:
        raise ValueError(f"Choose 1 to {MAX_FRAMES} input frames")
    if model.frames == frames:
        return model
    if model.frames != 1:
        raise ValueError("Changing the frame count requires a single-frame source checkpoint; "
                         "use the original pilot or keep the checkpoint's frame count")
    parameter = next(model.parameters())
    expanded = OpticalDecoder(model.profile, frames=frames).to(
        device=parameter.device, dtype=parameter.dtype)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    keys = ["inlet.0.0.weight"]
    if model.profile == "wide":
        keys.append("local_color.0.weight")
    for name in keys:
        state[name] = state[name].repeat(1, frames, 1, 1) / frames
    expanded.load_state_dict(state, strict=True)
    expanded.train(model.training)
    return expanded


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
