"""Multiframe optical classifiers with a shared browser tensor contract.

Both profiles accept stacked RGB, [N,3*T,4G,4G], and predict physical palette
logits, [N,16,G,G]. Neither receives keys, labels, or expected payload bytes.
"""

import torch
from torch import nn
from torch.nn import functional as F

from contract import CHANNELS, FRAME_COUNTS, MAX_FRAMES


MODEL_PROFILES = ("compact", "wide")


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
    """Compact baseline or wider feature-normalized U-Net with local RGB path.

    The compact profile is the 121,792-parameter ablation already tested on
    devices. Wide doubles its feature widths and normalizes feature channels.
    A small unnormalized cell-RGB readout preserves absolute color evidence
    alongside the normalized spatial context. Wide still requires retraining
    and held-out device evaluation; capacity is not a claim of better recovery.
    """

    def __init__(self, profile: str = "compact", frames: int = 1) -> None:
        super().__init__()
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Choose model profile from {MODEL_PROFILES}")
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Choose 1 to {MAX_FRAMES} input frames")
        self.profile = profile
        self.frames = frames
        width = 16 if profile == "compact" else 32
        normalize = profile == "wide"
        self.inlet = nn.Sequential(conv(3 * frames, width // 2, 2), conv(width // 2, width, 2))
        self.e0 = nn.Sequential(conv(width, width, normalize=normalize), conv(width, width, normalize=normalize))
        self.e1 = nn.Sequential(conv(width, 2 * width, 2, normalize), conv(2 * width, 2 * width, normalize=normalize))
        self.bottleneck = nn.Sequential(conv(2 * width, 4 * width, 2, normalize), conv(4 * width, 4 * width, normalize=normalize))
        self.d1 = nn.Sequential(conv(6 * width, 2 * width, normalize=normalize), conv(2 * width, 2 * width, normalize=normalize))
        self.d0 = nn.Sequential(conv(3 * width, width, normalize=normalize), conv(width, width, normalize=normalize))
        self.head = nn.Conv2d(width, 16, 1, bias=True)
        if profile == "wide":
            self.local_color = nn.Sequential(nn.Conv2d(3 * frames, 16, 1), nn.ReLU(), nn.Conv2d(16, 16, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
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
    shape = tuple(state["head.weight"].shape)
    if shape == (16, 16, 1, 1):
        return "compact"
    if shape == (16, 32, 1, 1):
        return "wide"
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
