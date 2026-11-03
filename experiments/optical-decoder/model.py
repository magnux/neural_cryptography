"""Multiframe optical classifiers with a shared browser tensor contract.

All profiles accept stacked RGB, [N,3*T,4G,4G], and predict physical palette
logits, [N,16,G,G]. None receives keys, labels, or expected payload bytes.
"""

import torch
from torch import nn
from torch.nn import functional as F

from contract import CHANNELS, FRAME_COUNTS, MAX_FRAMES, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX


MODEL_PROFILES = ("normalized-tube", "longtube")
DEFAULT_MODEL_PROFILE = "normalized-tube"


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


class ResidualPair(nn.Module):
    """Two nonlinear convolutions, with a learned shortcut only across transitions."""

    def __init__(self, in_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        self.shortcut = (nn.Identity() if in_channels == 32 and stride == 1 else
                         nn.Conv2d(in_channels, 32, 1, stride=stride, bias=True))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(F.relu(self.conv1(values)))
        return F.relu(self.shortcut(values) + residual)


class ResidualBlock(nn.Module):
    """A resolution block with one channel normalization after all residual pairs."""

    def __init__(self, in_channels: int, pairs: int, stride: int = 1) -> None:
        super().__init__()
        self.pairs = nn.Sequential(ResidualPair(in_channels, stride),
                                   *(ResidualPair(32) for _ in range(pairs - 1)))
        self.norm = ChannelLayerNorm(32)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.pairs(values)))


class OpticalDecoder(nn.Module):
    """Full-resolution 32-channel tube and deeper residual longtube.

    Both U-Nets start and end at the rectified crop resolution, 4G by 4G.
    Their bottleneck is G by G, exactly the target symbol-grid size. Only the
    outlet reduces the reconstructed full-resolution features to that grid.
    Normalized-tube preserves the established two-convolution normalized stages;
    longtube uses four-convolution residual resolution blocks, eight at its
    bottleneck, with one normalization per block. Neither has an RGB shortcut.
    """

    def __init__(self, profile: str = DEFAULT_MODEL_PROFILE, frames: int = 1) -> None:
        super().__init__()
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Choose model profile from {MODEL_PROFILES}")
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Choose 1 to {MAX_FRAMES} input frames")
        self.profile = profile
        self.frames = frames
        self.inlet = nn.Sequential(conv(3 * frames, 32, normalize=profile == "normalized-tube"),
                                   conv(32, 32, normalize=True))
        if profile == "normalized-tube":
            self.e0 = nn.Sequential(conv(32, 32, normalize=True), conv(32, 32, normalize=True))
            self.e1 = nn.Sequential(conv(32, 32, 2, True), conv(32, 32, normalize=True))
            self.bottleneck = nn.Sequential(conv(32, 32, 2, True), conv(32, 32, normalize=True))
            self.d1 = nn.Sequential(conv(64, 32, normalize=True), conv(32, 32, normalize=True))
            self.d0 = nn.Sequential(conv(64, 32, normalize=True), conv(32, 32, normalize=True))
        else:
            self.e0 = ResidualBlock(32, pairs=2)
            self.e1 = ResidualBlock(32, pairs=2, stride=2)
            self.bottleneck = ResidualBlock(32, pairs=4, stride=2)
            self.d1 = ResidualBlock(64, pairs=2)
            self.d0 = ResidualBlock(64, pairs=2)
        self.outlet = nn.Sequential(conv(32, 32, 2, normalize=True),
                                    conv(32, 32, 2, normalize=True))
        self.head = nn.Conv2d(32, 16, 1, bias=True)

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
        return self.head(self.outlet(d0))


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
                 "e0.pairs.0.conv1.weight" in state, "outlet.0.0.weight" in state)
    if signature == ((16, 32, 1, 1), 32, True, True, False, True):
        return "normalized-tube"
    if signature == ((16, 32, 1, 1), 32, False, False, True, True):
        return "longtube"
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

    Only the RGB-facing inlet convolution changes: every frame gets 1/T of
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
    name = "inlet.0.0.weight"
    state[name] = state[name].repeat(1, frames, 1, 1) / frames
    expanded.load_state_dict(state, strict=True)
    expanded.train(model.training)
    return expanded


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
