"""Multiframe optical classifiers with a shared browser tensor contract.

Fixed-frame profiles accept stacked RGB, [N,3*T,4G,4G]. Attention-tube accepts
eight RGB slots followed by eight presence planes, [N,32,256,256]. All predict
physical palette logits [N,16,G,G]; none receives payload truth or keys.
"""

import torch
from torch import nn
from torch.nn import functional as F

from contract import CHANNELS, FRAME_COUNTS, MAX_FRAMES, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX


MODEL_PROFILES = ("normalized-tube", "longtube", "attention-tube")
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


class WindowAttentionBlock(nn.Module):
    """C32, four-head attention over 8x8 windows on the fixed 64x64 cell grid.

    The second block shifts windows by four cells. Its additive mask prevents
    cyclic roll from connecting opposite image edges. Channel norms are per
    position; linear projections and attention export as standard ONNX ops.
    """

    def __init__(self, shift: int = 0) -> None:
        super().__init__()
        if shift not in (0, 4):
            raise ValueError("Attention window shift must be zero or four cells")
        self.shift = shift
        self.norm1 = ChannelLayerNorm(32)
        self.qkv = nn.Linear(32, 96)
        self.projection = nn.Linear(32, 32)
        self.norm2 = ChannelLayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        mask = torch.empty(0)
        if shift:
            regions = torch.zeros(1, 1, 64, 64)
            label = 0
            for ys in (slice(0, -8), slice(-8, -4), slice(-4, None)):
                for xs in (slice(0, -8), slice(-8, -4), slice(-4, None)):
                    regions[:, :, ys, xs] = label
                    label += 1
            windows = self.partition(regions).squeeze(-1)
            difference = windows[:, :, None] - windows[:, None, :]
            mask = torch.where(difference == 0, 0., -10000.).unsqueeze(1)
        self.register_buffer("window_mask", mask, persistent=False)

    @staticmethod
    def partition(values: torch.Tensor) -> torch.Tensor:
        batch, channels = values.shape[:2]
        return values.reshape(batch, channels, 8, 8, 8, 8).permute(
            0, 2, 4, 3, 5, 1).reshape(-1, 64, channels)

    @staticmethod
    def reverse(windows: torch.Tensor, batch: int) -> torch.Tensor:
        return windows.reshape(batch, 8, 8, 8, 8, 32).permute(
            0, 5, 1, 3, 2, 4).reshape(batch, 32, 64, 64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        normalized = self.norm1(values)
        if self.shift:
            normalized = torch.roll(normalized, shifts=(-self.shift, -self.shift), dims=(2, 3))
        windows = self.partition(normalized)
        qkv = self.qkv(windows).reshape(-1, 64, 3, 4, 8).permute(2, 0, 3, 1, 4)
        scores = (qkv[0] @ qkv[1].transpose(-2, -1)) * (8 ** -.5)
        if self.shift:
            scores = (scores.reshape(batch, 64, 4, 64, 64) + self.window_mask.unsqueeze(0)).reshape(-1, 4, 64, 64)
        attention = torch.softmax(scores, dim=-1)
        attended = (attention @ qkv[2]).transpose(1, 2).reshape(-1, 64, 32)
        attended = self.reverse(self.projection(attended), batch)
        if self.shift:
            attended = torch.roll(attended, shifts=(self.shift, self.shift), dims=(2, 3))
        values = values + attended
        feed_forward = self.ffn(self.norm2(values).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return values + feed_forward


class OpticalDecoder(nn.Module):
    """Full-resolution 32-channel tube, residual longtube and anchored attention.

    All U-Nets start and end at the rectified crop resolution, 4G by 4G.
    Their bottleneck is G by G, exactly the target symbol-grid size. Only the
    outlet reduces the reconstructed full-resolution features to that grid.
    Normalized-tube preserves the established two-convolution normalized stages;
    longtube uses four-convolution residual resolution blocks, eight at its
    bottleneck, with one normalization per block. Attention-tube keeps longtube's
    convolutional path, adds a learned global bias after the inlet and two
    attention blocks at the target-sized bottleneck. No model has an RGB shortcut.
    """

    def __init__(self, profile: str = DEFAULT_MODEL_PROFILE, frames: int | None = None) -> None:
        super().__init__()
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Choose model profile from {MODEL_PROFILES}")
        if frames is None:
            frames = 8 if profile == "attention-tube" else 1
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Choose 1 to {MAX_FRAMES} input frames")
        if profile == "attention-tube" and frames != 8:
            raise ValueError("Attention-tube uses eight phase-paired slots, with four to eight present frames")
        self.profile = profile
        self.frames = frames
        self.variable_frames = profile == "attention-tube"
        self.min_frames = 4 if self.variable_frames else frames
        self.input_channels = 4 * frames if self.variable_frames else 3 * frames
        self.input_layout = "phase-paired-rgb-presence-planes" if self.variable_frames else "stacked-rgb"
        self.output_label_space = "logical" if self.variable_frames else "physical"
        self.phase_masks = (0, 15, 7, 8) if self.variable_frames else ()
        self.supported_sizes = (64,) if self.variable_frames else (64, 128)
        self.inlet = nn.Sequential(conv(self.input_channels, 32, normalize=profile == "normalized-tube"),
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
        if self.variable_frames:
            # Learned per-channel global anchor, not a positional sinusoid or
            # observed image. The compact lattice expands only the bias: all
            # incoming image evidence remains at full 256x256 resolution.
            self.position_anchor = nn.Parameter(torch.empty(1, 32, 64, 64))
            nn.init.normal_(self.position_anchor, mean=0., std=.02)
            self.attention = nn.Sequential(WindowAttentionBlock(), WindowAttentionBlock(shift=4))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # Preserve calibrated overshoot in recordings/preprocessing; every
        # learned layer consumes the same bounded range for real and synthetic
        # inputs. ONNX exports this operation, including browser camera inference.
        if self.variable_frames:
            if not torch.jit.is_tracing() and (pixels.ndim != 4 or pixels.shape[1:] != (32, 256, 256)):
                raise ValueError("Attention-tube requires packed eight-slot G64 input [N,32,256,256]")
            # The tensor contract supplies spatially constant binary planes.
            # Read their flags once and reconstruct planes, so gating and
            # conditioning always use exactly the same eight availability bits.
            presence = pixels[:, 24:, :1, :1].clamp(0., 1.)
            rgb = torch.clamp(pixels[:, :24], NETWORK_INPUT_MIN, NETWORK_INPUT_MAX)
            rgb = (rgb.reshape(-1, 8, 3, 256, 256) * presence.unsqueeze(2)).reshape(-1, 24, 256, 256)
            pixels = torch.cat((rgb, presence.expand(-1, -1, 256, 256)), dim=1)
        else:
            pixels = torch.clamp(pixels, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX)
        features = self.inlet(pixels)
        if self.variable_frames:
            features = features + F.interpolate(self.position_anchor, size=(256, 256),
                                                 mode="bilinear", align_corners=False)
        e0 = self.e0(features)
        e1 = self.e1(e0)
        context = self.bottleneck(e1)
        if self.variable_frames:
            context = self.attention(context)
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
                 "e0.pairs.0.conv1.weight" in state, "outlet.0.0.weight" in state,
                 "position_anchor" in state, "attention.0.qkv.weight" in state)
    if signature == ((16, 32, 1, 1), 32, True, True, False, True, False, False):
        return "normalized-tube"
    if signature == ((16, 32, 1, 1), 32, False, False, True, True, False, False):
        return "longtube"
    if signature == ((16, 32, 1, 1), 32, False, False, True, True, True, True) and inlet.shape[1] == 32:
        return "attention-tube"
    raise ValueError("Checkpoint model does not match a supported decoder profile")


def frames_from_state(state: dict) -> int:
    """Read the input width; remaining tensor shapes are checked by the loader."""
    weight = state.get("inlet.0.0.weight") if isinstance(state, dict) else None
    if profile_from_state(state) == "attention-tube":
        return 8
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
