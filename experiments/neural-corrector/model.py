"""Neural matrix correctors with a shared browser tensor contract.

Fixed-frame profiles accept stacked RGB, [N,3*T,4G,4G]. U-tube-2x-attn accepts
eight RGB slots followed by eight presence planes, [N,32,256,256]. Multires
uses the same channel packing at [N,32,2G,2G] with independent G8..G256 anchors.
W-tube profiles use [N,32,4G,4G], G16..G128 anchors, and two or three
full-resolution U passes with independent weights.
All produce
palette logits [N,16,G,G]: physical classes for fixed-frame models, canonical
logical data classes for the phase-grouped model. The public output adapter
maps both to corrected logical matrix scores. No model receives payload truth
or keys.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from contract import CHANNELS, FRAME_COUNTS, MAX_FRAMES, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX


W_TUBE_PROFILES = ("W-tube", "VW")
W_TUBE_STAGE_COUNTS = {"W-tube": 2, "VW": 3}
MODEL_PROFILES = ("U-tube", "U-tube-2x", "U-tube-2x-attn", "multires", *W_TUBE_PROFILES)
DEFAULT_MODEL_PROFILE = "U-tube"
MULTIRES_SIZES = (8, 16, 32, 64, 128, 256)
W_TUBE_SIZES = (16, 32, 64, 128)
ANCHOR_MAX_MODES = 16
ANCHOR_METADATA = {"representation": "dct-low-frequency", "maxModes": ANCHOR_MAX_MODES,
                   "normalization": "ortho", "spatialScale": "grid-over-modes"}
BOTTLENECK_METADATA = {"blocks": ["conv", "attention", "conv", "attention", "conv", "attention"],
                       "layersPerBlock": 2}
W_TUBE_BOTTLENECK_METADATA = {"stages": 2, "blocks": ["conv", "attention"],
    "layersPerBlock": 2, "scales": [4, 2, 1, 2, 4], "fusion": "normalized-residual"}


def wtube_bottleneck_metadata(profile: str) -> dict:
    """Describe one exact complete-U stage count, without mutable shared lists."""
    if profile not in W_TUBE_PROFILES:
        raise ValueError("Expected a W-tube model profile")
    return {**W_TUBE_BOTTLENECK_METADATA, "stages": W_TUBE_STAGE_COUNTS[profile],
            "blocks": list(W_TUBE_BOTTLENECK_METADATA["blocks"]),
            "scales": list(W_TUBE_BOTTLENECK_METADATA["scales"])}


def dct_anchor_basis(size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II synthesis columns for the retained low frequencies.

    Construct derived geometry in float64 before the final cast, so G256's
    cosine basis has the same accuracy as smaller grids. These columns are
    deterministic buffers, not learned spatial values or checkpoint payload.
    """
    if type(size) is not int or size not in MULTIRES_SIZES:
        raise ValueError("Anchor grid must be one of G8, G16, G32, G64, G128 or G256")
    modes = min(size, ANCHOR_MAX_MODES)
    positions = torch.arange(size, device=device, dtype=torch.float64)[:, None] + .5
    frequencies = torch.arange(modes, device=device, dtype=torch.float64)[None]
    basis = torch.cos(math.pi * positions * frequencies / size) * math.sqrt(2./size)
    basis[:, 0] /= math.sqrt(2.)
    return basis.to(dtype=dtype)


def project_anchor(spatial: torch.Tensor) -> torch.Tensor:
    """Project a dense learned anchor onto this model's scaled cosine basis.

    This is an explicit initialization primitive, not checkpoint compatibility.
    A=(G/m) Q theta Q.T, so theta=(m/G) Q.T A Q. Its reconstruction is the
    least-squares low-frequency projection. The scale keeps mean squared anchor
    amplitude equal to mean squared coefficient amplitude at every grid size.
    """
    if (not isinstance(spatial, torch.Tensor) or spatial.ndim != 4
            or tuple(spatial.shape[:2]) != (1, 32)
            or spatial.shape[-2] != spatial.shape[-1]
            or not spatial.is_floating_point() or not bool(torch.isfinite(spatial).all())):
        raise ValueError("Anchor projection requires a finite floating [1,32,G,G] tensor")
    size = int(spatial.shape[-1])
    basis = dct_anchor_basis(size, device=spatial.device, dtype=spatial.dtype)
    modes = basis.shape[1]
    return (basis.transpose(0, 1) @ spatial @ basis) * (modes/size)


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
    """C32, four-head attention over 8x8 windows on a supported cell grid.

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
        # These are derived geometry, not weights. Keeping them out of the
        # checkpoint preserves the exact established G64 attention state.
        self.register_buffer("window_mask", self.boundary_mask(64) if shift else torch.empty(0),
                             persistent=False)

    @staticmethod
    def partition(values: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = values.shape
        return values.reshape(batch, channels, height//8, 8, width//8, 8).permute(
            0, 2, 4, 3, 5, 1).reshape(-1, 64, channels)

    @staticmethod
    def reverse(windows: torch.Tensor, batch: int, size: int = 64) -> torch.Tensor:
        return windows.reshape(batch, size//8, size//8, 8, 8, 32).permute(
            0, 5, 1, 3, 2, 4).reshape(batch, 32, size, size)

    @classmethod
    def boundary_mask(cls, size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
        # Region IDs reproduce the shifted-window boundary mask without image
        # slice mutation. At G8 the forward path disables shifts altogether.
        coordinates = torch.arange(size, device=device)
        regions = (coordinates >= size-8).to(torch.int64) + (coordinates >= size-4).to(torch.int64)
        labels = (regions[:, None]*3 + regions[None, :]).reshape(1, 1, size, size)
        windows = cls.partition(labels).squeeze(-1)
        different = windows[:, :, None] != windows[:, None, :]
        return (different.to(dtype)*-10000.).unsqueeze(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        # Browser artifacts are one static G each. Keep geometry in Python so
        # tracing folds window counts/masks rather than adding shape arithmetic
        # to the deployed graph. Eager training still selects G per minibatch.
        size, width = int(values.shape[2]), int(values.shape[3])
        if not torch.jit.is_tracing() and (size != width or size not in MULTIRES_SIZES):
            raise ValueError("Attention requires a supported square grid, G8 through G256")
        shift = self.shift if size > 8 else 0
        normalized = self.norm1(values)
        if shift:
            normalized = torch.roll(normalized, shifts=(-shift, -shift), dims=(2, 3))
        windows = self.partition(normalized)
        qkv = self.qkv(windows).reshape(-1, 64, 3, 4, 8).permute(2, 0, 3, 1, 4)
        scores = (qkv[0] @ qkv[1].transpose(-2, -1)) * (8 ** -.5)
        if shift:
            mask = (self.window_mask if size == 64 else
                    self.boundary_mask(size, device=values.device, dtype=scores.dtype))
            scores = (scores.reshape(batch, (size//8)**2, 4, 64, 64) + mask.unsqueeze(0)).reshape(-1, 4, 64, 64)
        attention = torch.softmax(scores, dim=-1)
        attended = (attention @ qkv[2]).transpose(1, 2).reshape(-1, 64, 32)
        attended = self.reverse(self.projection(attended), batch, size)
        if shift:
            attended = torch.roll(attended, shifts=(shift, shift), dims=(2, 3))
        values = values + attended
        feed_forward = self.ffn(self.norm2(values).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return values + feed_forward


class UTubeStage(nn.Module):
    """One 4G -> 2G -> G -> 2G -> 4G pass shared by W-tube and VW.

    Each skip contains normalized learned features. The bottleneck has exactly
    two convolutions and two attention layers; no operation goes below G.
    Normalized residual fusion gives the next pass both the incoming features
    and this pass's refinement, without exposing a raw RGB shortcut.
    """

    def __init__(self) -> None:
        super().__init__()
        self.e0 = ResidualBlock(32, pairs=2)
        self.e1 = ResidualBlock(32, pairs=2, stride=2)
        self.bottleneck = ResidualBlock(32, pairs=1, stride=2)
        self.attention = nn.Sequential(WindowAttentionBlock(), WindowAttentionBlock(shift=4))
        self.d1 = ResidualBlock(64, pairs=2)
        self.d0 = ResidualBlock(64, pairs=2)
        self.fusion = ChannelLayerNorm(32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        e0 = self.e0(features)
        e1 = self.e1(e0)
        context = self.attention(self.bottleneck(e1))
        d1 = self.d1(torch.cat((F.interpolate(context, scale_factor=2, mode="nearest"), e1), dim=1))
        d0 = self.d0(torch.cat((F.interpolate(d1, scale_factor=2, mode="nearest"), e0), dim=1))
        return F.relu(self.fusion(features + d0))


class NeuralCorrector(nn.Module):
    """Full-resolution 32-channel tube, residual U-tube-2x and anchored attention.

    The established U-tube profiles start and end at the rectified 4G crop.
    Their bottleneck is G by G, exactly the target symbol-grid size. Only the
    outlet reduces the reconstructed full-resolution features to that grid.
    U-tube preserves the established two-convolution normalized stages;
    U-tube-2x uses four-convolution residual resolution blocks, eight at its
    bottleneck, with one normalization per block. U-tube-2x-attn keeps U-tube-2x's
    convolutional path, adds a learned global bias after the inlet and two
    attention blocks at the target-sized bottleneck. No model has an RGB shortcut.
    Multires shares a one-level 2G->G->2G trunk with a separate anchor per G,
    six alternating blocks: three two-convolution segments and three pairs of
    window-attention layers. Each independent anchor is limited to 16x16 cosine
    modes per channel; no learned dense position map exists.
    W-tube restores two contractions to the target grid and repeats the complete
    U twice (W-tube) or three times (VW), with independent weights.
    Each U has one conv/attention pair at G; a third U adds a complete spatial
    contraction/reconstruction path rather than additional bottleneck cycles.
    """

    def __init__(self, profile: str = DEFAULT_MODEL_PROFILE, frames: int | None = None) -> None:
        super().__init__()
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Choose model profile from {MODEL_PROFILES}")
        if frames is None:
            frames = 8 if profile in ("U-tube-2x-attn", "multires", *W_TUBE_PROFILES) else 1
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Choose 1 to {MAX_FRAMES} input frames")
        if profile in ("U-tube-2x-attn", "multires", *W_TUBE_PROFILES) and frames != 8:
            raise ValueError(f"{profile} uses eight phase-paired slots, with four to eight present frames")
        self.profile = profile
        self.frames = frames
        self.variable_frames = profile in ("U-tube-2x-attn", "multires", *W_TUBE_PROFILES)
        self.min_frames = 4 if self.variable_frames else frames
        self.input_channels = 4 * frames if self.variable_frames else 3 * frames
        self.input_layout = "phase-paired-rgb-presence-planes" if self.variable_frames else "stacked-rgb"
        self.output_label_space = "logical" if self.variable_frames else "physical"
        self.phase_masks = (0, 15, 7, 8) if self.variable_frames else ()
        self.supported_sizes = (W_TUBE_SIZES if profile in W_TUBE_PROFILES else
                                MULTIRES_SIZES if profile == "multires" else
                                ((64,) if self.variable_frames else (64, 128)))
        self.pixels_per_cell = 2 if profile == "multires" else 4
        self.inlet = nn.Sequential(conv(self.input_channels, 32, normalize=profile == "U-tube"),
                                   conv(32, 32, normalize=True))
        if profile == "U-tube":
            self.e0 = nn.Sequential(conv(32, 32, normalize=True), conv(32, 32, normalize=True))
            self.e1 = nn.Sequential(conv(32, 32, 2, True), conv(32, 32, normalize=True))
            self.bottleneck = nn.Sequential(conv(32, 32, 2, True), conv(32, 32, normalize=True))
            self.d1 = nn.Sequential(conv(64, 32, normalize=True), conv(32, 32, normalize=True))
            self.d0 = nn.Sequential(conv(64, 32, normalize=True), conv(32, 32, normalize=True))
        elif profile == "multires":
            self.e0 = ResidualBlock(32, pairs=2)
            self.bottleneck = ResidualBlock(32, pairs=1, stride=2)
            self.attention = nn.Sequential(WindowAttentionBlock(), WindowAttentionBlock(shift=4))
            self.bottleneck_second = ResidualBlock(32, pairs=1)
            self.attention_second = nn.Sequential(WindowAttentionBlock(), WindowAttentionBlock(shift=4))
            self.bottleneck_third = ResidualBlock(32, pairs=1)
            self.attention_third = nn.Sequential(WindowAttentionBlock(), WindowAttentionBlock(shift=4))
            self.d0 = ResidualBlock(64, pairs=2)
            self.anchor_metadata = dict(ANCHOR_METADATA)
            self.bottleneck_metadata = {**BOTTLENECK_METADATA, "blocks": list(BOTTLENECK_METADATA["blocks"])}
            self.anchor_coefficients = nn.ParameterDict({
                str(size): nn.Parameter(torch.empty(1, 32, min(size, ANCHOR_MAX_MODES),
                                                      min(size, ANCHOR_MAX_MODES)))
                for size in self.supported_sizes})
            for size in self.supported_sizes:
                self.register_buffer(f"anchor_basis_{size}", dct_anchor_basis(size), persistent=False)
                nn.init.normal_(self.anchor_coefficients[str(size)], mean=0., std=.02)
        elif profile in W_TUBE_PROFILES:
            self.stages = nn.ModuleList(UTubeStage() for _ in range(W_TUBE_STAGE_COUNTS[profile]))
            self.anchor_metadata = dict(ANCHOR_METADATA)
            self.bottleneck_metadata = wtube_bottleneck_metadata(profile)
            self.anchor_coefficients = nn.ParameterDict({
                str(size): nn.Parameter(torch.empty(1, 32, min(size, ANCHOR_MAX_MODES),
                                                      min(size, ANCHOR_MAX_MODES)))
                for size in self.supported_sizes})
            for size in self.supported_sizes:
                self.register_buffer(f"anchor_basis_{size}", dct_anchor_basis(size), persistent=False)
                nn.init.normal_(self.anchor_coefficients[str(size)], mean=0., std=.02)
        else:
            self.e0 = ResidualBlock(32, pairs=2)
            self.e1 = ResidualBlock(32, pairs=2, stride=2)
            self.bottleneck = ResidualBlock(32, pairs=4, stride=2)
            self.d1 = ResidualBlock(64, pairs=2)
            self.d0 = ResidualBlock(64, pairs=2)
        self.outlet = nn.Sequential(conv(32, 32, 2, normalize=True),
                                    conv(32, 32, 1 if profile == "multires" else 2, normalize=True))
        self.head = nn.Conv2d(32, 16, 1, bias=True)
        if profile == "U-tube-2x-attn":
            # Learned per-channel global anchor, not a positional sinusoid or
            # observed image. The compact lattice expands only the bias: all
            # incoming image evidence remains at full 256x256 resolution.
            self.position_anchor = nn.Parameter(torch.empty(1, 32, 64, 64))
            nn.init.normal_(self.position_anchor, mean=0., std=.02)
            self.attention = nn.Sequential(WindowAttentionBlock(), WindowAttentionBlock(shift=4))

    def spatial_anchor(self, size: int) -> torch.Tensor:
        """Reconstruct only the selected independent anchor at target resolution.

        The G/m scale makes spatial RMS equal coefficient RMS (up to float32
        rounding), keeping initialization and coefficient step units consistent
        across resolutions. Interpolation to the profile's camera-raster size
        follows this reconstruction; only the positional prior is bandwidth-limited.
        """
        if self.profile not in ("multires", *W_TUBE_PROFILES) or type(size) is not int or size not in self.supported_sizes:
            raise ValueError("Spatial anchor requires a supported anchored-model grid")
        coefficients = self.anchor_coefficients[str(size)]
        basis = getattr(self, f"anchor_basis_{size}")
        return (basis @ coefficients @ basis.transpose(0, 1)) * (size/coefficients.shape[-1])

    def forward(self, pixels: torch.Tensor, grid_size: int | None = None) -> torch.Tensor:
        # Preserve calibrated overshoot in recordings/preprocessing; every
        # learned layer consumes the same bounded range for real and synthetic
        # inputs. ONNX exports this operation, including browser camera inference.
        if self.variable_frames:
            if not torch.jit.is_tracing():
                if (pixels.ndim != 4 or pixels.shape[1] != 32 or pixels.shape[2] != pixels.shape[3]
                        or pixels.shape[2] % self.pixels_per_cell
                        or pixels.shape[2] // self.pixels_per_cell not in self.supported_sizes):
                    raise ValueError(f"{self.profile} requires packed eight-slot input at "
                                     f"{self.pixels_per_cell} pixels/cell, grids "
                                     f"{', '.join('G'+str(size) for size in self.supported_sizes)}")
            side = pixels.shape[-1]
            # Export one static geometry at a time; tracing selects only its
            # matching anchor, never the full bank or an interpolated other G.
            size = int(side) // self.pixels_per_cell
            if grid_size is not None and grid_size != size:
                raise ValueError("Explicit grid size disagrees with the packed input")
            # The tensor contract supplies spatially constant binary planes.
            # Read their flags once and reconstruct planes, so gating and
            # conditioning always use exactly the same eight availability bits.
            presence = pixels[:, 24:, :1, :1].clamp(0., 1.)
            rgb = torch.clamp(pixels[:, :24], NETWORK_INPUT_MIN, NETWORK_INPUT_MAX)
            rgb = (rgb.reshape(-1, 8, 3, side, side) * presence.unsqueeze(2)).reshape(-1, 24, side, side)
            pixels = torch.cat((rgb, presence.expand(-1, -1, side, side)), dim=1)
        else:
            pixels = torch.clamp(pixels, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX)
        features = self.inlet(pixels)
        if self.variable_frames:
            anchor = self.spatial_anchor(size) if self.profile in ("multires", *W_TUBE_PROFILES) else self.position_anchor
            features = features + F.interpolate(anchor, scale_factor=self.pixels_per_cell,
                                                 mode="bilinear", align_corners=False)
        if self.profile in W_TUBE_PROFILES:
            for stage in self.stages:
                features = stage(features)
            return self.head(self.outlet(features))
        e0 = self.e0(features)
        if self.profile == "multires":
            context = self.attention(self.bottleneck(e0))
            context = self.attention_second(self.bottleneck_second(context))
            context = self.attention_third(self.bottleneck_third(context))
            d0 = self.d0(torch.cat((F.interpolate(context, scale_factor=2, mode="nearest"), e0), dim=1))
            return self.head(self.outlet(d0))
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
        raise ValueError("Checkpoint has no valid corrector head")
    inlet = state.get("inlet.0.0.weight")
    if not isinstance(inlet, torch.Tensor) or inlet.ndim != 4:
        raise ValueError("Checkpoint has no valid corrector inlet")
    stage_keys = [name for name in state if name.startswith("stages.")]
    if stage_keys:
        indexes = {name.split(".")[1] for name in stage_keys}
        profiles = {frozenset(str(i) for i in range(count)): profile
                    for profile, count in W_TUBE_STAGE_COUNTS.items()}
        profile = profiles.get(frozenset(indexes))
        if (profile is None or "anchor_coefficients.16" not in state or inlet.shape[1] != 32
                or any(f"stages.{i}.attention.{j}.qkv.weight" not in state
                       for i in range(W_TUBE_STAGE_COUNTS.get(profile, 0)) for j in (0, 1))):
            raise ValueError("W-tube weights require exactly two or three contiguous complete U stages")
        return profile
    if ("anchor_coefficients.64" in state and "attention_third.0.qkv.weight" in state
            and "bottleneck_third.pairs.0.conv1.weight" in state and inlet.shape[1] == 32):
        return "multires"
    if any(name.startswith(("anchors.", "anchor_coefficients.")) for name in state):
        raise ValueError("Unsupported multires weights: the current model requires low-frequency anchors "
                         "and six bottleneck blocks; start a new run for this architecture")
    signature = (tuple(state["head.weight"].shape), inlet.shape[0],
                 "inlet.0.1.weight" in state, "e0.0.1.weight" in state,
                 "e0.pairs.0.conv1.weight" in state, "outlet.0.0.weight" in state,
                 "position_anchor" in state, "attention.0.qkv.weight" in state)
    if signature == ((16, 32, 1, 1), 32, True, True, False, True, False, False):
        return "U-tube"
    if signature == ((16, 32, 1, 1), 32, False, False, True, True, False, False):
        return "U-tube-2x"
    if signature == ((16, 32, 1, 1), 32, False, False, True, True, True, True) and inlet.shape[1] == 32:
        return "U-tube-2x-attn"
    raise ValueError("Checkpoint model does not match a supported corrector profile")


def frames_from_state(state: dict) -> int:
    """Read the input width; remaining tensor shapes are checked by the loader."""
    weight = state.get("inlet.0.0.weight") if isinstance(state, dict) else None
    if profile_from_state(state) in ("U-tube-2x-attn", "multires", *W_TUBE_PROFILES):
        return 8
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 4
            or weight.shape[1] % CHANNELS != 0 or weight.shape[1] // CHANNELS not in FRAME_COUNTS):
        raise ValueError("Checkpoint does not have a supported input frame count")
    return int(weight.shape[1] // CHANNELS)


def initialize_frames(model: NeuralCorrector, frames: int) -> NeuralCorrector:
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
    expanded = NeuralCorrector(model.profile, frames=frames).to(
        device=parameter.device, dtype=parameter.dtype)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    name = "inlet.0.0.weight"
    state[name] = state[name].repeat(1, frames, 1, 1) / frames
    expanded.load_state_dict(state, strict=True)
    expanded.train(model.training)
    return expanded


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def processing_parameter_count(model: NeuralCorrector) -> int:
    """Conv/attention budget excludes learned positional anchor storage."""
    return sum(parameter.numel() for name, parameter in model.named_parameters()
               if name != "position_anchor" and not name.startswith("anchor_coefficients."))
