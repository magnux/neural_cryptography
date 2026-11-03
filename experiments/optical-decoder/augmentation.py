"""Small photometric perturbations of training captures, after burst assembly.

Recorded calibration overshoot survives the RGB conversion. The decoder remains
responsible for the final common [-1, 1] network-input clip. No geometry, target,
stored crop, synthetic example, or evaluation input is changed by RealAugmentation.
PhaseAugmentation separately applies phase/geometry symmetries to assembled
phase-grouped training inputs of either source.
"""

import numpy as np

from contract import CAPTURE_INPUT_MIN, CAPTURE_INPUT_MAX


class RealAugmentation:
    def __init__(self, *, seed, noise_std=.008, tint_stops=.06, exposure_stops=.08):
        for name, value, maximum in (("noise_std", noise_std, 1.),
                                     ("tint_stops", tint_stops, 4.),
                                     ("exposure_stops", exposure_stops, 4.)):
            if (isinstance(value, bool) or not np.isfinite(value)
                    or not 0 <= value <= maximum):
                raise ValueError(f"Real augmentation {name} must be finite and within 0..{maximum:g}")
        self.noise_std = float(noise_std)
        self.tint_stops = float(tint_stops)
        self.exposure_stops = float(exposure_stops)
        self.enabled = any((self.noise_std, self.tint_stops, self.exposure_stops))
        # Separate from both the anchor/burst draws and torch model randomness.
        self.seed = [seed, 0x4155474D]
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))

    def settings(self):
        return {"enabled": self.enabled, "scope": "real-training-only",
                "noiseStdRGB01": self.noise_std, "tintStops": self.tint_stops,
                "exposureStops": self.exposure_stops, "rngSeed": self.seed,
                "gainSharing": "one-exposure-and-rgb-tint-per-stack",
                "noiseSharing": "independent-per-pixel-and-exposure",
                "outputClip": [CAPTURE_INPUT_MIN, CAPTURE_INPUT_MAX]}

    def __call__(self, pixels, sources):
        """Return NCHW float32 inputs with only real rows changed, without mutation."""
        if (pixels.ndim != 4 or pixels.shape[1] % 3 or pixels.dtype != np.float32
                or len(sources) != len(pixels)):
            raise ValueError("Expected float32 NCHW RGB stacks with one source per row")
        real = [index for index, source in enumerate(sources) if source == "real"]
        if not self.enabled or not real:
            return pixels
        result = pixels.copy()
        for index in real:
            # Shared mild color/exposure shifts retain a coherent burst. Noise
            # differs between exposures, as it does between camera captures.
            exposure = self.rng.uniform(-self.exposure_stops, self.exposure_stops)
            tint = self.rng.uniform(-self.tint_stops, self.tint_stops, 3)
            gain = np.exp2(exposure+tint).astype(np.float32).reshape(1, 3, 1, 1)
            rgb = (pixels[index].reshape(-1, 3, *pixels.shape[2:]) + 1.)*.5
            rgb *= gain
            if self.noise_std:
                rgb += self.rng.normal(0., self.noise_std, rgb.shape).astype(np.float32)
            result[index] = np.clip(rgb*2.-1., CAPTURE_INPUT_MIN, CAPTURE_INPUT_MAX).reshape(pixels.shape[1:])
        return result


class PhaseAugmentation:
    """Exact phase-group symmetries and joint training-only spatial flips.

    Pixel values are never recolored. Reordering the four phase pairs corresponds
    to XORing data labels by one of the closed palette-mask group elements.
    Pilots are unchanged by phase relabeling. Spatial flips transform all inputs,
    targets and masks together; they are never applied during wire recovery.
    """

    def __init__(self, *, seed, flip_probability=.25, permutations=True):
        if (isinstance(flip_probability, bool) or not np.isfinite(flip_probability)
                or not 0 <= flip_probability <= 1 or type(permutations) is not bool):
            raise ValueError("Phase flip probability must be within 0..1 and permutations must be boolean")
        self.flip_probability = float(flip_probability)
        self.permutations = permutations
        self.seed = [seed, 0x50484153]
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))

    def settings(self):
        return {"scope": "phase-grouped-training-only", "phasePermutations": self.permutations,
                "horizontalFlipProbability": self.flip_probability,
                "verticalFlipProbability": self.flip_probability,
                "rngSeed": self.seed, "recovery": "unaugmented-canonical-coordinates"}

    @staticmethod
    def transform(pixels, labels, mask, *, delta=0, horizontal=False, vertical=False):
        from phases import PHASE_MASKS
        if (pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[0] != 32
                or labels.ndim != 2 or mask.shape != labels.shape or mask.dtype != np.bool_
                or pixels.shape[-2:] != tuple(4*size for size in labels.shape)
                or delta not in PHASE_MASKS):
            raise ValueError("Expected packed phase-paired RGB/presence, labels and boolean mask")
        source_phases = [PHASE_MASKS.index(value ^ delta) for value in PHASE_MASKS]
        source_slots = [2*phase+slot for phase in source_phases for slot in range(2)]
        rgb = pixels[:24].reshape(8, 3, *pixels.shape[-2:])[source_slots].reshape(24, *pixels.shape[-2:])
        result = np.concatenate((rgb, pixels[24:][source_slots]), axis=0)
        target = labels.copy()
        target[mask] ^= delta
        target_mask = mask.copy()
        for axis, enabled in ((-1, horizontal), (-2, vertical)):
            if enabled:
                result = np.flip(result, axis=axis)
                target = np.flip(target, axis=axis)
                target_mask = np.flip(target_mask, axis=axis)
        # torch.from_numpy cannot consume negative-stride flip views.
        return (np.ascontiguousarray(result), np.ascontiguousarray(target),
                np.ascontiguousarray(target_mask))

    def __call__(self, pixels, labels, masks):
        from phases import PHASE_MASKS
        transformed = [self.transform(row, target, mask,
                       delta=int(self.rng.choice(PHASE_MASKS)) if self.permutations else 0,
                       horizontal=self.rng.random() < self.flip_probability,
                       vertical=self.rng.random() < self.flip_probability)
                       for row, target, mask in zip(pixels, labels, masks)]
        return tuple(np.stack([row[index] for row in transformed]) for index in range(3))
