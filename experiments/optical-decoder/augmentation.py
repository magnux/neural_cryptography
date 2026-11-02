"""Small photometric perturbations of training captures, after burst assembly.

Recorded calibration overshoot survives the RGB conversion. The decoder remains
responsible for the final common [-1, 1] network-input clip. No geometry, target,
stored crop, synthetic example, or evaluation input is changed here.
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
