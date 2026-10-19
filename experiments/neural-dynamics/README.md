# Neural dynamics and symmetric neural-key experiment

This experiment treats recurrent convolutional networks as dynamical systems
and studies them as candidate keyed transformations. The latest original
`neural_key_experiment.py` has not yet been recovered; there is intentionally
no reconstructed script in this directory. See the [source inventory](../../docs/SOURCE_INVENTORY.md).

## Experiments previously implemented

- Gaussian versus orthogonal initialization, normalization, depth, and activation
  comparisons, including sine and ReLU.
- Fixed and rotating convolutions, plus a Feistel construction.
- Input/key perturbations, avalanche matrices, bit bias, marginal entropy,
  covariance dimension, symmetry probes, surrogate learning, and numerical agreement.
- Discrete-palette image/camera experiments; later revisions made sizes, colors,
  and multiple frames configurable.

The comparisons exposed increasing bias and reduced variation in several
recurrent configurations. Sine and Feistel variants improved the measured
diffusion behavior. These empirical probes do not establish pseudorandomness,
unforgeability, security bits, or superiority to conventional cryptography.

The floating-point dynamical experiment is separate from the exact binary
circuit used by the [Python ZKBoo core](../python-zkboo/README.md) and browser
proof engine. Shared naming does not make their architectures interchangeable.

## Restore next

Restore `neural_key_experiment.py`, its matching `neural_optical.py` utility,
and original tests from the latest user copy or project archive. Verify their
actual command-line options before documenting runnable commands here. The
original notebook belongs in [notebooks/](../../notebooks/README.md).
