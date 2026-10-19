# Neural dynamics and symmetric neural-key experiment

This executable experiment treats recurrent convolutional networks as dynamical
systems and studies them as candidate keyed transformations. It compares wave
and invertible Feistel architectures, fixed and rotating kernels, Gaussian and
orthogonal initialization, normalization, and nonlinearities. The conceptual
notebook is published in [Colab](https://colab.research.google.com/drive/19kV6VOT5zi6daMn6RaNWz_lDCxeAZgci#scrollTo=irKCCLQhd0_t).

The script was recovered from the supplied Python project archive. Its numerical
experiment is retained; its optical utility import now points to the one shared
module in [shared/python](../../shared/python/README.md). Git history records
changes; this directory contains one implementation.

## Run

From the repository root, install NumPy and PyTorch in a virtual environment:

```sh
python -m pip install -r experiments/neural-dynamics/requirements.txt
python experiments/neural-dynamics/neural_key_experiment.py --compare
python experiments/neural-dynamics/neural_key_experiment.py --self-test
python experiments/neural-dynamics/neural_key_experiment.py \
  --size 16 --colors 16 --frames 4 --architecture feistel \
  --init orthogonal --rotation bilinear --camera
```

No Hugging Face, image library, or web framework is needed. The experiment uses
CUDA when available, with `--device cpu` for CPU execution. Its random seeds are
reproducible experimental settings, not secure credentials.

## Input and computation

| Option | Meaning |
| --- | --- |
| `--size` / `--matrix-size` | Actual input and output side, 8–512 cells; default 8 |
| `--colors` | Symbol alphabet: 2, 4, 8, or 16; default 4 |
| `--frames` | Exact matrices per challenge and response, 1–8; default 1 |
| `--depths` | Recurrent depths; default `1,4,8,16,32` |
| `--architecture` | `wave` or `feistel` |
| `--activation` | `linear`, `relu`, `tanh`, `silu`, or `sine` |
| `--rotation` | `none` or bilinear rotation of the convolution kernels |
| `--norm` | Layer, instance, mixed, or no normalization |

A challenge has shape `[frames, size, size]`. Symbol bits become input planes;
frames are stacked vertically for convolution. Responses have the same sequence
shape. The Feistel shuffle spans the entire sequence and preserves its bits.
Feistel requires an even total bit count: with 2/8 colors, an odd side needs an
even number of frames. The wave architecture uses a public embedding into the
hidden channels and thresholds the output planes.

The larger size and frame settings change the actual neural computation. They
are independent of the optical dimensions of the [ZKBoo CLI](../python-zkboo/README.md),
where the proof circuit has its own small `--state-size`.

## Diagnostics and outputs

The script measures input/key perturbations, sampled avalanche matrices, bit
bias, marginal entropy, covariance participation ratio, symmetry probes,
surrogate prediction, and float/batch agreement. It also tests exact inversion
of the Feistel construction. `--compare` runs the architecture comparisons and
writes `comparison.csv` and `comparison.json`.

The default output is `neural_key_results/`:

- `summary.json`: configuration, sequence dimensions, diagnostics, timings, and capture checks.
- `examples.svg`: challenge and responses at selected depths.
- `challenge_display.html`, `response_display.html`, PNGs, and `*_frames.npz`: exact symbol sequences and animated viewers.
- `trajectory.npz`: bounded state history, when enabled.
- `avalanche.npz`: sampled input/output bit-flip estimates, when enabled.

The raw neural grids have no transport header. Capture must supply complete,
ordered frames. `--camera` uses aligned cell means, palette gain/offset
calibration, and Gaussian noise; it does not perform real camera acquisition,
moire removal, motion correction, error correction, or perspective recovery.
See [shared Python formats](../../shared/python/README.md) for the NPZ layout.

## Large inputs

The script does not silently reduce the sample count or depth. Its approximate
working-array preflight can reduce batch size and rejects estimates exceeding
`--max-memory-mb` (default 1024). That estimate excludes interpreter, PyTorch,
and backend workspace overhead, so it is not a process memory limit.

A bounded execution check at the maximum dimensions is:

```sh
python experiments/neural-dynamics/neural_key_experiment.py \
  --size 512 --colors 16 --frames 8 --samples 2 --channels 4 \
  --batch-size 1 --depths 1 --attack-epochs 0 --avalanche-inputs 0 \
  --numerics-samples 0 --trajectory-mb 0 --max-memory-mb 2048 --device cpu
```

Two samples and one round only check execution. They do not support statistical
conclusions. Covariance uses a sample-space calculation instead of allocating
a response-bits-squared matrix. Pair correlations sample at most 512 output
coordinates; avalanche probes have a separately bounded output count.

Good diffusion, balanced marginal statistics, and failed small surrogate attacks
do not establish pseudorandomness, unforgeability, security bits, or superiority
to conventional cryptography. Float arithmetic may differ across devices and
precision. The exact binary circuit used by the ZKBoo engine is a separate
construction, not a quantized implementation of this floating-point model.
