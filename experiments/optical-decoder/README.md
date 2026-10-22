# Optical decoder — model, training, evaluation, and export

This implements M1 feasibility and the M2 capture/training pipeline from the
[optical decoder plan](../../docs/OPTICAL_DECODER.md). The
**121,792-parameter U-Net** has fixed browser profiles for physical grid sizes
64 and 128. The submitted device reports support starting at G64. See the
[capture and training guide](../../docs/OPTICAL_DECODER_TRAINING.md) for the full
workflow, from browser captures through synthetic generation and RTX 4090 training.

No trained production/default weights are shipped. Export without a checkpoint
uses random weights for feasibility checks; export with `--checkpoint` uses the
trained weights and their provenance. The existing classical decoder remains
responsible for live transport until held-out recovery validates a candidate.
Numerical parity measures runtime agreement, not optical accuracy.

## Setup and export

From the repository root, in your Python virtual environment:

```sh
python -m pip install -r experiments/optical-decoder/requirements.txt
python experiments/optical-decoder/export.py
python -m unittest discover -s experiments/optical-decoder/tests -v
```

For a selected trained checkpoint:

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/real/best.pt \
  --out experiments/optical-decoder/artifacts/selected
npm ci --prefix experiments/optical-decoder/browser-runtime
npm run prepare:decoder -- --artifacts experiments/optical-decoder/artifacts/selected
npm run test:decoder:wasm -- --artifacts experiments/optical-decoder/artifacts/selected
```

The shared checkpoint loader uses `torch.load(weights_only=True)`, checks finite
float32 parameter shapes, and validates dataset split provenance. Public model
metadata includes checkpoint hashes and bounded scalar training settings; local
session identities and filesystem paths stay in the checkpoint. Training uses
`best.pt`, selected by validation exact-frame recovery with loss as the tie-break.

CPU execution is the default and sufficient for this step. If installing
PyTorch separately, choose the CPU or CUDA build appropriate for the machine;
the later training milestone will use the RTX 4090. Exporting these models
does not need a GPU. There is no Hugging Face dependency or training framework.

By default, export writes `artifacts/smoke/` beside this README. That directory
is generated output and should not be committed. `--out PATH`, `--seed 7`, and
`--threads 1` are configurable. The command always exports both 64 and 128
profiles, using the **same** initialized weights. The seed is public diagnostic
data, unrelated to runtime cryptographic randomness.

Export performs ONNX graph validation and runs two distinct inputs per shape
through PyTorch and Python ONNX Runtime before writing the final manifest. The
graph uses opset 17 and the explicit TorchScript exporter (`dynamo=False`), with
Conv, ReLU, Concat, Resize, and Constant operations. This avoids an additional
`onnxscript` dependency. Exact installed package versions and operator names
are recorded in the manifest. An unsupported graph or failed parity check stops
the export; it does not produce a passing report.

If the native Python runtime cannot initialize on a restricted machine, export
with `--skip-ort-check` only to obtain a bundle for browser validation. Graph
checking and PyTorch fixture generation still run, but the manifest explicitly
records `pythonParityStatus.status: "not_run"`; it never calls the native check
passed. Native checks run in a subprocess so a backend abort becomes a clear
export error. The numerical browser test must still pass on the exported files.

Use the repository's decoder preparation command documented with the browser
bench to copy this bundle and the pinned local ONNX Runtime Web assets into the
served app. The browser performs all inference locally. These Python tools are
build/test tools, not a service used during an authentication attempt.

## Tensor and bundle contract

The current manifest has `format: "optical-decoder"`, `version: 1`, and
`status: "untrained"` or `"trained"`. Trained exports require `checkpointSha256`
and public `training` metadata. Regenerate old local exports after updating;
there is no earlier-format reader. Each case contains a SHA-256 digest and byte count for:

| File | Meaning |
| --- | --- |
| `model-64.onnx` / `model-128.onnx` | Static batch-one inference graph, identical weights for both sizes |
| `input-64.f32` / `input-128.f32` | Diagnostic tensor, little-endian float32, contiguous NCHW |
| `expected-64.f32` / `expected-128.f32` | PyTorch logits, same numeric encoding |

The input is named `pixels`, shape `[1,3,4G,4G]`; output is named `logits`, shape
`[1,16,G,G]`. The grid includes reference cells. The head predicts **physical**
palette classes and performs no softmax. The receiver masks inactive classes,
maps palette phases back into logical symbol order, and excludes reference
cells before passing scores to the current error-recovery code.

Preprocessing receives calibrated RGB in camera-byte units, divides by 127.5,
subtracts 1, and clips to `[-1.5,1.5]`. Intermediate arithmetic uses float64,
then writes float32 NCHW; do not clamp calibrated RGB to `[0,255]` first.
`contract.normalize_rgb()` mirrors this final receiver step. The JavaScript
receiver continues to own detection, homography, and color calibration.

The diagnostic inputs contain spatial/channel variation but **no known payload
labels**. They are deterministic numerical tests, not synthetic training data.
Expected logits are used only to compare inference output, never to repair a
camera payload. Both runtimes check the parity criterion elementwise:

`abs(actual - expected) <= atol + rtol * abs(expected)`.

| Check | `atol` | `rtol` |
| --- | --- | --- |
| Native Python ONNX Runtime | `2e-5` | `2e-4` |
| Browser ONNX Runtime Web | `1e-4` | `1e-3` |

The manifest's `tolerance` field records the stricter native check. Browser
reports include their own thresholds to allow for different execution backends.

The browser must verify bundle lengths and hashes and reject non-finite outputs.
Browser measurements on the actual phone and PC remain necessary even after
Python/ONNX checks pass. Test WASM as well as WebGPU where available, both grid
sizes, and camera-active runs. Record output agreement, warm median/p95 timing,
load time, backend, model hash, and UI responsiveness.

## Files and acceptance

- `model.py`: the small single-frame U-Net, with no normalization layers.
- `contract.py`: shape constants, RGB normalization, fixture generation, numeric
  parity, and little-endian serialization.
- `export.py`: static ONNX graphs, optional explicit native-runtime validation,
  and manifest generation with an honest validation status.
- `tests/test_decoder.py`: tensor contract, model structure/input sensitivity,
  graph shape checks, and actual runtime parity.
- `checkpoint.py`: one bounded checkpoint loader for training, evaluation, and export.
- `data.py`: hashed local datasets and whole-session/payload split checks.
- `synthetic.py`: correlated moiré, blur, sampling, color, and noise corruptions.
- `train.py`: shape/palette-grouped AdamW training, real/synthetic mixtures, and validation selection.
- `evaluate.py`: held-out cell/confidence metrics and exact recovery through the canonical JS decoder.
- `../../scripts/decoder-dataset.mjs`: canonical clean examples, offline joins, baseline refresh, and CRC/RS evaluation.
- `tests/test_training.py`, `tests/test_export_checkpoint.py`: training/decoder integration, split isolation, masking, and checkpoint export checks.

Without optional model/export packages, the tests explicitly skip those portions
while still testing the NumPy tensor contract. Install the full requirements
before claiming export validation. A subprocess probes native ONNX Runtime;
unavailable native execution is a separate visible skip. The short CPU training
smoke validates wiring only; it is not a useful receiver checkpoint. M2 remains
open until a trained model improves held-out real recovery without regressing
the 4-color control. Use the [training guide](../../docs/OPTICAL_DECODER_TRAINING.md)
and [roadmap](../../docs/ROADMAP.md) for the next device work.
