# Optical decoder — model, training, evaluation, and export

This implements M1 feasibility and the M2 capture/training pipeline from the
[optical decoder plan](../../docs/OPTICAL_DECODER.md). The
**normalized-compact** default and the **normalized-tube/compact/wide** comparison profiles support
1–64 RGB exposures, with fixed browser grids 64 and 128. The default has
**125,600 parameters at five frames**, with a 16→16 inlet, 16/32/64 features,
channel LayerNorm after every hidden convolution, and no direct RGB readout.
It stays below 500k through 64 frames (151,088 parameters). See the
[capture and training guide](../../docs/OPTICAL_DECODER_TRAINING.md) for the full
workflow, from browser captures through synthetic generation and RTX 4090 training.

No trained production/default weights are shipped. Export without a checkpoint
uses random weights for feasibility checks; export with `--checkpoint` uses the
trained weights and their provenance. The existing classical decoder remains
responsible for live transport until held-out recovery validates a candidate.
Numerical parity measures runtime agreement, not optical accuracy.

Fresh training and export default to `normalized-compact`. `--model normalized-tube`,
`--model compact` and `--model wide` are explicit comparison profiles. Profile inference and
checkpoint loading validate all parameter shapes; changing internal width needs
fresh weights. Do not initialize tube from a normalized-compact or wide checkpoint.

The current optional **normalized-tube** keeps the dense crop at full resolution
through a stride-1 3T→32→32 inlet and a constant-width 32-channel U-Net. At G64,
the U-Net spatial sides are **256→128→64→128→256**, with learned feature skips
and 64-channel decoder concatenations. Two normalized stride-2 outlet convolutions
then reduce **256→128→64**, followed by the 32→16 head. At G128 all spatial sides
double. All **fourteen hidden convolutions** have LayerNorm/ReLU; no direct RGB
path exists. Its `140112 + 864T` parameters give **144,432 at T=5** and **195,408 at T=64**.
This tests full-resolution feature processing before output reduction. Convolution
MACs are **4,928,307,200** at G64/T=5 versus 168,558,592 for normalized-compact
(about 29.24×), and activation memory also increases. This is an operation count,
not measured browser latency. See the
[tube diagram and comparison](../../docs/OPTICAL_DECODER.md#normalized-tube--full-resolution-experiment).

This replaces the earlier tube with inlet downsampling. The required outlet
distinguishes the current checkpoint structure; earlier tube checkpoints are
rejected rather than reinterpreted. Start the full-resolution model from scratch.

Capture calibration preserves overshoot: `x_capture = clip(c/127.5 - 1, -1.5, 1.5)`.
Every model begins with `x_network = clip(x_capture, -1, 1)` **before any learned
layer**, including any RGB readout. This clamp is part of the exported ONNX graph,
so Python training/evaluation and browser inference enforce the same range.
Stored recordings, joined tensors and checksums remain unchanged. Existing
synthetics can be reused. A common input range fixes the clipping mismatch; it
does not establish that synthetic and camera artifact distributions are identical.

Recording files remain unchanged: each stores one RGB crop. `--frames 5` stacks
five distinct same-phase exposures along the channel axis. Explicitly initializing
from a single-frame pilot expands only the RGB-facing convolution weights, repeats
them T times and divides by T; biases and the backbone are preserved.
Identical repeated exposures reproduce the original model function at initialization.
Checkpoint loading itself remains strict; export never silently changes frame count.
To change an existing multiframe model's count, initialize again from the single-frame
pilot. Each training burst must contain at least T distinct captures.

There is one plain masked-cross-entropy training path, with 32,000 updates by
default, AdamW weight decay `.01`, and learning rate `1e-4` for fresh weights or
`5e-5` for an initialized model. Full validation runs every 1,000 updates. The
unsuccessful adaptation recipe and its extra loss/scheduler options have been
removed. Mild real-only noise/tint/exposure augmentation is now enabled by three
independent strength controls; validation and stored captures stay unchanged.
Source/group metrics, baseline protection, and the optional
session-direction sidecar remain useful general tools. A throttled tqdm bar shows
training CE, learning rate and latest validation CER/exact recovery. Validation
summaries stay short; `history.json` retains full records at each evaluation,
including interval source/group metrics. Use `--no-progress` to hide the bar;
redirected output automatically uses concise text without terminal control codes.

The [recommended next run](../../docs/OPTICAL_DECODER_TRAINING.md#recommended-next-training-run)
trains **`--model normalized-tube --frames 5` from scratch** at `1e-4` for 32,000
updates into `normalized-tube5-fullres`, using `--batch-size 2 --accumulate 8`
for effective batch 16 with lower peak memory. Reuse `target-synthetic-resolution-train`, the
existing real train/validation datasets, and direction sidecar; no synthetic
regeneration is required. Keep mild real augmentation and do not pass
`--initialize`, since this topology is incompatible with earlier tube and
normalized-compact checkpoints. The previous resolution/glare
recipe remains a corpus reference, and existing run outputs remain available.
`--paired-targets` places each real stack alongside a
synthetic stack of that exact session/payload/frame/phase, with matching target
hashes. Each side uses five distinct exposures from its own burst; independent
synthetic corruption bursts and palette phases are never mixed. Target content
is aligned between sources; held-out real captures test generalization. No clean
target is used during inference.

Synthetic generation includes curved moiré, integrated arbitrary-direction linear,
rotational and radial-zoom motion blur, Gaussian blur/noise, white-balance and
exposure shifts, drifting rolling-shutter bands and bounded row-wise skew, plus
soft, rotated rectangle/ellipse shadows and reflections. Drifting elliptical
glare adds localized saturation and a broad tinted halo that reduces contrast.
`--severity-min` enables per-burst severity draws up to `--severity`.
`--bursts-per-target`
controls independent corruption trajectories; `--frames` controls correlated
exposures within each. The recommended paired corpus uses 32 bursts × 12 frames
per target. These are calibrated-crop approximations, not a full camera simulator.
The capture-resolution stage selects 75% of bursts by default. After the other
effects and noise, it area-downsamples to an integer native side drawn uniformly
from .25–1 times the crop side, applies a small Gaussian blur there, and bilinear-
upsamples back. The side and blur stay fixed within a burst. Default maximum sigma
is .35 native pixels, with each selected burst drawing .5–1 times that maximum.
G64 therefore simulates 64–256 native pixels per side before returning to the
same 256×256 model input; G128 uses 128–512 before returning to 512×512.
This addresses the small-phone-screen-in-PC-camera case without changing the
network, labels or recording files. One native pixel per cell is a hard endpoint,
not a promise that the data remains recoverable. Capture scale/blur are independent
of severity, except that severity zero keeps the clean crop unchanged. The stage
uses its own RNG; seed 41 retains the earlier glare recipe's other artifact draws.
Use `--capture-probability`, `--capture-min-scale`, `--capture-max-scale` and
`--capture-blur-sigma` to control it; zero probability bypasses it completely.
See the [training guide](../../docs/OPTICAL_DECODER_TRAINING.md#generate-the-synthetic-pilot)
for controls and the retained corpus-generation recipe.

Source pools are indexed once. Offline manifests allow 256 MiB without a fixed
row cap; generation defaults to seven frames, one burst, and a configurable
1 TiB estimated output budget (`--max-output-mib 0` disables that estimate).
Browser capture limits are independent. Original recordings and joined real
corpora retain their existing formats and identities.

The retained single-frame baseline recovered 69/768 captures. Paired single-frame
training recovered 68/768 at step 22k, with phone→PC still at zero and worse
validation loss despite falling training loss. The normalized five-frame run
recovered **19/144 (13.19%)** fixed-random validation groups at step 31,000, with
32.43% cell error and no wrong acceptances. PC→phone recovered 19/96; phone→PC
remained 0/48. Classical fusion recovered 0/144 with the same observations.
The newer fresh normalized-compact history reaches 20k with CE 1.0514, cell error
32.759%, and 8/144 exact groups; the original run at 20k had CE 1.1657, cell error
32.995%, and the same 8/144. Phone→PC still recovers nothing, with validation CE
rising while its training CE falls. This is a remaining direction-specific
generalization gap. Only that partial history is available; no final-run or tube
result is claimed. The current tube changes feature resolution and output reduction;
the data split and loss stay unchanged.
Multiframe evaluation applies a fixed seed-7 random permutation within each burst
before making disjoint stacks, and compares classical fusion over the exact same exposures; its denominator differs from a single-frame
report. Existing checkpoints can be reevaluated under the corrected grouping
without retraining; the method and seed appear in each report.
See [architecture diagrams](../../docs/OPTICAL_DECODER.md#implemented-architectures)
for the implemented narrower normalized model. No live throughput improvement is claimed.

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
  --checkpoint checkpoints/optical-decoder/normalized-tube5-fullres/best.pt \
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
With explicit real groups, those selection metrics are averaged equally across
groups. Initialized runs evaluate and preserve their starting checkpoint at
step zero, so worse fine-tuning cannot replace that baseline. Training settings,
session mappings and their hash remain in the local run/checkpoint; exports keep
only the understood public scalar/hash provenance.

CPU execution is sufficient for export and small pipeline checks. If installing
PyTorch separately, choose the CPU or CUDA build appropriate for the machine;
full training is intended for the RTX 4090. Exporting these models
does not need a GPU. There is no Hugging Face dependency or training framework.

By default, export writes `artifacts/smoke/` beside this README. That directory
is generated output and should not be committed. `--out PATH`, `--seed 7`, and
`--threads 1` are configurable. The command always exports both 64 and 128
profiles, using the **same** initialized weights. The seed is public diagnostic
data, unrelated to runtime cryptographic randomness.

Export performs ONNX graph validation and runs two distinct inputs per shape
through PyTorch and Python ONNX Runtime before writing the final manifest. The
graph uses opset 17 and the explicit TorchScript exporter (`dynamo=False`), with
Clip, Conv, ReLU, Concat, Resize, and constants. Normalized profiles also use
channel-normalization reductions/basic arithmetic; wide adds AveragePool. This avoids an additional
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
`status: "untrained"` or `"trained"`, plus required integer `framesPerInput` from 1 through 64
and `channels: 3 * framesPerInput`. Trained exports require `checkpointSha256`
and public `training` metadata. Regenerate old local exports after updating;
there is no earlier-format reader. Each case contains a SHA-256 digest and byte count for:

| File | Meaning |
| --- | --- |
| `model-64.onnx` / `model-128.onnx` | Static batch-one inference graph, identical weights for both sizes |
| `input-64.f32` / `input-128.f32` | Diagnostic tensor, little-endian float32, contiguous NCHW |
| `expected-64.f32` / `expected-128.f32` | PyTorch logits, same numeric encoding |

The input is named `pixels`, shape `[1,3T,4G,4G]` for integer T=1…64; channels are
`R0,G0,B0,R1,G1,B1,...`. A G64 five-frame input is `[1,15,256,256]`.
Raw float input memory is `0.75 × T` MiB at G64 and `3 × T` MiB at G128,
before batches, activations and runtime copies. Exported graphs have a fixed T;
changing the capture count requires a corresponding trained/exported model.
The output is named `logits`, shape
`[1,16,G,G]`. The grid includes reference cells. The head predicts **physical**
palette classes and performs no softmax. The receiver masks inactive classes,
maps palette phases back into logical symbol order, and excludes reference
cells before passing scores to the current error-recovery code.

Capture preprocessing receives calibrated RGB in camera-byte units, divides by
127.5, subtracts 1, and clips to `[-1.5,1.5]`. The network then clips to `[-1,1]`
as its first graph operation; manifests require `networkInputClip: [-1,1]`. Intermediate arithmetic uses float64,
then writes float32 NCHW; do not clamp calibrated RGB to `[0,255]` first.
`contract.normalize_rgb()` mirrors this capture step; `OpticalDecoder.forward()`
provides the subsequent network-input clamp. The JavaScript
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

- `model.py`: normalized-compact default plus normalized-tube/compact/wide comparison U-Nets;
  all graphs apply final input clipping before learned layers.
- `contract.py`: shape constants, RGB normalization, fixture generation, numeric
  parity, and little-endian serialization.
- `export.py`: static ONNX graphs, optional explicit native-runtime validation,
  and manifest generation with an honest validation status.
- `tests/test_decoder.py`: tensor contract, model structure/input sensitivity,
  graph shape checks, and actual runtime parity.
- `checkpoint.py`: one bounded checkpoint loader for training, evaluation, and export.
- `data.py`: hashed local datasets and whole-session/payload split checks.
- `bursts.py`: distinct same-burst training subsets and deterministic disjoint
  evaluation stacks, without rewriting recorded samples.
- `synthetic.py`: independently seeded corruption bursts with moiré, blur/noise,
  exposure/WB, rolling-shutter bands/skew, soft shadows/reflections, drifting
  glare, native capture-resolution loss and per-burst severity ranges.
- `augmentation.py`: mild real-only training noise/tint/exposure, with an
  independent RNG and no changes to labels, recordings or evaluation.
- `train.py`: plain masked CE, paired-target option, source/group metrics,
  validation selection and initialized-baseline protection.
- `sampling.py`: indexed source/group/session/payload/phase sampling, exact real/
  synthetic target pairs, update quotas, and repeated-burst avoidance.
- `session_groups.py`: explicit capture-direction sidecars validated against joined
  datasets, without changing recording formats.
- `evaluate.py`: held-out cell/confidence, session/group and balanced-selection
  metrics, plus exact recovery and equal-observation classical fusion through the
  canonical JS decoder.
- `../../scripts/decoder-dataset.mjs`: canonical clean random/recorded targets,
  offline joins, baseline refresh, and CRC/RS evaluation.
- `tests/test_training.py`, `tests/test_export_checkpoint.py`: training/decoder integration, split isolation, masking, and checkpoint export checks.

Without optional model/export packages, the tests explicitly skip those portions
while still testing the NumPy tensor contract. Install the full requirements
before claiming export validation. A subprocess probes native ONNX Runtime;
unavailable native execution is a separate visible skip. The short CPU training
smoke validates wiring only; it is not a useful receiver checkpoint. M2 remains
open until a trained model improves held-out real recovery without regressing
the 4-color control. Use the [training guide](../../docs/OPTICAL_DECODER_TRAINING.md)
and [roadmap](../../docs/ROADMAP.md) for the next device work.
