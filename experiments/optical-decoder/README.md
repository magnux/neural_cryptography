# Optical decoder — model, training, evaluation, and export

This implements the model/export foundation and capture/training pipeline in the
[optical decoder plan](../../docs/OPTICAL_DECODER.md). Only two architectures remain:
**`normalized-tube` is the default**, and **`longtube`** adds residual depth. Both
support 1–64 stacked RGB exposures, 32 hidden channels and 16 physical-palette
logits per cell. Start with G64/16 colors and a G64/4-color control.

| Property | `normalized-tube` | `longtube` |
| --- | --- | --- |
| G64 U spatial sizes | 256→128→64→128→256 | Same |
| Outlet spatial sizes | 256→128→64 | Same |
| Main convs per resolution block / bottleneck | 2 / 2 | 4 / 8, arranged in residual pairs |
| Hidden main convolutions / LayerNorms | 14 / 14 | 28 / 8, plus four shortcut projections |
| Parameters, five frames | 144,432 | 279,792 |
| Parameters, 64 frames | 195,408 | 330,768 |
| G64/T5 convolution MACs | 4,928,307,200 | 8,363,442,176 |

The U starts at the full rectified input size; its bottleneck matches the target
matrix. Only the outlet reduces decoded features to cells. This preserves
cell-level spatial evidence for random payloads, without asserting that
larger-scale illumination/channel correlations cannot be useful. Longtube uses
two residual pairs per outer resolution block and four at the bottleneck,
followed by **one channel LayerNorm per whole block**. Its inlet has two convs
and one normalization; each of the two outlet transitions remains normalized.
Neither model has a direct RGB shortcut. See the
[architecture diagrams](../../docs/OPTICAL_DECODER.md#implemented-architectures).

Longtube has about 1.70× tube's convolution MACs at G64/T5 and higher activation
memory. MACs exclude normalization/resampling/memory traffic and are not measured
browser latency. G128 remains a structural export/parity case with a 512×512
input and 128×128 bottleneck. Train a separate G128 network if needed; no multi-size
routing is implemented or assumed validated.

The existing full-resolution tube code/state is unchanged, so its trained
checkpoint remains loadable for evaluation or continuation. Longtube needs fresh
weights; cross-profile initialization and removed architectures are rejected.
No trained default weights are shipped. Exports without a checkpoint use random
weights for feasibility checks. Live optical transport still uses the classical
decoder; numerical parity establishes runtime agreement, not optical accuracy.

The submitted tube run selected step 29k of 32k: **68/144 exact five-frame groups
(47.22%)**, **17.31% cell error**, CE **.4661**. PC→phone recovered 68/96 (70.83%);
phone→PC recovered 0/48 with 46.94% cell error. Final 32k also recovered 68/144, with
17.34% cell error. Classical fusion recovered none using the same exposures.
The run used batch 16 × accumulation 2, effective 32; earlier runs used effective 16,
so this is not a pure architecture ablation. See the
[training result](../../docs/OPTICAL_DECODER_TRAINING.md#current-models-and-latest-result)
and [validation record](../../docs/VALIDATION.md).

The [next training recipe](../../docs/OPTICAL_DECODER_TRAINING.md#recommended-next-training-run)
trains **`--model longtube --frames 5` fresh** at `1e-4` for 32,000 updates into
`longtube5`, with **batch 8 × accumulation 4 = effective 32** to preserve the successful
update size while allowing for larger activation memory. Generate
`target-synthetic-train` using the existing clean training targets, seed 41,
32 bursts × 12 frames, the resolution/glare recipe and
`--rgb-offset-probability .6 --rgb-offset-pixels .5`. Keep paired 50:50 real/synthetic
sampling, direction balancing, ordinary masked cross-entropy and the same mild
real-only noise/tint/exposure augmentation. Do not pass `--initialize` for this
new architecture. Keep the current tube checkpoint as the comparison.

Capture calibration stores `x = clip(c/127.5 - 1, -1.5, 1.5)`. Both models begin
with `clip(x, -1, 1)` before learned processing, including in exported ONNX.
Recordings, labels and hashes are unchanged. A common input range does not erase
artifact differences between synthetic and real sources.

`--frames T` changes inlet channels automatically; the hidden width stays 32.
Training samples T distinct random captures from one burst/session/target/phase
and shuffles their RGB slots. Evaluation uses a fixed seed 7 permutation within
each burst, then forms disjoint groups; classical fusion uses those same samples.
No chronological-neighbor grouping, duplicate padding or cross-burst mixing is
used. `synthetic.py --frames` means generated exposures per burst; training
`--frames` means exposures in one model input. Each training burst must have at
least T members. Explicit single→T initialization within the same profile repeats
and divides the inlet weights by T; changing one multiframe checkpoint's count
is rejected. Loading/export never changes frame count implicitly.

Synthetic corruption includes curved moiré; linear, rotational and radial motion
blur; blur/noise, white balance and exposure; rolling-shutter bands/skew; soft
shadows/reflections; glare; and native-capture resolution loss. New RGB offsets
shift channel exposure windows along the same motion trajectory, approximating
motion-related color misregistration. They apply only when motion is active,
remain centered/bounded, and use an independent RNG to leave other artifact
draws unchanged. This is a calibrated-crop approximation, not a claim about every
camera's channel exposure timing. The capture-resolution stage still simulates
64–256 native pixels per side at G64 before reexpanding to 256. It cannot restore
information lost at low native resolution.

Generation has no fixed row cap, a 256 MiB manifest limit, and a configurable 1 TiB
estimated output budget (`--max-output-mib 0` disables that estimate). Browser
capture memory limits remain separate. Pairing aligns target content between
real/synthetic training samples; clean references never enter live inference.
Whole-session/payload splits and held-out real evaluation remain mandatory.

Training defaults to 32,000 updates, AdamW weight decay `.01`, LR `1e-4` fresh or
`5e-5` initialized, and full validation every 1,000 updates. A tqdm bar shows loss
and important metrics; `history.json` retains full evaluation records and interval
source/group metrics. `--no-progress` hides the bar. Initialized runs preserve
their initial checkpoint if continuation fails to improve it. The
[capture/training guide](../../docs/OPTICAL_DECODER_TRAINING.md) contains all commands.

## Setup and export

To add recorded sessions, use the
[folder-based rebuild workflow](../../docs/OPTICAL_DECODER_TRAINING.md#organize-recordings-and-rebuild-datasets).
It joins `local-data/{phone,pc}/{train,val}` recordings, reconstructs paired clean
targets, regenerates the standard synthetic corpora and writes direction groups.
Run it after active training finishes; the standard generated directories are
replaced. Raw recordings and checkpoints remain intact.

From the repository root, in your Python virtual environment:

```sh
python -m pip install -r experiments/optical-decoder/requirements.txt
python experiments/optical-decoder/export.py
python -m unittest discover -s experiments/optical-decoder/tests -v
```

For a selected trained checkpoint:

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/longtube5/best.pt \
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
shapes, using the **same** selected weights. This checks structure/parity,
not G128 optical accuracy for a G64-trained checkpoint. The seed is public diagnostic
data, unrelated to runtime cryptographic randomness.

Export performs ONNX graph validation and runs two distinct inputs per shape
through PyTorch and Python ONNX Runtime before writing the final manifest. The
graph uses opset 17 and the explicit TorchScript exporter (`dynamo=False`), with
Clip, Conv, ReLU, Concat, Resize, and constants, together with channel-normalization
reductions/basic arithmetic and residual addition for longtube. This avoids an additional
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

- `model.py`: default full-resolution normalized-tube and residual longtube;
  both clip input before learned layers and keep the bottleneck at the target grid.
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
  glare, native capture-resolution loss, channel-offset RGB motion blur and
  per-burst severity ranges.
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
- `../../scripts/rename-decoder-recordings.mjs`: checksum-prefixed filenames for
  current sender/receiver exports, without rewriting recording contents.
- `../../scripts/rebuild-decoder-data.mjs`: discover split folders, validate
  pairing and isolation, and rebuild the fixed real/paired-synthetic datasets
  and receiver-direction sidecar.
- `tests/test_training.py`, `tests/test_export_checkpoint.py`: training/decoder integration, split isolation, masking, and checkpoint export checks.

Without optional model/export packages, the tests explicitly skip those portions
while still testing the NumPy tensor contract. Install the full requirements
before claiming export validation. A subprocess probes native ONNX Runtime;
unavailable native execution is a separate visible skip. The short CPU training
smoke validates wiring only; it is not a useful receiver checkpoint. M2 remains
open until a trained model improves held-out real recovery without regressing
the 4-color control. Use the [training guide](../../docs/OPTICAL_DECODER_TRAINING.md)
and [roadmap](../../docs/ROADMAP.md) for the next device work.
