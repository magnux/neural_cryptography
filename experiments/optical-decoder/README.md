# Optical decoder — model, training, evaluation, and export

This implements the model/export foundation and capture/training pipeline in the
[optical decoder plan](../../docs/OPTICAL_DECODER.md). **`normalized-tube` remains
the default**; `longtube` adds residual depth; **`attention-tube`** combines that
body with window attention, a learned global positional anchor and variable
4–8-observation inputs grouped across four palette phases. This model requires
G64/16 colors and four permuted phases; use the fixed-frame profiles for a
G64/4-color control.

| Property | `normalized-tube` | `longtube` | `attention-tube` |
| --- | --- | --- | --- |
| G64 U spatial sizes | 256→128→64→128→256 | Same | Same |
| Outlet spatial sizes | 256→128→64 | Same | Same |
| Observations | Fixed T, 1–64 | Fixed T, 1–64 | Variable 4–8; fixed capacity 8 |
| Input channels | 3T | 3T | 24 RGB + 8 presence planes |
| Convs per resolution block / bottleneck | 2 / 2 | 4 / 8, residual pairs | Same as longtube |
| Additional learned features | None | Residual shortcut projections | Global positional tensor and two window-attention blocks |
| Parameters | 144,432 at T=5 | 279,792 at T=5 | 432,848 |
| Supported export grids | 64, 128 | 64, 128 | 64 only |

The U starts at the full rectified input size; its bottleneck matches the target
matrix. Only the outlet reduces decoded features to cells. Longtube uses two
residual pairs per outer resolution block and four at the bottleneck, followed
by one channel LayerNorm per whole block. Its inlet has two convs and one
normalization; each outlet transition remains normalized. No model has a raw
RGB shortcut. See the [architecture diagrams](../../docs/OPTICAL_DECODER.md#implemented-architectures).

Attention-tube adds a learned **32×64×64 global positional tensor**, initialized
with normal standard deviation .02, bilinearly expanded to 256×256 and added
immediately after the inlet. This anchors all subsequent computation to a learned
coordinate reference. The lattice contributes 131,072 parameters; a fully
independent 32×256×256 tensor would exceed two million. The image features still
remain at full resolution. The model then adds two attention blocks after its
64×64 bottleneck: four heads, 32 channels, 8×8 windows, a 32→64→32 feed-forward
network, pre-normalization and residual paths. The second block shifts windows
by four pixels and masks wraparound boundaries. No fixed sinusoidal or frame-slot
positional encoding is used.

Its input is **`[N,32,256,256]`**, with the first 24 planes containing eight RGB
slots and the last eight containing spatially constant binary presence flags.
Slots have fixed phase meanings: **P0a, P0b, P1a, P1b, P2a, P2b, P3a, P3b**.
Missing RGB is masked inside the model. Training selects K uniformly from
4/5/6/7/8, with at least one capture per phase and at most two. It randomizes
which phases supply second captures and placement within each phase pair.
Paired real/synthetic examples use the same K. Each training group needs two
captures per phase to support the full range; live inference can start with
four captures, **one from each phase**, rather than four of the same phase.
Recording files remain unchanged.

The output data labels are canonical phase-zero logical symbols. Reference
cells keep their physical colors and are excluded from the loss. Four exact
XOR phase-group permutations and joint horizontal/vertical flips augment
training. The input RGB, labels and masks are transformed consistently; presence
flags move with their slots. Augmentations do not alter stored captures or
validation, and transformed training labels are not decoded as original bytes.

Evaluation uses fixed seed-7 random choices within each phase, forming disjoint
pairs and nested 4/5/6/7/8 subsets across the four phases. It reports each count
separately, including direction-specific CER and exact recovery. Classical fusion
uses the same actual observations, reversing each capture's own palette mapping
before combining logical scores. Candidates now represent a logical matrix
across phases, so compare per-direction rates and observation budgets rather
than old single-phase denominators. Checkpoint selection averages all five K
results and balances real directions when a session sidecar is supplied.
The immediate target is **at least 50% exact recovery for phone→PC**, reported
per K; this is a goal, not a measured result for the new architecture.

The [next training recipe](../../docs/OPTICAL_DECODER_TRAINING.md#recommended-next-training-run)
trains **`--model attention-tube --frames 8` fresh** at `1e-4` for 32,000 updates
into `attention-phase8`, using **batch 16 × accumulation 2 = effective batch 32**.
Rebuild the paired synthetic corpus with coherent four-phase groups before this
run. Use `real-train`, `target-synthetic-train`, `real-validation` and the current
direction sidecar. The model retains masked cross-entropy, paired 50:50
real/synthetic sampling and mild real-only photometric augmentation. Do not pass
`--initialize` from a previous same-phase attention checkpoint: tensor shapes
alone cannot establish the new input/output meaning. Fixed-profile tube/longtube
checkpoints remain loadable for their own evaluation or continuation.

The ongoing longtube history through 26,000 updates reached 82/144 exact
five-frame groups at step 24,000: PC→phone 82/96, phone→PC 0/48. Phone→PC training
CE fell from .814 at 16k to .643 at 26k while validation CE rose from 1.165 to
1.360. The sampler already allocates equal real examples to both directions;
additional physical sessions improve diversity. These are ongoing-run results,
not measurements of phase-grouped attention-tube.

Capture calibration stores `x = clip(c/127.5 - 1, -1.5, 1.5)`. Every model clips
RGB to [-1,1] before learned processing, including exported ONNX. Presence
planes remain binary. This leaves stored recordings, labels and hashes intact.
No trained default weights are shipped; untrained exports only establish
numerical feasibility. Live optical transport still uses the classical decoder.
Attention-tube's spatial anchor is G64-specific; future G128 attention work needs
separate design/training. Existing tube/longtube G128 exports check structure and
parity, not cross-size recognition with G64-trained weights.

Synthetic corruption includes curved moiré; linear, rotational and radial motion
blur; RGB-offset exposure windows; noise, white balance and exposure; rolling
shutter, shadows/reflections, glare and native-capture resolution loss. Generation
uses the canonical sender-derived clean targets and retains their split and
session identity. Noisy camera crops are never treated as clean truth. More
synthetic variants do not add new physical capture conditions.

Training defaults to 32,000 updates, AdamW weight decay .01, LR `1e-4` fresh or
`5e-5` initialized, and full validation every 1,000 updates. A tqdm bar shows loss
and key metrics; `history.json` retains evaluation and interval source/group
statistics. `--no-progress` hides the bar. Initialized same-profile runs preserve
their initial checkpoint if continuation fails to improve it. Full commands are
in the [capture/training guide](../../docs/OPTICAL_DECODER_TRAINING.md).

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
  --checkpoint checkpoints/optical-decoder/attention-phase8/best.pt \
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
groups; attention-tube also averages equally across its five observation counts.
Initialized runs evaluate and preserve their starting checkpoint at
step zero, so worse fine-tuning cannot replace that baseline. Training settings,
session mappings and their hash remain in the local run/checkpoint; exports keep
only the understood public scalar/hash provenance.

CPU execution is sufficient for export and small pipeline checks. If installing
PyTorch separately, choose the CPU or CUDA build appropriate for the machine;
full training is intended for the RTX 4090. Exporting these models
does not need a GPU. There is no Hugging Face dependency or training framework.

By default, export writes `artifacts/smoke/` beside this README. That directory
is generated output and should not be committed. `--out PATH`, `--seed 7`, and
`--threads 1` are configurable. The command exports the supported grids for the selected profile: G64 only
for attention-tube; G64 and G128 with the same weights for the two fixed-frame
profiles. The latter checks structure/parity, not G128 optical accuracy for a
G64-trained checkpoint. The seed is public diagnostic
data, unrelated to runtime cryptographic randomness.

Export performs ONNX graph validation and runs two distinct inputs per shape
through PyTorch and Python ONNX Runtime before writing the final manifest. The
graph uses opset 17 and the explicit TorchScript exporter (`dynamo=False`), with
Clip, Conv, ReLU, Concat, Resize, and constants, together with channel-normalization
reductions/basic arithmetic and residual additions. Attention-tube also uses
standard MatMul, Softmax, reshape/transpose, mask and projection operations. This avoids an additional
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
`status: "untrained"` or `"trained"`. It declares the exact profile, required
`inputLayout`, `framesPerInput` capacity, `minFramesPerInput`, `frameCounts`,
`channels`, grid cases, `outputLabelSpace`, `phaseMasks` and `networkInputClip`.
Trained exports also require
`checkpointSha256` and public `training` metadata. Regenerate local exports after
updating; there is no earlier-bundle reader.

| Profile layout | Capacity / allowed counts | Input shape |
| --- | --- | --- |
| `stacked-rgb` for tube/longtube | T / `[T]`, T=1…64 | `[1,3T,4G,4G]` |
| `phase-paired-rgb-presence-planes` for attention-tube | 8 / `[4,5,6,7,8]` | `[1,32,256,256]` |

The fixed-frame RGB order is `R0,G0,B0,R1,G1,B1,...`. Attention-tube uses that
order for its first 24 planes, with two slots per phase, followed by eight
constant binary presence planes. Its metadata declares `phaseMasks: [0,15,7,8]`
and `outputLabelSpace: "logical"`; fixed-frame profiles use physical labels.
Each export case declares `framesAvailable` for its numerical fixture and
includes SHA-256 and byte count for `model-G.onnx`,
`input-G.f32` and `expected-G.f32`; tensors are little-endian contiguous float32.
The single ONNX input remains named `pixels`. Attention input storage is 8 MiB
per batch-one example before runtime buffers. Fixed-frame raw input storage is
`0.75 × T` MiB at G64 or `3 × T` MiB at G128. Supplying fewer observations to
attention-tube still executes its fixed eight-slot graph.

The output is named `logits`, shape `[1,16,G,G]`, without softmax. Fixed-frame
profiles predict physical palette classes, so their receiver adapter reverses
the observed phase mapping. Attention-tube predicts **canonical logical data
classes directly** and must not receive a second phase reversal. Both paths
exclude reference cells before passing scores to the current error-recovery
code. The grid includes references, but their logits are not payload evidence.

Capture preprocessing receives calibrated RGB in camera-byte units, divides by
127.5, subtracts 1, and clips to `[-1.5,1.5]`. The network then clips RGB to `[-1,1]`
before learned processing; manifests require `networkInputClip: [-1,1]`. Intermediate arithmetic uses float64,
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
Python/ONNX checks pass. Test WASM as well as WebGPU where available, each
supported grid size, the variable observation counts and camera-active runs. Record output agreement, warm median/p95 timing,
load time, backend, model hash, and UI responsiveness.

## Files and acceptance

- `model.py`: default full-resolution tube, residual longtube and masked
  attention-tube with a learned positional anchor and local window attention.
  All clip RGB before learned layers and keep the bottleneck at the target grid.
- `contract.py`: shape constants, RGB normalization, fixture generation, numeric
  parity, and little-endian serialization.
- `export.py`: static ONNX graphs, optional explicit native-runtime validation,
  and manifest generation with an honest validation status.
- `tests/test_decoder.py`: tensor contract, model structure/input sensitivity,
  graph shape checks, and actual runtime parity.
- `checkpoint.py`: one bounded checkpoint loader for training, evaluation, and export.
- `data.py`: hashed local datasets and whole-session/payload split checks.
- `bursts.py`: same-phase fixed-frame subsets or four-phase masked groups,
  canonical target validation and deterministic nested evaluation, without rewriting recordings.
- `synthetic.py`: coherent four-phase groups or independent corruption bursts with moiré, blur/noise,
  exposure/WB, rolling-shutter bands/skew, soft shadows/reflections, drifting
  glare, native capture-resolution loss, channel-offset RGB motion blur and
  per-burst severity ranges.
- `augmentation.py`: mild real-only noise/tint/exposure, plus training-only exact
  phase permutations and joint spatial flips with transformed labels/masks.
  Recording files and evaluation inputs remain unchanged.
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
