# Neural corrector

Original Neural-Corrector code and documentation are part of **Neural Cryptography**
by **Alejandro Hernandez Ruiz (magnux)**, licensed under [Apache-2.0](../../LICENSE).
Preserve the project [NOTICE](../../NOTICE) on redistribution; see
[licensing scope](../../docs/LICENSING.md) for model/data releases and
[citation metadata](../../CITATION.cff) for research use.

A family of small PyTorch models that converts one or more rectified camera
frames into a corrected optical matrix. The network predicts palette-class
scores at each cell; the existing optical byte decoder remains responsible for
symbol packing, CRC and Reed–Solomon recovery. Expected payloads are offline
supervision and diagnostics, never network inputs.

The [architecture guide](../../docs/NEURAL_CORRECTOR.md) explains the design,
contains a separate diagram for every profile, and documents all input and loss
masks. The [training guide](../../docs/NEURAL_CORRECTOR_TRAINING.md) covers
recordings, synthetic generation, training and evaluation. The
[Corrector Lab guide](../../docs/CORRECTOR_LAB.md) describes checkpoint export,
static serving and browser tests.

The next transport is [Neural-QR](../../docs/NEURAL_QR.md), designed to supply
geometry, phase identity and normalized crops without a separately decoded dense
header. Its first model contract remains G64/four phases so the trained corrector
can be evaluated before deciding whether new-layout fine-tuning is needed.

## Model family

| Profile | Input forms | Internal design | Parameters |
| --- | --- | --- | --- |
| `U-tube` | Single RGB frame or fixed T-frame burst | Full-resolution U, 32 channels, two convs per block | 140,976 at T=1; 144,432 at T=5 |
| `U-tube-2x` | Single RGB frame or fixed T-frame burst | Same U width; residual blocks with four outer/eight bottleneck convs | 276,336 at T=1; 279,792 at T=5 |
| `U-tube-2x-attn` | Four-phase burst, 4–8 observations in eight slots | Residual U, learned global positional anchor, two window-attention blocks | 432,848 |

`U-tube` is the default profile. The trained four-phase experiment uses
`U-tube-2x-attn`. All profiles produce `[N,16,G,G]` logits and use four input pixels
per output cell. For G64, their U operates at **256→128→64→128→256**, followed by a
**256→128→64** outlet. Block outputs remain 32 channels. Skip concatenations
briefly have 64 channels; the head produces 16 logits. No raw RGB bypass reaches
the output.

Fixed-frame profiles accept T=1…64, G64/G128 and four or sixteen colors. Each burst
contains one logical matrix and one palette phase. They predict physical palette
labels, which the receiver maps back to logical symbols. The attention profile
requires **G64, sixteen colors and four permuted phases**, and predicts canonical
logical data symbols directly. Its spatial anchor and attention windows are
currently tied to G64.

The four-phase input uses slots **P0a, P0b, P1a, P1b, P2a, P2b, P3a, P3b**:
one or two captures per phase, with K uniformly drawn from 4…8 during training.
The first 24 planes contain RGB; eight additional binary planes identify present
slots. Missing RGB is gated inside the model. Four-frame inference means one
capture from **each** phase, not four consecutive exposures of one phase.

Every model clips RGB to `[-1,1]` before learned processing. Recording conversion
retains calibration overshoot with `clip(c/127.5−1,−1.5,1.5)`. Stored files remain
unchanged. The spatial loss mask excludes reference-grid targets while preserving
reference pixels as input evidence. An active-class restriction excludes unused
palette channels. See [masks and the objective](../../docs/NEURAL_CORRECTOR.md#masks-and-the-training-objective)
for the distinction between these masks and the shifted-attention boundary mask.

## Training and data

The recordings contain calibrated dense crops. Offline joins recover exact
sender-derived targets, and the rebuild workflow keeps receiver/sender sessions
and payloads in their assigned train/validation split. Real inputs can receive
mild photometric augmentation. Synthetics include curved moiré, capture-resolution
loss, rolling shutter, motion/RGB-offset blur, white balance/exposure changes,
shadows, reflections, glare and sensor noise.

Serialized recording, dataset and checkpoint format identifiers remain stable.
Renaming this component does not require editing existing capture JSON. Profile
names in checkpoint metadata are separate from filenames; the serving guide
explains the explicit naming update for already trained weights.

Phase-grouped synthetics use one coherent optical trajectory across all four
palette assignments. Training applies exact XOR phase-group permutations and
joint horizontal/vertical flips to inputs and targets; these augmentations do
not rewrite recordings and do not run during evaluation. The supervised loss is
ordinary data-cell cross-entropy. Paired sampling balances real/synthetic sources
and mapped real directions; additional sessions improve diversity rather than
changing those direction weights.

Training defaults to 32,000 updates, AdamW weight decay .01, LR `1e-4` fresh or
`5e-5` initialized, and validation every 1,000 updates. The live bar displays
training metrics; validation checkpoints print a separate consolidated summary.
Default JSON reports retain aggregate, source/group and per-K statistics without
per-session or per-burst records. Detailed diagnostics are opt-in; see the
training guide for the current switches and report schema.

The successful reported 32,000-update phase-grouped run achieved **529/560 =
94.46%** exact observation cases across five K presets of the same 112 logical
candidates. Phone→PC exact recovery was **70.8%, 77.1%, 87.5%, 100%, 100%** for
K4…K8; PC→phone reached 100% at each K. Validation CE was .0198 and cell error
.64%. These are recorded-validation results supplied by the user. Fresh-camera
browser performance remains the next measurement.

## Entry points

| File | Responsibility |
| --- | --- |
| `model.py`, `contract.py` | Network profiles, tensor shapes, normalization and numerical fixtures |
| `data.py`, `bursts.py`, `phases.py` | Dataset validation, distinct-capture grouping and canonical phase targets |
| `synthetic.py`, `augmentation.py` | Coherent corruptions and training-only transformations |
| `sampling.py`, `session_groups.py` | Source/direction balance and session assignments |
| `train.py` | Optimization, progress summaries, compact history and checkpoint selection |
| `evaluate.py` | Held-out cell metrics and canonical JavaScript payload recovery |
| `checkpoint.py` | Strict finite float32 parameter/metadata validation |
| `export.py` | ONNX graph export, parity fixtures and manifest |
| `browser-runtime/` | Pinned ONNX Runtime Web dependencies and Node WASM check |

These Python tools prepare and test artifacts; they are not an inference service
used during an authentication attempt. CPU execution is sufficient for export
and small checks. Full training is intended for the available RTX 4090. No
Hugging Face dependency or higher-level training framework is required.

## Setup, export and serving

From the repository root, inside your Python virtual environment:

```sh
python -m pip install -r experiments/neural-corrector/requirements.txt
python -m unittest discover -s experiments/neural-corrector/tests -v
```

For a trained checkpoint whose metadata uses its current profile name:

```sh
python experiments/neural-corrector/export.py \
  --checkpoint checkpoints/neural-corrector/U-tube-2x-attn/best.pt \
  --out experiments/neural-corrector/artifacts/selected
npm ci --prefix experiments/neural-corrector/browser-runtime
npm run prepare:corrector -- --artifacts experiments/neural-corrector/artifacts/selected
npm run test:corrector:wasm -- --artifacts experiments/neural-corrector/artifacts/selected
```

Export loads weights with `torch.load(weights_only=True)`, verifies exact finite
float32 shapes and split provenance, checks the ONNX graph, and normally tests two
PyTorch/native-ONNX inputs per exported size. It records package versions,
operators, tensor contracts, hashes and byte counts. Native runtime failure is
an export failure; `--skip-ort-check` is an explicit diagnostic exception that
marks the native check as not run, and still requires browser runtime validation.

The prepared browser assets are **ONNX, a manifest, numerical fixtures and the
pinned JavaScript/WASM runtime**. The `.pt` file stays on the training/export
machine. Its roughly 1.7 MB weight size does not determine the entire browser
bundle size; the attention ONNX graph is approximately 2.84 MB, and runtime
assets are additional. The static demo server serves these files normally;
inference runs locally inside the browser worker. Detailed commands and the
one-time profile-metadata naming step for an existing checkpoint are in the
[serving guide](../../docs/CORRECTOR_LAB.md).

Without `--checkpoint`, export produces **untrained** numerical smoke fixtures.
Those fixtures establish execution/parity only and cannot establish optical
recognition. Generated weights, crops, logs and runtime assets remain ignored by
Git. No trained default checkpoint ships with the repository.

## Runtime contract and evaluation

The ONNX input is `pixels`; the output is `logits`. Tensors are contiguous,
little-endian float32 in NCHW order. The manifest declares the model profile,
input layout, fixed capacity, allowed actual counts, output label space,
normalization bounds and artifact hashes. The runtime validates that contract
before inference; matching tensor shapes alone do not prove matching semantics.

| Layout | Input tensor | Output label space |
| --- | --- | --- |
| `stacked-rgb` | `[N,3T,4G,4G]` | Physical palette classes |
| `phase-paired-rgb-presence-planes` | `[N,32,256,256]` | Canonical logical data classes |

The output is always `[N,16,G,G]`. The score adapter applies softmax to active
classes, reverses a phase mapping only for physical-output models, and excludes
reference cells before byte recovery. Expected bytes are compared **after**
independent CRC/RS acceptance; they never help classification or repair.

Attention evaluation uses nested K4/K5/K6/K7/K8 subsets of the same fixed shuffled
phase pairs. The classical comparator sees the same actual captures, each mapped
using its own phase. Reports distinguish logical candidates from repeated
observation cases. Checkpoint selection averages direction metrics equally and,
for attention, averages the five capture-count results. This balanced selection
score can differ from the sample-weighted headline exact rate.

A batch-one attention input occupies **8 MiB** before runtime work buffers. It
uses the same eight-slot graph at K4 and K8; collection time can fall with fewer
observations while inference work stays approximately constant. No phone/WebGPU
latency or live goodput is inferred from offline recovery percentages.

The standalone Corrector Lab supports numerical parity and camera crop inference.
The live optical receiver still needs the separate integration that feeds learned
scores into its recovery and authentication flow. This boundary keeps model
verification, byte acceptance and cryptographic authentication explicit.
