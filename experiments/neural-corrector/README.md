# Neural corrector

> The current collector records **Neural-QR v2, 4×4 pixels/cell,
> G16/G32/G64/G128**. Collection, [offline imports and paired synthetic generation](../../docs/W_TUBE_DATA.md)
> and **W-tube training/evaluation** are available. W-tube export and browser
> integration remain pending. The separate
> `multires` profile described below consumes existing prepared **2×2-cell**
> datasets and cannot train on the new recordings. Keep them in a separate
> directory; see [collection instructions](../../docs/MULTIRES_COLLECTION.md).

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

The next integrated transport is [Neural-QR](../../docs/NEURAL_QR.md), designed to supply
geometry, phase identity and normalized crops without a separately decoded dense
header. **W-tube** processes its current four-pixel inputs through two complete
U stages, with independent low-frequency anchors at G16/G32/G64/G128. The
separate `multires` architecture shares a trunk across six sizes, with independent
per-size low-frequency anchors and six alternating convolution/attention
blocks, with two layers in each block. The
[current collector](../../docs/MULTIRES_COLLECTION.md) records G16…G128 at 4×4
pixels per cell for W-tube, with a public collection code for offline association.
The earlier prepared 2×2-cell datasets remain usable for the existing `multires`
experiment. Browser integration of W-tube is a later step; the U-tube experiments
retain their independent four-pixels-per-axis input contract.

## Model family

| Profile | Input forms | Internal design | Parameters |
| --- | --- | --- | --- |
| `W-tube` | Four-phase burst, 4–8 observations in eight slots; G16/G32/G64/G128 | Two full-resolution U stages, 32 channels, two convolutions + two attention layers per bottleneck, independent cosine anchors | 454,992 processing + 32,768 anchors = **487,760** |
| `multires` | Four-phase burst, 4–8 observations in eight slots; G8…G256 | One-level 2G→G→2G U; six alternating conv/attention blocks; independent cosine anchors | 231,120 shared + 43,008 anchor coefficients = 274,128 |
| `U-tube` | Single RGB frame or fixed T-frame burst | Full-resolution U, 32 channels, two convs per block | 140,976 at T=1; 144,432 at T=5 |
| `U-tube-2x` | Single RGB frame or fixed T-frame burst | Same U width; residual blocks with four outer/eight bottleneck convs | 276,336 at T=1; 279,792 at T=5 |
| `U-tube-2x-attn` | Four-phase burst, 4–8 observations in eight slots | Residual U, learned global positional anchor, two window-attention blocks | 432,848 |

`U-tube` is the default profile. The trained four-phase experiment uses
`U-tube-2x-attn`. Select `--model multires --frames 8` only for prepared 2×2-cell Neural-QR datasets.
Select **`--model W-tube --frames 8`** for the new version-2 four-pixel corpus.
All profiles produce `[N,16,G,G]` logits. W-tube and the three U-tube profiles use four input
pixels per axis per output cell; `multires` uses two. For G64, the U-tube path operates at **256→128→64→128→256**, followed by a
**256→128→64** outlet. Block outputs remain 32 channels. Skip concatenations
briefly have 64 channels; the head produces 16 logits. No raw RGB bypass reaches
the output.

Fixed-frame profiles accept T=1…64, G64/G128 and four or sixteen colors. Each burst
contains one logical matrix and one palette phase. They predict physical palette
labels, which the receiver maps back to logical symbols. The attention profile
requires **G64, sixteen colors and four permuted phases**, and predicts canonical
logical data symbols directly. Its spatial anchor and attention windows are
currently tied to G64.

W-tube has two independently parameterized **4G→2G→G→2G→4G** stages. Each
bottleneck contains two convolutions and two 8×8-window attention layers, with
shifts zero and four. Each U exit fuses its full-resolution input and reconstructed
features through residual addition, channel normalization and ReLU. A single
selected anchor is added after the inlet; its 16×16 cosine coefficients reconstruct
a G×G field and are enlarged fourfold. The final outlet performs **4G→2G→G**
reduction before the sixteen-class head. See [the W-tube diagrams](../../docs/NEURAL_CORRECTOR.md#w-tube).

`multires` reconstructs a separate G×G positional field from at most 16×16
learned cosine coefficients per channel, then enlarges that field 2× at the
inlet. Only the anchor is frequency-limited; image features and output logits
retain full spatial detail. Its bottleneck is C→A→C→A→C→A, where C contains two
convolutions and A contains two window-attention layers.

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
sender-derived targets, keeping receiver/sender sessions and payloads in their
assigned train/validation split. The [W-tube rebuild](../../docs/W_TUBE_DATA.md)
combines native version-2 recordings, bicubic-resampled version-1 Neural-QR
crops, and eligible original G64 payload crops. It regenerates clean targets and
harder paired synthetics at a **4:1 synthetic/real capture ratio**, rounded up to
complete four-phase bursts. The W-tube training mix remains **50/50**; disk
counts do not set the sampling ratio. Existing model training can apply
mild photometric augmentation. Synthetics include curved moiré, capture-resolution
loss, rolling shutter, motion/RGB-offset blur, white balance/exposure changes,
shadows, reflections, glare and sensor noise.

W-tube and multires training use one resolution per update, choose resolutions uniformly,
and balance sources, recorded device directions and physical sessions. Their
validation measures exact **logical data matrices** and masked cell error, with
equal weight across resolutions, directions and K4…K8. There are no packet bytes
or CRC recovery claims in the raw-symbol Neural-QR dataset.

For W-tube, native real validation is the primary population whenever available;
auxiliary origins have separate summaries. A corpus without native validation is
explicitly reported as auxiliary-only. A separate native/auxiliary sampling quota
within the real training half remains future work. Start W-tube from scratch;
continuation via `--initialize` requires a W-tube checkpoint and a new run folder.
The [W-tube training guide](../../docs/NEURAL_CORRECTOR_TRAINING.md#train-w-tube)
provides complete training, continuation and evaluation commands.

Current collection files use `neural-qr-sender` / `neural-qr-captures`, and their
joined datasets use `neural-qr-dataset`, version 2. Explicit offline conversion
produces that same contract with origin tags; the live collector accepts only the
current recording format. Existing prepared OBM and 2×2-cell datasets retain
their separate experiment contracts. Checkpoint
formats for the implemented U-tube models remain unchanged. Profile
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
Per-grid rows separate real training, synthetic training and real validation.
`CER (cell)` pools data cells; `CER(size)` averages per-grid rates equally.
Default JSON reports retain aggregate, source-by-grid/group and per-K statistics without
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
| `evaluate.py` | Held-out cell metrics; exact Neural-QR matrices or independent OBM payload recovery |
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

For the existing U-tube browser demo, using a checkpoint whose metadata uses its current profile name:

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
machine. That U-tube checkpoint's roughly 1.7 MB weight size does not determine the entire
browser bundle size; the attention ONNX graph is approximately 2.84 MB, and runtime
assets are additional. The static demo server serves these files normally;
inference runs locally inside the browser worker. Detailed commands and the
one-time profile-metadata naming step for an existing checkpoint are in the
[serving guide](../../docs/CORRECTOR_LAB.md).

For multires, `export.py --checkpoint PATH --sizes 64 --out PATH` exports a
static selected-size graph with only its own reconstructed G×G anchor, baked
into an export-only model copy. Training keeps independent coefficient tensors;
no FFT operator is required by the browser. A trained checkpoint may
export only sizes actually included in training. Current multires checkpoints
record low-frequency anchor and six-block bottleneck metadata; prior dense-anchor
multires runs belong to their source revision and are not automatically loaded.
This does not yet update the existing browser worker or packet transport to the new input/format contract.

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
| `phase-paired-rgb-presence-planes` (U-tube-2x-attn) | `[N,32,256,256]` | Canonical logical data classes |
| `phase-paired-rgb-presence-planes` (multires) | `[N,32,2G,2G]` | Canonical logical data classes |

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
The Neural-ZKBoo demo now feeds learned G64 scores into packet recovery and has
completed a physical-device exchange with QR ACKs. The next integration uses
Neural-QR and `multires`; model verification, byte acceptance and cryptographic
authentication remain distinct checks.
