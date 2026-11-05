# Neural corrector — collect, train, evaluate, and export

The current **multires** training path consumes header-free Neural-QR recordings
at G8/G16/G32/G64/G128/G256 with **2×2 pixels per cell**, sixteen colors, and
four palette phases. One checkpoint shares its convolution/attention trunk
across sizes and learns a separate anchor for each grid. Its input has eight
capture slots; every training example uses four to eight observations, with
one or two observations from each phase. Its output is a corrected logical
matrix of sixteen-class logits. See [architecture and masks](NEURAL_CORRECTOR.md).

The trained `U-tube-2x-attn` model and its existing 4×4-cell datasets remain a
separate experiment. Their commands are retained [below](#independent-u-tube-experiment-prepared-44-cell-datasets).
The new acquisition matrices carry raw symbols: multires **exact matrix** metrics
are not the earlier OBM5 CRC/RS recovery metrics, and are not an authentication
or goodput measurement.

## Collect and organize real examples

Follow [Multires collection](MULTIRES_COLLECTION.md) for the device workflow.
The server stages the sender automatically; the receiver loads the active
public session configuration and uploads its own captures when finished.
Sender truth is joined only offline and never enters receiver inference.
Already recorded current Neural-QR files remain valid, including the smaller
capture pools used before the increased collection budget.

Uploaded files are organized as:

```text
local-data/recordings/
  train/g64/sender/*.json
  train/g64/receiver/*.json
  validation/g64/sender/*.json
  validation/g64/receiver/*.json
```

The other grid sizes follow the same structure. Keep every physical recording
session and logical target entirely within one split. Record both directions
and distinct lighting conditions in training and validation. Device/camera
metadata is retained automatically; the training sampler derives direction
labels from that metadata, with an optional explicit `--session-groups` override.

The collector retains a stable core, a controlled shoulder and a small tail of
samples. Its sampled RGB distance measures variation, not correctness; there
is no requirement for the distance to increase. See [balanced retention](MULTIRES_COLLECTION.md#admission-and-balanced-retention).
Each target must have **at least two captures of every phase** to support all
K=4…8 training and validation inputs. Current 8-, 16- and 32-capture pools all
meet this requirement when complete.

## Rebuild paired multires datasets

Install the training dependencies in the Python environment you intend to use:

```sh
python -m pip install -r experiments/neural-corrector/requirements.txt
```

Inspect the counts and disk estimate first, then generate:

```sh
node scripts/rebuild-corrector-data.mjs --plan
node scripts/rebuild-corrector-data.mjs --python python
```

Defaults are `--recordings-root local-data/recordings` and
`--out local-data/multires`. The resulting stable tree is:

```text
local-data/multires/
  real/train/g8/manifest.json
  real/validation/g8/manifest.json
  clean/train/g8/manifest.json
  clean/validation/g8/manifest.json
  synthetic/train/g8/manifest.json
  synthetic/validation/g8/manifest.json
  build.json
```

Every recorded grid gets the same structure, and a `test` split is included if
present. Each split/grid has its own manifest. The real files preserve calibrated
camera evidence; the clean files contain four exact palette renders per target;
the synthetic files corrupt those same targets. Target identity, both physical
session identities, device pair, split and camera metadata remain attached.
The join verifies sender hashes, phase profile and split ownership. This cannot
prove that an operator aimed at the correctly assigned matrix.

The generation recipe defaults to:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--bursts-per-target` | `8` | Independent corruption trajectories for each real target. |
| `--frames` | `8` | Captures per phase of each trajectory; **not** model input slots. |
| phases | `4` | All phases share a coherent camera/environment trajectory. |
| `--severity-min` / `--severity` | `.35` / `.8` | Random severity within this interval per trajectory. |
| `--max-output-mib` | `1048576` | Conservative aggregate disk budget; `0` explicitly removes the budget check. |

This makes **8 × 8 × 4 = 256 synthetic captures per target**. The existing
realistic noise, curved moiré, exposure/white balance, reflection/glare,
rolling shutter, motion/RGB-offset blur and capture-resolution effects are
retained. The resolution stage never drops below one camera pixel per logical
cell. Synthetic classical baselines are recomputed from the corrupted crop;
clean labels never influence them.

Generation is an offline disk workload. At G256, 256 float32 crops plus their
classical score arrays take roughly **1.8 GiB per target**, before filesystem
and metadata overhead. `--plan` includes all recorded sizes and splits in its
estimate. The generator streams captures instead of holding a complete corpus
in RAM. Rebuild stages the replacement and only swaps it into place after all
joins and generation succeed; allow enough disk for the old and new dataset
simultaneously. It removes the previous **managed** dataset after the swap.
It refuses to replace an unmarked directory or one containing unrelated files,
and never deletes recordings or checkpoints. Existing `multires-joined` or
older independently prepared dataset directories are not touched.

Repeat the same rebuild command after collecting new sessions. Training should
be stopped before replacing datasets it is reading. Do not change a recorded
split merely to improve the validation score; reserve genuinely new sessions
for final testing.

### Individual pipeline commands

The wrapper is optional. These commands perform the same operations for one grid:

```sh
node scripts/corrector-dataset.mjs join-uploads \
  --recordings-root local-data/recordings --out local-data/multires-joined
node scripts/corrector-dataset.mjs targets \
  --dataset local-data/multires-joined/train/g64 \
  --out local-data/multires-clean-g64
python experiments/neural-corrector/synthetic.py \
  --clean local-data/multires-clean-g64 --out local-data/multires-synthetic-g64 \
  --phase-grouped --bursts-per-target 8 --frames 8 \
  --severity-min .35 --severity .8 --capture-min-scale .5 --seed 41
```

Each output must be new or empty. For downloaded current Neural-QR recordings,
use `corrector-dataset.mjs join --senders FILE[,FILE] --captures FILE[,FILE]
--split train --out NEW_DIRECTORY` before `targets`. Optional exclusions are a
JSON array of `receiverSessionId/timestamp` capture IDs passed with `--exclude`.
Unmatched captures remain unlabeled; conflicting assignments are rejected.

## Train multires

Start a fresh shared model for the first multires run:

```sh
python experiments/neural-corrector/train.py \
  --model multires --frames 8 \
  --train local-data/multires/real/train local-data/multires/synthetic/train \
  --validation local-data/multires/real/validation \
  --paired-targets --real-fraction .5 \
  --batch-size 2 --accumulate 8 --lr .0001 --steps 32000 \
  --validation-samples 0 --device cuda \
  --out checkpoints/neural-corrector/multires-phase8
```

Parent directories are scanned recursively for manifests. Each optimizer update
uses one grid size selected uniformly among available training sizes, with
homogeneous-size microbatches. Within that grid, source/group/session/target
sampling prevents a recording with more captures from dominating. Paired mode
uses equal numbers of real and synthetic examples of the **same logical target**
and the same K observation budget. Dataset counts do not determine this ratio:
`--real-fraction .5` keeps training **50/50**. Loss first averages over data cells
within each example and then over examples, so a larger matrix does not receive
a larger gradient simply because it has more cells.

`--batch-size 2 --accumulate 8` is a conservative starting point for the largest
512×512 input on the 4090, giving sixteen examples per optimizer update. Measure
actual VRAM before increasing the microbatch. The samples per phase in a stored
burst can exceed eight; the model randomly draws only the four to eight captures
needed by that example. Sampling is not restricted to consecutive frames.
Training retains spatial flips, phase-pair permutations and small real-only
photometric augmentation with aligned targets/masks.

The command defaults to 32,000 total optimizer updates even if `--steps` is
omitted. With all six sizes, each receives about one sixth of those updates;
this is not 32,000 updates per size. All updates train the shared trunk. The live bar
shows training CE/CER and source metrics. Validation prints separately, with
per-grid, per-direction and per-K summaries. Lean reports contain aggregate
metrics, not per-session or per-burst arrays. Checkpoint selection uses the
reported size-balanced exact-matrix score, with CE as a tiebreak; inspect the
per-size values so success at G8 does not conceal failure at G256.

To continue a multires checkpoint in a **new output directory**, add:

```sh
--initialize checkpoints/neural-corrector/multires-phase8/best.pt --lr .00005
```

Initialization loads weights with a fresh optimizer; it is not an exact resume
of AdamW state. An explicit `--model multires --initialize
checkpoints/neural-corrector/u-tube-2x-attn-phase8/best.pt` can also warm-start
matching shared layers and the G64 anchor, leaving new anchors/layers fresh.
The changed 2×2-cell input geometry still needs training, so fresh initialization
is the recommended first run. Checkpoints record which grid sizes were actually
optimized; exporting an untrained size is rejected.

## Evaluate multires

```sh
python experiments/neural-corrector/evaluate.py \
  --dataset local-data/multires/real/validation --split validation \
  --checkpoint checkpoints/neural-corrector/multires-phase8/best.pt \
  --max-samples 0 --device cuda \
  --out local-results/neural-corrector/multires-validation.json
```

Every K=4…8 evaluation preset uses seeded, fixed random draws from held-out
bursts, not consecutive frames. Every phase is represented, with at most two
captures per phase. Results include CE, data-cell error rate and exact logical
matrix rate, grouped by grid, direction and K. Fixed reference cells are visible
to the model but excluded from loss/error scoring. Exact matrix means every
scored data cell is correct, **before any packet framing or error correction**.
The same score cannot be compared directly with the old CRC/RS-assisted recovery
rate. Use `--split test` with `local-data/multires/real/test` for a final new-session
evaluation; train and checkpoint-selection sessions are rejected there.

The deployment path exports a selected trained size from the shared checkpoint
and includes only that size's anchor. This task prepares training/evaluation;
new Neural-QR packet/authentication integration remains a separate step. See
[model deployment design](NEURAL_CORRECTOR.md) for that boundary.

## Independent U-tube experiment: prepared 4×4-cell datasets

Use the experiment's pinned dependencies in a virtual environment. Select an
appropriate PyTorch CUDA build for the 4090; CPU works for small smoke runs.
**For a U-tube-2x-attn run on prepared 4×4-cell data, skip this optional pilot generation
and use the [target-paired run below](#training-another-run).** It
starts fresh combined-model weights using the current paired synthetics. Existing
U-tube and U-tube-2x checkpoints remain available for their own evaluation or continuation.

For a fresh pilot, reuse existing clean corpora or generate them below. Generate
corrupted corpora into empty directories. Existing prepared OBM5 datasets
remain usable without conversion; the new Neural-QR recordings are a separate
input contract. The following independent random-payload route
also remains available for broader pretraining:

```sh
python -m pip install -r experiments/neural-corrector/requirements.txt

node scripts/corrector-dataset.mjs generate --out local-data/clean-train \
  --count 2048 --size 64 --colors 16 --phases 4 --seed 100 --split train
node scripts/corrector-dataset.mjs generate --out local-data/clean-validation \
  --count 256 --size 64 --colors 16 --phases 4 --seed 10000 --split validation
node scripts/corrector-dataset.mjs generate --out local-data/clean-test \
  --count 256 --size 64 --colors 16 --phases 4 --seed 20000 --split test

python experiments/neural-corrector/synthetic.py --clean local-data/clean-train \
  --out local-data/synthetic-train --frames 7 --severity .65 --seed 7
python experiments/neural-corrector/synthetic.py --clean local-data/clean-validation \
  --out local-data/synthetic-validation --frames 7 --severity .65 --seed 8
python experiments/neural-corrector/synthetic.py --clean local-data/clean-test \
  --out local-data/synthetic-test --frames 7 --severity .8 --seed 9
```

`--count` counts fresh payloads; phases and burst frames multiply the resulting
samples. These commands produce 57,344 corrupted training samples and 7,168 each
for validation/test. They are offline disk workloads, not browser buffers.
Generation accepts up to 2**20 payloads; corpus manifests allow 256 MiB and have
no fixed sample-count ceiling. Synthetic output defaults to an estimated 2**20
MiB budget (1 TiB); `--max-output-mib 0` explicitly removes that disk-budget check.
The 256 MiB JSON bound still applies per manifest; use several corpus directories
for larger collections. Python and JavaScript use the same offline manifest ceiling.
Browser capture memory limits are separate and unchanged. Training partitions
source pools once, avoiding a full-corpus scan on every microbatch. Keep all outputs of one payload in the same split. Choose disjoint seed
ranges for clean corpora; changing a folder name is not a split. Add separate
4-color corpora with fresh seeds to both training and validation; the training
CLI accepts multiple directories. Use the same pattern for G128 when ready.

The generator applies an RGB stripe carrier before blur/resampling, curved and
chirped channel-dependent moiré, bounded subcell jitter, short motion blur,
color/gamma changes, clipping, and shot/read noise. The following effects are
enabled by default:

| Option | Default | Effect |
| --- | --- | --- |
| `--white-balance-stops` | `.6` | Residual channel-dependent RGB gains in 80% of bursts. |
| `--exposure-stops` | `1.5` | Signed exposure gain in 80% of bursts, with highlight/shadow clipping. |
| `--shade-opacity` | `.45` | Maximum strength of soft, rotated rectangle/ellipse shadows and reflections. |
| `--shade-probability` | `.7` | Probability of a burst receiving one broad shape, sometimes with a smaller second shape. |
| `--glare-probability` | `.65` | Probability of a drifting elliptical highlight with a broader veiling halo. |
| `--glare-strength` | `.6` | Maximum glare strength before severity scaling; zero disables it. |
| `--capture-probability` | `.75` | Probability of a burst receiving the reduced-capture-resolution stage. |
| `--capture-min-scale` | `.25` | Minimum native capture side as a fraction of the stored crop side. |
| `--capture-max-scale` | `1` | Maximum native capture side as a fraction of the stored crop side. |
| `--capture-blur-sigma` | `.35` | Maximum Gaussian blur sigma in native capture pixels; selected bursts draw between half and all of this value. |
| `--severity-min` | omitted | If set, draw one severity per independent burst, uniformly from this minimum to `--severity`; otherwise use fixed severity. |
| `--rolling-shutter-stops` | `.65` | Maximum row-dependent exposure-band strength, with mild channel variation. |
| `--rolling-shutter-probability` | `.7` | Probability of correlated drifting rolling-shutter bands/skew. |
| `--rolling-shutter-skew` | `.5` | Maximum row-dependent horizontal displacement in dense-crop pixels, before severity scaling. |
| `--bursts-per-target` | `1` | Independently seeded corruption trajectories for each clean target. |
| `--motion-probability` | `.7` | Fraction of bursts with integrated exposure motion. |
| `--linear-blur-pixels` | `2.5` | Full arbitrary-direction exposure travel in dense-crop pixels, before severity scaling. |
| `--rotation-blur-degrees` | `.35` | Full rotational sweep around a varying center, before severity scaling. |
| `--zoom-blur-fraction` | `.006` | Full radial scale change during an exposure, before severity scaling. |
| `--motion-samples` | `7` | Temporal samples averaged per simulated exposure. |
| `--rgb-offset-probability` | `.6` | Probability of channel-offset exposure windows, conditional on motion being active. |
| `--rgb-offset-pixels` | `.5` | Maximum channel displacement along the exposure trajectory in dense-crop pixels, before severity scaling. |

Photometric and motion strengths are scaled by `--severity`; at `.65`, a `.65`-stop rolling-shutter
limit becomes `.4225` stops. Zero strength disables its effect; set rolling-shutter
probability to zero to disable both bands and skew, or shade probability to zero
to disable overlays. Capture resolution and its blur are separate geometric
controls, not severity-scaled. `--severity 0` returns the clean crop unchanged,
including bypassing the capture-resolution stage.

Rolling-shutter effects approximate exposure bands from a rolling readout and
screen flicker: broad/narrow bands evolve over successive frames, with slight
tilt and color variation. A small row-dependent skew is bounded together with
registration jitter to remain below half a four-pixel cell. Exposure blur can
spread neighboring-cell evidence while keeping the target grid centered.
This does not simulate a transition between different displayed payloads; those
mixed-label captures should still be excluded.

Shadows multiply the image by a spatially varying, slightly tinted attenuation;
reflections add light with highlight clipping. Shapes vary in angle, aspect,
position, edge softness and opacity gradient, with correlated movement across a
burst. The main shape's bounding rectangle covers roughly 18–32% of the crop;
ellipses and edge clipping reduce the affected area. Neither corruption choice
nor its parameters depend on payload symbols. Multiple independently seeded
bursts provide different conditions for the same target, while frames within a
burst remain correlated.

Glare adds a localized bright core and a wider, slightly tinted scattered-light
halo. The halo reduces local contrast; the core can saturate highlights. Its
position and intensity evolve smoothly through a burst. Glare has an independent
random stream, so changing its controls does not redraw unrelated nuisance
parameters. A severity range preserves moderate examples alongside hard ones;
the recipe records its bounds and each capture records its burst's actual value.

Reduced capture resolution approximates a small phone display occupying fewer
pixels in the PC camera, even though rectification produces a fixed-size crop.
After the other image effects and sensor noise, selected bursts use area
downsampling, a small Gaussian blur at that native resolution, and bilinear
upsampling to the original crop size. One native side and blur sigma are drawn
per burst and kept fixed across its exposures. The native side is uniform over
the integer pixel sizes in the configured scale range; an independent random
stream leaves existing artifact draws unchanged. The recipe and each burst's
capture settings are recorded in synthetic provenance.

At the default `.25–1` scale range, G64 crops go from 256×256 to 64–256 pixels
per side and back; G128 crops go from 512×512 to 128–512 and back. The lower
endpoint is one native pixel per matrix cell. That is an intentionally difficult
case, not a guarantee of recoverable symbols. Upsampling does not restore lost
spatial evidence. This stage preserves model input size, labels and recording
formats, and does not simulate marker/header detection or full camera geometry.
Set `--capture-probability 0` to disable the entire stage; setting only blur
sigma to zero still permits downsampling.

Motion blur now integrates several centered exposure samples with arbitrary-
direction linear travel, rotation around an off-center pivot and radial zoom.
Strength/direction evolve
coherently within a burst; zero controls disable the respective effects. This
models image evidence spread, not a changed class label or integer-cell shift.

RGB-offset blur shifts each channel's exposure window along the same linear,
rotational and radial trajectory. Centered channel offsets keep the mean window
aligned, and a dense-crop pixel bound limits the displacement. The sampled offsets
remain correlated within a burst; their independent random stream leaves other
artifact draws unchanged. This approximates motion-related color misregistration
and fringing, without assuming that every RGB camera exposes its channels at
different times. It acts only when motion is active; setting either new control
to zero disables it, and severity zero remains clean. Labels and masks do not move.

These are **calibrated-crop approximations**, not a full camera/display/readout
simulation. They do not reproduce detection, recalibration, or header failures.
The algorithm/settings and independent burst identity are recorded in synthetic
provenance. Synthetic classical baselines are recomputed from each corrupted
crop; real baselines use independently captured current-exposure samples.
Held-out real recovery must establish whether the additional effects help.

## Train and evaluate

There is one training path: plain PyTorch/AdamW, float32, ordinary masked
cross-entropy, microbatches of 4 and accumulation of 4 (effective batch 16).
Training defaults to **32,000 updates**, learning rate `1e-4` for fresh weights or
`5e-5` for an initialized checkpoint, and weight decay `.01`. Full validation
runs every 1,000 updates. `--steps`, `--lr`, `--eval-every` and
`--validation-samples` remain explicit controls. There is no adaptation recipe,
hard-cell objective or plateau scheduler.

Real training stacks receive mild online photometric augmentation, after random
stack assembly and before the model's existing [-1,1] input clip:

| Option | Default | Units and behavior |
| --- | ---: | --- |
| `--real-noise-std` | `.008` | Gaussian noise standard deviation in RGB [0,1] units, about two camera-byte levels; independent across exposures/pixels. |
| `--real-tint-stops` | `.06` | Small per-channel log2 gains, shared across a stack. |
| `--real-exposure-stops` | `.08` | Small global log2 gain, shared across a stack. |

Set all three to zero to disable this augmentation. Its dedicated random stream
does not change sampler draws. It does not modify synthetic inputs, stored real
captures, labels, masks, validation or inference. Calibration overshoot is retained
until the model's final input clip. Settings and provenance record the controls;
real training CE now measures augmented inputs, so an initial rise relative to an
unaugmented run is not by itself a regression. Compare unchanged validation.

For a fresh synthetic pilot:

```sh
python experiments/neural-corrector/train.py \
  --train local-data/synthetic-train \
  --validation local-data/synthetic-validation \
  --frames 5 --device cuda --out checkpoints/neural-corrector/tube-pilot
```

Microbatches share a grid size and palette. Reference cells and inactive palette
classes are excluded from the data-cell objective. Standard source sampling
remains available; `--paired-targets` enables the matched-real/synthetic route
[below](#training-another-run). An optional `--session-groups` sidecar
balances real acquisition directions and adds direction-specific evaluation.

For a quick CPU pipeline check, use `--device cpu --steps 10 --batch-size 1
--accumulate 1 --eval-every 5 --validation-samples 4` without paired sampling.
Paired sampling requires an even effective batch of at least two. Short smoke
runs validate wiring and need not recover payloads.

Exact recovery uses the current JS palette mapping, reliability ranking, RS and
CRC; accepted bytes are compared with expected bytes afterward. An accepted
incorrect payload stops training for investigation. Expected payload bytes never
enter model inference or repair.

The trainer evaluates at step zero. An initialized model is eligible for
`best.pt` immediately, preserving its starting performance if training never
improves it. A fresh model's random step-zero weights are diagnostic only.
With explicit real groups, fixed-frame checkpoint selection uses mean exact
recovery across those groups, then mean CE as the tie-break; otherwise it uses
global metrics. U-tube-2x-attn averages the per-count selection metrics across
K=4/5/6/7/8, balancing real groups within each K when the sidecar is supplied.
Read the per-direction results too: mean recovery alone can improve while a
still-failing direction stays at zero.

`--validation-samples 0` evaluates all validation captures and is the default.
Capped evaluation uses a fixed session/profile-balanced subset and must include
every mapped real group. Standalone `evaluate.py --max-samples 0` also evaluates
all examples. Default reports keep aggregate and per-source/per-group CE,
cell error, exact/CRC recovery counts, confidence-calibration error and p50/p95
inference timing. They omit per-session and per-burst/capture rows. Attention
reports keep K=4/5/6/7/8 summaries without embedding a second full diagnostic
report for every K. See [report contents](#metrics-and-report-files).

Multiframe reports count complete input groups, not individual exposures.
`inputCaptures` records consumed observations; `burstGrouping` records unused
captures and incomplete groups. Fixed-frame models make disjoint T-frame groups
within each phase pool. The phase-grouped model pairs observations within each
phase, combines four phase pairs of one logical matrix, and evaluates nested
K=4…8 subsets. Six complete logical groups with 32 captures per phase yield
96 candidates per K. Do not compare this denominator directly with a single-frame
report. Both classical and neural evaluation receive the same actual observations;
expected bytes are checked only after independent RS/CRC acceptance. Offline
recovery rates do not measure live throughput.

The evaluation draw is a fixed seed-7 permutation within each phase burst.
Fixed-frame profiles group at their fixed T; U-tube-2x-attn makes pairs in each
phase and combines four pairs from the same logical matrix. It evaluates nested
4/5/6/7/8 subsets, starting with one capture from every phase. It is independent of timestamps, directory paths
and input-file order. Existing checkpoints can be reevaluated without retraining;
use a new report filename because previous chronological reports used different
input groups. `selection` and `burstGrouping` report the method and seed. Training
settings/history also record the policy.

Whole-session and whole-payload split checks, including checkpoint ancestry,
remain mandatory. Keep sender/receiver identities intact when generating paired
targets. Use untouched test sessions only after selecting the model and receiver
settings on validation.

## Current profiles and reported result

| Profile | Input observations | Parameters | Purpose |
| --- | --- | ---: | --- |
| `U-tube` (default) | Fixed T=1…64 | 144,432 at T=5 | Full-resolution normalized U-Net with a 32-channel tube |
| `U-tube-2x` | Fixed T=1…64 | 279,792 at T=5 | Deeper residual resolution blocks |
| `U-tube-2x-attn` | K=4…8, eight phase-paired slots | 432,848 | Residual tube, learned global anchor and window attention |

The output has 16 logits per target cell for every profile. The score adapter
normalizes the label convention: fixed-input models predict physical colors,
while the phase-grouped model predicts canonical logical symbols. See the
[separate architecture diagrams and masks](NEURAL_CORRECTOR.md) for the full
network layout, convolution counts and input semantics.

The user-reported completed phase-grouped run selected its final **32,000**th
update after approximately **10 h 29 min**. Its validation CE was **.0198** and
cell error **.64%**. The final training interval had CE **.0109**, split into
**.0200 real** and **.0018 synthetic**. Those training inputs include online
augmentation; validation inputs do not.

| Captures K | Overall exact recovery | Phone→PC | PC→phone |
| --- | ---: | ---: | ---: |
| 4 | 98/112 (87.50%) | 34/48 (70.83%) | 64/64 (100%) |
| 5 | 101/112 (90.18%) | 37/48 (77.08%) | 64/64 (100%) |
| 6 | 106/112 (94.64%) | 42/48 (87.50%) | 64/64 (100%) |
| 7 | 112/112 (100%) | 48/48 (100%) | 64/64 (100%) |
| 8 | 112/112 (100%) | 48/48 (100%) | 64/64 (100%) |

There are **112 shared candidates**, evaluated at five observation counts:
**529/560 = 94.46%** is the raw aggregate recovery rate. It is not 560 independent
matrices. The checkpoint-selection score is **93.54%**, because it gives equal
weight to both directions within each K and then equal weight to the five K
values. The figures differ by weighting, not by a recovery discrepancy. The
per-direction counts above are inferred from the logged denominators and rounded
rates; the full run artifacts were not supplied with that result.

The phone→PC target has been exceeded on this recorded validation set. Retain
untouched test sessions for the selected checkpoint and test both physical
transmission directions in the browser before claiming live performance.

## Metrics and report files

The live tqdm bar contains only **training** statistics for the current interval:
overall CE/CER, real then synthetic CE/CER, mapped groups in sorted order, and
learning rate. It does not repeat the last validation result. At each validation
step, a stable multiline summary prints:

- **Train:** consolidated interval metrics in the same order as the live bar.
- **Val:** CE/CER, raw exact count/rate, the separately labelled selection score,
  and K4→K8 recovery where applicable.
- **Val groups:** per-group CE/CER/recovery in sorted name order.

`--show-goal` optionally adds the phone→PC K4→K8/50% goal line at validation
steps. It is off by default. Training CER comes from the same masked forward
pass as training CE; it does not run CRC recovery on each training minibatch.
Both metrics exclude reference cells and inactive palette classes consistently
with the objective.

| File | Default contents |
| --- | --- |
| `settings.json` | Run configuration, model/input convention, sampling and augmentation settings |
| `history.json` | One compact row per validation step: consolidated training metrics, aggregate/group/source validation summaries and per-K summaries |
| `baseline-validation.json` | Compact step-zero evaluation |
| `validation.json` | Compact evaluation of the selected checkpoint |
| `final-validation.json` | Compact final-step evaluation |
| Standalone evaluation `--out` | The same compact report structure |
| `best.pt` | Selected weights and required training/split provenance |

Default evaluation JSON has `reportDetail: "summary"`. It excludes session identities,
per-candidate recovery rows, capture/burst IDs, detailed confusion matrices,
calibration-bin arrays and per-sample timings. Group/direction statistics,
source statistics, exact/CRC counts, K summaries, selection score and grouping
counts remain. Summary size scales with the number of groups/presets and
validation steps, rather than repeating every recorded candidate in history.
No metric is rounded before checkpoint selection.

When diagnosing a specific failure, explicitly request a separate detailed
report. The normal output stays compact:

```sh
python experiments/neural-corrector/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/neural-corrector/U-tube-2x-attn/best.pt \
  --session-groups local-data/session-groups.json \
  --device cuda --max-samples 0 \
  --out local-results/neural-corrector/validation.json \
  --detailed-report local-results/neural-corrector/validation-details.json
```

The two output paths must differ. Detailed diagnostics can still be large and
are not embedded in training history. Existing reports are not rewritten.

## Generate the updated paired synthetic corpus

For existing prepared OBM5 datasets, generate target-paired synthetics as follows;
reuse `target-clean-train`, or reconstruct it once with the first command below.
Generate the phase-grouped corpus into an empty `target-synthetic-train`
directory. This keeps seed 41 and the resolution/glare/RGB-offset recipe, while
correlating slow optical conditions across each four-phase group. Existing joined
real data remain unchanged.

```sh
# Reuse target-clean-train if already generated; otherwise run:
node scripts/corrector-dataset.mjs targets \
  --dataset local-data/real-train \
  --out local-data/target-clean-train

python experiments/neural-corrector/synthetic.py \
  --clean local-data/target-clean-train \
  --out local-data/target-synthetic-train \
  --phase-grouped --bursts-per-target 32 --frames 12 --seed 41 \
  --severity-min .45 --severity .9 \
  --glare-probability .7 --glare-strength .65 \
  --white-balance-stops .8 --exposure-stops 1.75 \
  --shade-opacity .55 --rolling-shutter-stops .8 \
  --capture-probability .75 --capture-min-scale .25 --capture-max-scale 1 \
  --capture-blur-sigma .35 \
  --rgb-offset-probability .6 --rgb-offset-pixels .5
```

`targets` reconstructs canonical clean matrices from the sender-derived bytes
and frame metadata already in the joined dataset. It verifies regenerated
headers, physical labels and masks against the source, and makes one clean
example per receiver session/payload/frame/phase, regardless of how many real
captures show it. **No noisy camera pixels are treated as clean truth.** The
sender/receiver identities, whole-payload identity and split are preserved;
recording files remain unchanged. Multiple joined source directories can be
passed as a comma-separated `--dataset` list. This command supports established OBM5
single logical frames with palette phases; it rejects incomplete multipart
payloads rather than inventing missing target bytes.

Thirty-two coherent phase groups with twelve captures per phase provide **384
synthetic observations per clean phase target**. A logical target has four such
phase targets. Slow optical conditions are shared across the four phases, while
noise, motion, moiré, rolling shutter and exposure drift change over the capture
timeline. `--phase-grouped` requires all four phases, G64/16 colors and consistent
logical data targets before writing outputs. This operates on established
prepared datasets and does not convert Neural-QR recordings.
Severity varies from .45 to .9 per burst. Glare, white balance, exposure, shade,
rolling shutter and resolution settings match the previous paired corpus; RGB
exposure offsets add the new effect. This retains moderate cases
rather than making every example maximally degraded. The resolution stage then
applies to 75% of bursts, independently of severity: G64 uses a native side of
64–256 pixels and blur sigma .175–.35 native pixels. Other bursts skip that stage.
The mixture includes full-resolution evidence; it does not assume that all
one-pixel-per-cell captures can be decoded. The generator reports the
actual target/sample count and estimated disk budget before writing; G64 RGB
tensors alone occupy 288 MiB per target at these settings. Channel offsets apply
to 60% of motion-active bursts, not 60% of all bursts, with a .5-pixel maximum
before severity scaling. The separate RNG keeps the other seeded draws intact.

Generate training targets **only from `real-train`**. Do not add validation or
test target matrices to training, including through a pilot checkpoint. Synthetic
derivatives of a real target remain in that target's split; changing a seed or
folder name does not make them independent held-out data.

The matched examples remove target-content differences between the real and
synthetic parts of an update. They do not by themselves make camera artifacts
identical or prove domain alignment; the new corruptions and held-out real
results test that hypothesis. The clean references are offline supervision,
not an additional model input or a browser decoding oracle.

## Training another run

Use the **already prepared 4×4-cell OBM5 datasets** for this recipe. If you need
fresh synthetic partners for those datasets, use the target-paired commands
above, writing into empty output directories. The current Neural-QR join cannot
feed this training command until production multires support is implemented.

For a fresh U-tube experiment, train `U-tube-2x-attn` from scratch with eight
phase-paired slots and randomly selected 4–8 observations at `1e-4` for 32,000
updates. Use existing `real-train`, `target-synthetic-train`, `real-validation`
and their direction sidecar. Each training group must have at least two distinct
captures of every phase, although each draw uses only one or two. All examples
must be G64/16 colors with four permuted phases and the established 4×4-cell
input contract. Independently generated old synthetic phase bursts require
regeneration with the paired group recipe.

Use **batch 16 × accumulation 2 = effective batch 32**. Batch 8 × 4 retains that
effective batch if activation memory is limiting. The new output directory is
`U-tube-2x-attn-run`. This directory must be new. Cross-profile initialization
is rejected; a continuation of the current phase-grouped model may use its
validated `--initialize` checkpoint and an explicit learning rate. The current
successful run does not need retraining merely to export it for the browser.

```sh
python experiments/neural-corrector/train.py \
  --train local-data/real-train local-data/target-synthetic-train \
  --validation local-data/real-validation \
  --model U-tube-2x-attn --frames 8 \
  --session-groups local-data/session-groups.json \
  --paired-targets --real-fraction .5 \
  --batch-size 16 --accumulate 2 \
  --real-noise-std .008 --real-tint-stops .06 --real-exposure-stops .08 \
  --phase-flip-probability .25 \
  --lr .0001 --steps 32000 --eval-every 1000 --validation-samples 0 \
  --device cuda --out checkpoints/neural-corrector/U-tube-2x-attn-run
```

Each input uses fixed slots **P0a, P0b, P1a, P1b, P2a, P2b, P3a, P3b**.
K is uniform over 4/5/6/7/8. Every phase contributes one capture; K−4 randomly
selected phases contribute a second, and placement within each pair is random.
The remaining slots contain zero RGB and zero presence indicators. Real and
synthetic target partners receive the same K; members never cross physical
sessions, logical frames or independent synthetic phase groups. Targets are
canonical phase-zero data symbols, with unchanged physical reference labels.
Real photometric augmentation occurs before padding and never affects presence.

Two training-only augmentations apply to both real and synthetic groups:

- **Exact phase permutations**, enabled by default: choose d from `[0,15,7,8]`,
  reorder whole phase pairs and their presence indicators, then XOR only data
  labels by d. For example, d=15 swaps phases 0↔1 and 2↔3. RGB values are not
  recolored. `--no-phase-permutations` disables this augmentation.
- **Joint spatial flips:** independently flip horizontally/vertically with
  `--phase-flip-probability .25` each. Apply the same flip to all RGB slots,
  labels and masks; zero disables both. This provides classification examples,
  not new canonical wire-format messages. Neither augmentation is applied to
  validation or stored files. Recovery on transformed predictions would first
  have to reverse the transformation.

With effective batch 32 and two mapped directions, each update contains eight
real PC→phone examples, eight real phone→PC examples and sixteen matched
synthetic partners. Extra sessions add diversity without changing this balance.
Synthetic `--frames 12` means captures per phase burst; training `--frames 8`
means capacity, with 4–8 actual captures per input. Four-frame inference requires
**one from each phase**, not the first four camera exposures regardless of phase.

The run writes the compact files described under
[metrics and report files](#metrics-and-report-files). `byFrameCount` holds
K4→K8 metric summaries, `uniqueCandidates` counts shared logical groups, and
`observationCases` counts their repeated evaluations across K. Selection weights
K equally and, with the direction sidecar, real directions equally; CE breaks
selection ties. It never selects whichever K happens to look best.

History retains training CE/CER, source/group metrics, `trainFrameCounts` and
compact `validationByFrameCount` values. The optional goal summary is small;
per-session or per-candidate diagnostics require the standalone detailed-report
option. Inspect K4 and K8 separately alongside the per-direction summaries.

### Evaluate all five observation counts

```sh
python experiments/neural-corrector/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/neural-corrector/U-tube-2x-attn-run/best.pt \
  --session-groups local-data/session-groups.json \
  --max-samples 0 --device cuda \
  --out local-results/neural-corrector/U-tube-2x-attn-validation.json
```

The evaluator shuffles each phase pool once with fixed seed 7, makes disjoint
pairs, and joins pairs from the four phases of the same logical matrix. It uses
one member from each pair at K=4, then adds second members in a fixed shuffled
phase order for K=5/6/7/8. Every K therefore tests the same candidate targets
with nested evidence. Classical scores reverse each member's own phase mapping;
neural predictions are already canonical. Both receive the same actual captures,
and padding never counts as evidence. Remainders and incomplete groups are
reported, never duplicated into a candidate.

For six complete logical groups with 32 captures in each of four phases, there
are **96 candidates per K**. The five per-count evaluations reuse candidates;
their sum is not an independent matrix count. Compare per-direction/per-count
rates and equal observation budgets instead of old single-phase success totals.
Fixed-frame U-tube/U-tube-2x evaluation retains its existing grouping.

For this fresh run, step-zero `baseline-validation.json` is diagnostic random
weights, with no parent checkpoint. Share `validation.json`, `history.json` and
the standalone validation report. Keep untouched test sessions for the selected
model/settings. The completed run has exceeded the phone→PC target on recorded
validation; assess generalization on new sessions and confirm zero wrong accepted
payloads. Check the fixed-profile 4-color control separately before live integration.

### Fixed five-frame residual example

The earlier command remains supported, with its own output directory:

```sh
python experiments/neural-corrector/train.py \
  --train local-data/real-train local-data/target-synthetic-train \
  --validation local-data/real-validation \
  --model U-tube-2x --frames 5 \
  --session-groups local-data/session-groups.json \
  --paired-targets --real-fraction .5 --batch-size 8 --accumulate 4 \
  --lr .0001 --steps 32000 --eval-every 1000 --validation-samples 0 \
  --device cuda --out checkpoints/neural-corrector/U-tube-2x5
```

Use this only when intentionally training the fixed-input residual profile,
with a new output directory. Its fixed-five-frame grouping is different from
the phase-grouped model; compare rates at a stated observation budget.

## Export the selected checkpoint for the demo

The [Corrector Lab guide](CORRECTOR_LAB.md#export-the-trained-checkpoint) is the
canonical serving recipe. It explains the one-time descriptive profile rename
for existing checkpoints, CPU ONNX export, prepared static assets, trusted HTTPS,
backend selection, caching, and device checks. The browser serves ONNX weights
and graph data; the `.pt` checkpoint stays in the training workspace.

The implementation already provides export, numerical fixtures, asset
preparation and worker inference. The next demo change connects corrected
logical scores to live RS/CRC recovery and displays accepted content, then
reuses that channel for Neural ZKBoo authentication. The offline validation
result is the reason to proceed to those tests, not evidence that real-time
recovery or optical throughput has already been measured.
