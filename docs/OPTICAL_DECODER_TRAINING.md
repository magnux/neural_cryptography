# Optical decoder M2 — capture, train, and evaluate

The software for collecting examples, generating synthetic interference,
training the small U-Net, evaluating recovery, and exporting a checkpoint is
implemented. **Real-device improvement is still an acceptance test**, requiring
new capture sessions and a trained checkpoint. Start with G64/16 colors, using
G64/4 colors as a control. G128 remains supported for experiments, but the
submitted M1 reports only establish inference feasibility at G64.

The [design](OPTICAL_DECODER.md) and [roadmap](ROADMAP.md) describe how this feeds
the Neural ZKBoo color-authentication POC. M2 evaluates single captures. Learned
temporal fusion and authentication integration remain subsequent work.

## Collect real examples

Prepare and serve the existing color app, with trusted HTTPS for the phone:

```sh
npm run prepare:apps
python apps/color-optical-bench/serve.py --port 8000
```

Open **Capture Dataset** at `capture.html`, linked from Decoder Lab. Collection
does not need PyTorch, model files, or ONNX Runtime. Use the existing
[HTTPS serving instructions](DEVELOPMENT.md#local-serving) when using two devices.

1. On the sender, select **64×64**, **16 colors**, **4 phases**, and **Create
   random sequence**. Grid-8 references and RS32 are fixed for this collector.
   Save the sender manifest immediately; a new sequence replaces the old one.
2. Use **Focus matrix** to fit the complete marker into the screen. The display
   stays static. Close focus to change phase after the receiver finishes.
3. On the receiver, choose **Collect camera samples**, select the camera and
   **720p**, and press **Start camera**. Alignment scans start automatically at
   the selected interval (500 ms by default; 250/500/1000/2000 ms available).
   There is only one in-flight worker task; slow processing lowers the actual
   scan rate. The translucent green rectangle fills the detected matrix area.
   An unreadable header clears the fill. The overlay is never part of saved pixels.
   **Mirror preview** in the camera corner flips the video and alignment overlay
   together. It affects only the display; recorded pixels, detection, labels and
   optional originals retain their original orientation. The choice persists
   across camera restarts within the page.
4. The camera, scan interval, recording buttons, and live counters sit above
   the camera in one compact viewport. Set **Pool per phase** to **32** and press
   **Record**. Each new matrix/phase needs consecutive identical valid headers
   over at least 500 ms before saving. Missing headers reset the guard. Payload
   CRC failures remain useful examples.
5. The first 32 captures fill the pool. Recording then continues automatically:
   a randomized, sampled-distance heuristic compares each new capture to the
   current phase pool and may replace a redundant observation. The count remains
   32; **Seen** and **Replaced** show continuing activity. Small angle, distance,
   focus, or white-balance changes can provide new examples. No manual thinning
   action is required. The maximum choices are 8/16/32/64; reducing it below a
   retained pool requires saving and clearing first.
6. **Mean distance** summarizes the current phase pool. Watch it together with
   replacement count and the live image. A larger value is not automatically
   better; avoid collecting only damaged images. See the heuristic below.
7. Advance the sender manually when ready. Recording remains armed and starts
   a separate pool after the new phase passes the guard. **Pause** before
   **Save receiver captures**, then **Clear downloaded batch** before resuming.
   Clearing keeps the physical session ID. Download once per final batch: saving
   overlapping copies and joining both creates duplicate capture IDs.
8. Generate fresh sequences and save each sender manifest. More sessions should
   also contain new payloads, not only more correlated frames of a few messages.
   Keep each displayed phase static while collecting. The sender's physical-session
   button is separated from the matrix by extra spacing and a divider.
9. Record device/direction/brightness/conditions in the notes. Repeat with roles
   reversed when alignment permits, and collect the 4-color control. Start a
   **New physical session** when changing camera/device arrangement or returning
   later; rear and front cameras should have distinct session IDs. Keep entire
   sessions together in a split. Stopping the camera or hiding the page pauses
   recording and preserves samples; reopening does not restart the camera.

**Recording files are unchanged:** sender and receiver formats, versions, record
fields, tensor normalization, and offline label joining remain the same. Existing
recordings can be joined directly. Existing optional collection metadata keeps
its shape (`keepRandom` is now 0; `recentThinning` is empty). The selection method
value describes automatic distance replacement. No conversion is needed.

An initial collection can use two training sessions and one validation session
per direction, with two fresh payloads per session. Reserve a later session for
the final test, before selecting model settings. This is a starting budget,
not enough to establish generalization to other screen/camera pairs. Add data
where the held-out failures show a gap.

Each download is capped at **128 saved samples / 160 MiB** of encoded data
across all phase pools, with at most 100 diagnostics within the same byte budget.
The byte budget can stop a G128 batch before its requested sample counts; this
pauses collection and asks you to save/clear. Use a smaller per-phase maximum
on memory-constrained phones. Browser memory use is higher than export size,
especially during JSON download. Optional original PNGs are limited to two per
batch and 1 MiB each; oversized originals are omitted while tensor collection
continues. The receiver file contains calibrated pixels and may contain the
camera scene if originals are enabled. Store downloads in ignored `local-data/`.

### Interpreting diversity

The worker samples the same 1,024 stratified, deterministically jittered RGB
positions in each aligned, color-corrected dense crop. It computes the RMS RGB
difference for every pair of retained signatures, then averages those distances.
The displayed percentage is relative to the nominal 0–255 RGB range (normalization
is inverted by a factor of 127.5); calibrated overshoot can exceed that range.
It is an estimate from sampled pixels, not the full-tensor distance. Fewer than
two samples displays no metric. Signatures are temporary and are not exported
as training features. Only captures with the same complete frame identity and
palette phase are compared. Computation stays in the worker.

Once a phase pool is full, the worker draws up to eight reference samples and
up to eight candidate records at random. It estimates novelty by nearest-reference
RGB distance and replaces a crowded candidate only if novelty improves by more
than max(0.05 percentage points, 5% of the candidate's crowding distance). For
selection only, each RGB component's difference is capped at 25 percentage points
so a few extreme pixels cannot dominate. An isolation guard rejects incoming
samples farther than max(12 percentage points, three times the median reference
pair distance). These are initial tunable heuristic constants, not correctness
thresholds. Randomness chooses comparison subsets and resolves equal candidates.

Selection receives no labels or CRC outcome. It can miss duplicates outside the
sampled references, and neither global mean distance nor image quality is guaranteed
to increase. The displayed mean remains the ordinary, unclipped sampled RMS metric,
so it is distinct from the clipped selection score. A phase's count stays bounded;
replacement preserves the stable-header guard, byte budget, and snapshot limits.
Receiver exports retain the same distance-summary fields and recording schema.

For device checks, report camera/direction, grid/palette, interval, pool maximum,
seen/replaced counts, mean distance at first fill and later, and whether the
controls/counters and camera fit together on screen. Marker/header acquisition
failures remain separate: this iteration improves collection, not the detector.

The guard cannot prove the absence of a rolling-shutter display transition.
Keep the sender static, review questionable captures, and exclude mixed frames
offline. Unreadable headers are kept as diagnostics, optionally with an original
image, and receive no inherited labels. For the PC, resolve marker/header
acquisition before expecting this payload classifier to help.

## Join labels offline

The receiver never loads the sender manifest. Move both exports to the training
computer and join them with the canonical JavaScript codec:

```sh
node scripts/decoder-dataset.mjs join \
  --senders local-data/sender-a.json,local-data/sender-b.json \
  --captures local-data/receiver-batch-a.json,local-data/receiver-batch-b.json \
  --split train --out local-data/real-train
```

Use actual downloaded filenames. Repeat into separate empty directories for
`validation` and later `test`, with whole sessions and fresh payloads assigned
in advance. One directory can contain multiple sessions. Join validates complete
header metadata, phase, payload digest, and sender consistency. Unmatched
records remain excluded/unlabeled; conflicting manifests are rejected.

To remove a questionable sample, write a JSON array of its
`receiverSessionId/timestamp` capture IDs and pass `--exclude exclusions.json`.
These IDs appear in joined sample provenance. Rejoin into a new empty directory;
do not edit pixel/label binaries manually.

The directory contains a `manifest.json` and hashed binaries: normalized NCHW
float32 inputs, physical palette labels, data-cell masks, classical scores,
and expected chunk bytes. Labels are regenerated from the existing OBM5
encoder, including reference cells, parity, and padding. Python implements no
parallel optical wire format. Reference cells remain in the image but are
excluded from the classification loss. Expected bytes are used only after
independent decoding to measure whether accepted output is exact.

## Generate the synthetic pilot

Use the experiment's pinned dependencies in a virtual environment. Select an
appropriate PyTorch CUDA build for the 4090; CPU works for small smoke runs.
If you already generated the clean corpora below, reuse them: only the corrupted
synthetic train/validation/test corpora need regeneration to include the new
photometric effects. Existing real recordings and joined datasets remain usable;
their recording and dataset formats have not changed. Choose empty output
directories; generation refuses to overwrite an existing corpus. If you keep the
previous generated data for comparison locally, use new output paths here and
pass those same paths to training and evaluation.

```sh
python -m pip install -r experiments/optical-decoder/requirements.txt

node scripts/decoder-dataset.mjs generate --out local-data/clean-train \
  --count 2048 --size 64 --colors 16 --phases 4 --seed 100 --split train
node scripts/decoder-dataset.mjs generate --out local-data/clean-validation \
  --count 256 --size 64 --colors 16 --phases 4 --seed 10000 --split validation
node scripts/decoder-dataset.mjs generate --out local-data/clean-test \
  --count 256 --size 64 --colors 16 --phases 4 --seed 20000 --split test

python experiments/optical-decoder/synthetic.py --clean local-data/clean-train \
  --out local-data/synthetic-train --frames 7 --severity .65 --seed 7
python experiments/optical-decoder/synthetic.py --clean local-data/clean-validation \
  --out local-data/synthetic-validation --frames 7 --severity .65 --seed 8
python experiments/optical-decoder/synthetic.py --clean local-data/clean-test \
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
enabled by default, so the existing commands above also generate them:

| Option | Default | Effect |
| --- | --- | --- |
| `--white-balance-stops` | `.6` | Channel-dependent RGB gains, in 80% of bursts; the maximum absolute log2 gain is scaled by severity. |
| `--exposure-stops` | `1.5` | Signed exposure gain, in 80% of bursts, including both underexposure and overexposure; the maximum absolute log2 gain is scaled by severity. |
| `--shade-opacity` | `.45` | Maximum alpha for a translucent black or white rectangle, scaled by severity. |
| `--shade-probability` | `.7` | Fraction of bursts receiving a rectangle; its area is sampled between 20% and 30% of the crop. |

At `--severity .65`, the respective maximum strengths are ±0.39 stops of
white-balance gain, ±0.975 stops of exposure gain, and 0.2925 rectangle opacity.
Zero strength disables the corresponding added effect; use
`--shade-probability 0` to disable rectangles. `--severity 0` returns the clean
crop unchanged. The generator records its algorithm and settings in synthetic
provenance so different corruption runs can be distinguished.

The white-balance and exposure offsets are shared across each burst with mild
correlated drift. A rectangle retains its color and approximate placement,
with softened edges and slight motion. These are semi-transparent blends, not
opaque cutouts, and neither their selection nor their parameters use payload
symbols or labels. The mixture includes milder captures as well as difficult
ones; saturation can still erase color information and cannot always be undone.

Disturbances evolve across a burst rather than drawing unrelated images. This
is a **calibrated-crop approximation** of residual color distortion; it does
not simulate the full camera pipeline, recalibration, or header failures.
More varied synthetics are not evidence of improved device recovery: held-out
real sessions determine whether the added effects reduce the observed gap.
Synthetic classical baselines are recomputed from the same corrupted crop;
real baselines use the independently captured current-exposure samples.

## Train and evaluate

Both recipes use plain PyTorch/AdamW, float32, microbatches of 4 and accumulation
of 4 (effective batch 16). The maximum budget defaults to **32,000 updates**.
Use `--steps` to override it. Microbatches share a grid size and palette; reference
cells and inactive palette classes are excluded from the data-cell objective.

| Recipe | Default selection | Learning rate | Objective | Validation |
| --- | --- | ---: | --- | --- |
| `pilot` | Fresh runs, or initialized runs without real training data | 1e-4 fresh; 5e-5 initialized | Ordinary per-frame mean CE | 128 captures every 200 updates |
| `adapt` | Initialized runs containing real training captures | 1e-5 | 90% mean CE + 10% hardest-quarter CE | All captures every 1,000 updates |

Use `--recipe pilot` explicitly for the plain-CE baseline. The adaptation recipe
requires `--initialize`, real training data, and an explicit `--session-groups`
sidecar covering every real receiver session in training and validation. Both
splits must represent the same real groups. The recommended combined run is
[below](#recommended-next-training-run); an existing trained wide pilot can be reused.

```sh
python experiments/optical-decoder/train.py \
  --recipe pilot --train local-data/synthetic-train \
  --validation local-data/synthetic-validation \
  --model wide --device cuda --out checkpoints/optical-decoder/wide-pilot
```

For a quick CPU pipeline check, use `--device cpu --steps 10 --batch-size 1
--accumulate 1 --eval-every 5 --validation-samples 4`, with a small corpus and an
empty run directory. A short smoke run need not recover a payload.

Validation always uses unaugmented inputs and ordinary data-cell CE, even when
the training objective includes hard cells. Exact recovery uses the current JS
palette mapping, reliability ranking, RS and CRC; accepted bytes are compared
with expected bytes afterward. An accepted incorrect payload stops training for
investigation. Expected payload bytes never enter model inference or repair.

The trainer evaluates at step zero. An initialized model is eligible for
`best.pt` immediately, so an unsuccessful adaptation cannot overwrite its starting
performance with a worse model. A fresh pilot's step-zero random weights are
only diagnostic, not a trained checkpoint. With group metadata, checkpoint
selection uses the mean exact-recovery rate across real groups, then their mean
CE as the tie-break. Without real group metadata it uses the global metrics.

Pilot validation uses a fixed session/profile-balanced subset by default.
`--validation-samples 0` evaluates the entire validation corpus; it is the
adaptation default. Capped evaluation must still include every mapped real group.
Standalone `evaluate.py --max-samples 0` also evaluates all examples. Reports
include per-session and per-group CE/error/recovery, confidence calibration,
phase identities, raw/hard/erasure recovery, wrong accepted frames, and inference
timing. Both methods receive one exposure per candidate; these are not temporal
throughput measurements.

Whole-session and payload split checks, including checkpoint ancestry, remain
mandatory. Keep original sender/receiver session IDs intact when moving or
joining files. Use untouched test sessions only after selecting the model and
receiver settings on validation.

## Model profiles and the latest training result

| Profile | Parameters | Changes |
| --- | ---: | --- |
| `compact` | 121,792 | Existing plain U-Net baseline, including your trained checkpoints |
| `wide` | 487,104 | Doubled feature widths; per-location channel LayerNorm on feature convolutions; unnormalized pooled RGB readout added to context logits |

Both take `[N,3,4G,4G]` calibrated pixels and output `[N,16,G,G]` physical-color
logits. LayerNorm does not normalize raw RGB or depend on batch statistics. The
separate color readout preserves absolute color evidence. The latest combined
model/data iteration improves submitted validation recovery; it does not isolate
the architecture's contribution. Wider browser inference
needs fresh device latency measurements; fixed G64/G128 export and WASM parity
are checked separately from recovery quality.

Use `--model compact` for a fresh baseline. `--initialize` automatically retains
the checkpoint's architecture; it does not widen existing weights. An explicit
conflicting `--model` fails. To test wide, train a new pilot from your existing
datasets, then fine-tune that pilot. Both are active profiles in the same current
checkpoint representation; weights are validated exactly, never partially loaded.

The 2026-10-31 log contains 1,536 joined real training captures and 768 real
validation captures, with no exclusions. The 12 training and six validation
capture bundles are correlated collections, not 2,304 independent device setups.
The 1,200/600 diagnostic entries are bounded acquisition event records, not counts
of rejected training samples. Synthetic counts are consistent: 2,048/256/256
payloads × four phases × nine frames = 73,728/9,216/9,216 examples.

| Run / checkpoint | Step | Exact frames | Data-cell error | Validation CE |
| --- | ---: | ---: | ---: | ---: |
| Synthetic selected | 16,000 | 125/128 (97.66%) | 0.455% | 0.01347 |
| Synthetic final | 20,000 | 121/128 (94.53%) | 0.529% | 0.01484 |
| Real selected | 5,200 | 7/128 (5.47%) | 40.50% | 1.14159 |
| Real final | 32,000 | 1/128 (0.78%) | 46.75% | 2.20347 |

Classical recovery is 68/128 (53.125%) on the synthetic subset and zero on the
real subset. These are fixed 128-capture validation measurements, not full-corpus
or untouched test results. `wide-real/best.pt` correctly retains step 5,200.
The large later increase in real validation loss and deterioration in accuracy
support overfitting in addition to a synthetic/real distribution gap. The new
architecture, augmentations, burst length, and real collection changed together;
their separate contributions cannot be established from these runs.

The source mixture is already explicit: `--real-fraction .5` gives approximately
half the draws to real data when both sources share a profile, despite only about
2% of stored training examples being real. In that submitted run, effective batch
16 and 32k updates drew each real crop about 167 times on average (about 27 by
step 5,200), without online real augmentation. Its `trainLoss` was the latest
update's mixed-source mean, not an averaged real-only training loss. The current
adaptation recipe adds augmentation and interval-averaged source/group metrics.

Review found no obvious physical-color/phase, mask, or loss aggregation bug.
Actual alignment and label quality still require inspecting capture tensors.
Synthetic crops clip to nominal RGB while real affine-calibrated inputs can
overshoot to [-1.5,1.5]. Camera processing and interference may differ in other
ways. The log does not identify which mismatch matters most.

The subsequent full evaluation of that same selected checkpoint covers all 768
real validation captures: **69 exact recoveries (8.98%)**, 39.32% cell error and
CE 1.10503; classical recovery is zero. All 69 accepted outputs are correct and
require ordinary hard-decision RS correction. There are no raw CRC passes or
additional erasure-assisted recoveries. The higher rate than 7/128 reflects a
larger evaluation of the same model, not another model improvement.

Matching receiver sessions to the supplied capture paths gives:

| Direction | Exact recovery | Cell error |
| --- | ---: | ---: |
| PC screen → phone camera | 69/512 (13.48%) | 24.76% |
| Phone screen → PC camera | 0/256 | 68.43% |

All successes come from two of six sessions (46 and 23 frames). The four other
sessions recover nothing. Each session contributes 128 captures with 32 per phase.
Class frequencies are nearly balanced (6.11–6.34% each). Aggregate expected
calibration error is 3.11 percentage points; the highest-confidence quarter of
cells is 99.15% correct at 97.90% average confidence. Group-specific calibration
may differ, but these measurements prioritize acquisition-domain generalization
and overfitting over global confidence adjustment or class-frequency weighting.

## Recommended next training run

Run the implemented **combined adaptation recipe once**, keeping the wide
architecture and the trained `wide-pilot/best.pt`. Reuse the current real and
synthetic datasets: neither recording again, regenerating synthetics, nor
retraining the pilot is required for this experiment.

First create a local direction sidecar. The example below follows the supplied
folder organization: `local-data/phone/` contains captures recorded **on the phone**
(PC→phone), and `local-data/pc/` contains captures recorded **on the PC** (phone→PC).
Verify that this describes the files being passed. Flags assign direction;
filenames, camera names and UUIDs are never used to infer it.

```sh
python experiments/optical-decoder/session_groups.py \
  --pc-to-phone local-data/phone/*decoder-captures*.json \
  --phone-to-pc local-data/pc/*decoder-captures*.json \
  --datasets local-data/real-train local-data/real-validation \
  --out local-data/session-groups.json

python experiments/optical-decoder/train.py \
  --recipe adapt \
  --train local-data/real-train local-data/synthetic-train \
  --validation local-data/real-validation \
  --initialize checkpoints/optical-decoder/wide-pilot/best.pt \
  --session-groups local-data/session-groups.json \
  --device cuda --out checkpoints/optical-decoder/wide-adapt
```

The sidecar command reads actual receiver session IDs from current capture
exports, rejects conflicting assignments, and verifies coverage against the
joined datasets before writing. Extra mappings are allowed; missing real-session
mappings fail. It refuses to overwrite an existing sidecar. Choose an empty run
directory too. No recording, tensor, dataset or checkpoint format migration is
needed; session mappings stay in a separate file.

| Setting | Adaptation default |
| --- | --- |
| Source allocation | 50% synthetic / 50% real, controlled by `--real-fraction` |
| Real allocation | Equal quotas per mapped direction, then session → payload → phase → frame |
| Objective | `0.9 × mean CE + 0.1 × mean CE of the hardest 25% of valid cells`, computed per frame |
| Online real-only augmentation | Exposure ±0.15 stops, per-channel WB ±0.1 stops, RGB noise standard deviation 0.005 |
| Optimizer | AdamW, LR 1e-5, weight decay 0.05; gradient norm capped at 5 |
| Validation | All validation captures every 1,000 updates, plus the step-zero baseline |
| LR control | Halve after four checks without meaningful selection improvement; floor 2e-6 |
| Stopping | Stop after twelve checks without meaningful selection improvement; maximum 32k updates |

Source/direction quotas apply to the **whole optimizer update**, not independently
to its microbatches. With one common profile, two directions, and the default
batch/accumulation settings, each update contains eight synthetic, four PC→phone,
and four phone→PC captures. For other batch sizes, fractional source counts use
seeded stochastic rounding and group remainders are randomized. The sampler
avoids repeating bursts within an update while distinct bursts are available,
and avoids repeating exact samples until the relevant pool is exhausted. It
indexes once rather than scanning the corpus on every update. Every represented
real grid/palette profile must contain all real groups, preventing a profile from
silently dropping the difficult direction.

Quotas apply within the profile selected for that update. If additional 4-color
or G128 profiles contain only synthetic examples, those updates are entirely
synthetic and the overall real fraction falls below 50%. Check the recorded
`trainSources` draw counts for the realized mixture; the eight/four/four allocation
above describes the current shared G64/16 profile, not every heterogeneous corpus.

Augmentation is mild and online: labels/masks and geometry are unchanged,
synthetic samples pass through unchanged, and measured real moiré is retained.
The final tensor keeps the calibrated `[-1.5,1.5]` contract without first clipping
to nominal RGB. The noise standard deviation is specified in RGB 0–1 units
(0.01 in normalized units). No additional strong moiré, opaque shading or spatial
smoothing is added. The sampler and augmentation use separate seeded generators.

Checkpoint saving and plateau patience have distinct purposes. `best.pt` accepts
any strict improvement in direction-balanced exact recovery, then lower ordinary
CE. Patience resets on a higher balanced exact rate, or, when that rate ties, a
**relative CE decrease greater than 0.001 (0.1%)** from the meaningful-progress
anchor. This prevents tiny CE fluctuations from extending training indefinitely.
LR reductions do not reset the stopping count. With the defaults, four checks
mean 4,000 updates and twelve mean 12,000 updates. Use `--lr-patience 0` or
`--stop-patience 0` to disable the respective control; the explicit `pilot` recipe
disables both by default and keeps plain CE/zero online augmentation.

The run writes `settings.json`, `history.json`, `baseline-validation.json`,
`final-validation.json`, selected `validation.json` and `best.pt`. History records
interval-averaged objective, separate real/synthetic CE and draw counts,
per-direction CE and counts, repeated-burst/sample counts, validation groups,
selection metrics and LR changes. These training values are measured on the
actual training inputs, including real augmentation, rather than being directly
comparable to unaugmented validation loss. The exact group mapping is copied to
the local run/checkpoint and hashed; public exports expose only the validated
scalar/hash provenance, not session identities or local paths.

Compare the existing real checkpoint and the new adaptation checkpoint on the
**same full validation corpus and grouping**:

```sh
python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/optical-decoder/wide-real/best.pt \
  --session-groups local-data/session-groups.json \
  --max-samples 0 --device cuda \
  --out local-results/optical-decoder/wide-real-grouped-validation.json

python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/optical-decoder/wide-adapt/best.pt \
  --session-groups local-data/session-groups.json \
  --max-samples 0 --device cuda \
  --out local-results/optical-decoder/wide-adapt-validation.json
```

Send both evaluation reports and the adaptation run's `history.json`. The main
question is whether phone→PC improves while PC→phone is preserved or improves;
a gain confined to the two already successful sessions leaves the main failure
unresolved. Check that reported source/group draw counts match the intended
allocation and whether stopping selected the starting model or a later step.
Validation remains model-selection data, with correlated captures; after choosing
the model, evaluate untouched test sessions separately. Add their receiver IDs
to a new sidecar using their capture exports if group coverage is needed, without
moving them into training or validation.

The combined recipe deliberately leaves focal loss, label smoothing, Brier loss,
temperature fitting and class-frequency weighting out of this run. Same-phase
consistency and RS-aware auxiliary objectives remain deferred, along with a
comparison of temporal fusion under equal observation budgets. The present
experiment tests the practical combined improvement rather than running a long
ablation sequence. Relevant background includes
[regularization under group shifts](https://arxiv.org/abs/1911.08731) and
[confidence calibration](https://arxiv.org/abs/1706.04599); those publications do
not establish a recovery gain for this optical decoder.

## Export the selected checkpoint

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/wide-adapt/best.pt \
  --out experiments/optical-decoder/artifacts/selected
npm ci --prefix experiments/optical-decoder/browser-runtime
npm run prepare:decoder -- --artifacts experiments/optical-decoder/artifacts/selected
npm run test:decoder:wasm -- --artifacts experiments/optical-decoder/artifacts/selected
```

Export produces both G64 and G128 graphs using the selected weights, with fresh
PyTorch numerical fixtures. The current `optical-decoder` manifest records
`trained` status, checkpoint hash, and bounded public training provenance.
It omits local paths and session/payload lists; full split provenance remains
inside the local checkpoint. Export without `--checkpoint` still creates an
untrained feasibility bundle. Only the current format is accepted: regenerate
exports and run preparation after updating this code.

Use [Decoder Lab](DECODER_LAB.md) to repeat fixture parity and camera-active
timings on both devices. A restricted host may need the explicit
`--skip-ort-check` option, which records native parity as not run; Web/WASM and
device parity must still be checked. Tests and exports do not deploy a site.

**M2 acceptance:** held-out real exact recovery or recovery time improves over
the classical baseline at equal observation budgets, without regressing the
4-color control. Report failures as well as successes. Then connect the trained
scores to temporal recovery and the color authentication flow; do not wait for
G256 or a particular QR throughput target.
