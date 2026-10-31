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
**If you already have `wide-pilot/best.pt`, skip this optional pilot generation
and use the [target-paired run below](#recommended-next-training-run).** Its
weights remain a useful initialization; a fresh pilot is not required after
changing corruptions.

For a fresh pilot, reuse existing clean corpora or generate them below. Generate
corrupted corpora into empty directories. Real recordings and joined datasets
remain usable without conversion. The following independent random-payload route
also remains available for broader pretraining:

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
enabled by default:

| Option | Default | Effect |
| --- | --- | --- |
| `--white-balance-stops` | `.6` | Residual channel-dependent RGB gains in 80% of bursts. |
| `--exposure-stops` | `1.5` | Signed exposure gain in 80% of bursts, with highlight/shadow clipping. |
| `--shade-opacity` | `.45` | Maximum strength of soft, rotated rectangle/ellipse shadows and reflections. |
| `--shade-probability` | `.7` | Probability of a burst receiving one broad shape, sometimes with a smaller second shape. |
| `--rolling-shutter-stops` | `.65` | Maximum row-dependent exposure-band strength, with mild channel variation. |
| `--rolling-shutter-probability` | `.7` | Probability of correlated drifting rolling-shutter bands/skew. |
| `--rolling-shutter-skew` | `.5` | Maximum row-dependent horizontal displacement in dense-crop pixels, before severity scaling. |
| `--bursts-per-target` | `1` | Independently seeded corruption trajectories for each clean target. |

Strengths are scaled by `--severity`; at `.65`, a `.65`-stop rolling-shutter
limit becomes `.4225` stops. Zero strength disables its effect; set rolling-shutter
probability to zero to disable both bands and skew, or shade probability to zero
to disable overlays. `--severity 0` returns the clean crop unchanged.

Rolling-shutter effects approximate exposure bands from a rolling readout and
screen flicker: broad/narrow bands evolve over successive frames, with slight
tilt and color variation. A small row-dependent skew is bounded together with
registration jitter and motion blur to remain below half a four-pixel cell.
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
`--validation-samples` remain explicit controls. The unsuccessful adaptation
recipe, its hard-cell loss, online augmentation and plateau scheduler have been
removed; there is no `--recipe` option.

For a fresh synthetic pilot:

```sh
python experiments/optical-decoder/train.py \
  --train local-data/synthetic-train \
  --validation local-data/synthetic-validation \
  --model wide --device cuda --out checkpoints/optical-decoder/wide-pilot
```

Microbatches share a grid size and palette. Reference cells and inactive palette
classes are excluded from the data-cell objective. Standard source sampling
remains available; `--paired-targets` enables the matched-real/synthetic route
[below](#recommended-next-training-run). An optional `--session-groups` sidecar
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
With explicit real groups, checkpoint selection uses mean exact recovery across
those groups, then mean CE as the tie-break; otherwise it uses global metrics.
Read the per-direction results too: mean recovery alone can improve while a
still-failing direction stays at zero.

`--validation-samples 0` evaluates all validation captures and is the default.
Capped evaluation uses a fixed session/profile-balanced subset and must include
every mapped real group. Standalone `evaluate.py --max-samples 0` also evaluates
all examples. Reports include per-session/group CE, error and recovery,
confidence calibration, phase identities, raw/hard/erasure recovery, wrong
accepted frames, and inference timing. These are single-exposure measurements,
not temporal throughput measurements.

Whole-session and whole-payload split checks, including checkpoint ancestry,
remain mandatory. Keep sender/receiver identities intact when generating paired
targets. Use untouched test sessions only after selecting the model and receiver
settings on validation.

## Model profiles and the latest training result

| Profile | Parameters | Structure |
| --- | ---: | --- |
| `compact` | 121,792 | Plain U-Net baseline |
| `wide` | 487,104 | Doubled widths, feature-channel LayerNorm, and an unnormalized pooled-RGB readout alongside context logits |

Both take `[N,3,4G,4G]` calibrated pixels and output `[N,16,G,G]` physical-color
logits. LayerNorm does not normalize raw RGB or depend on batch statistics.
`--initialize` keeps the checkpoint's exact architecture; it does not widen
weights. An explicitly conflicting `--model` fails. The model and browser tensor
contract are unchanged by paired training.

The latest submitted evaluation compares the same 768 captures from six physical
sessions. The discarded adaptation recipe did not beat the previous checkpoint:

| Validation metric | `wide-real/best.pt` | Discarded adaptation run |
| --- | ---: | ---: |
| Exact recovered frames | 69/768 (8.98%) | 62/768 (8.07%) |
| Overall cell error | 39.32% | 39.96% |
| PC → phone cell error | 24.76% | 25.13% |
| Phone → PC exact recovery | 0/256 | 0/256 |

All 62 later successes were already successful before; neither run recovered any
phone→PC frame. Keep `wide-real/best.pt` as the comparison baseline and
`wide-pilot/best.pt` as the recommended initialization. The recordings are
correlated captures, not 768 independent physical setups. Historical validation
measurements are recorded in [VALIDATION.md](VALIDATION.md); the current next
experiment is matched target content plus improved corruption generation.

## Recommended next training run

**Reuse `wide-pilot/best.pt` and train on real captures paired with synthetics of
the exact same target matrices.** Starting from scratch is possible by omitting
`--initialize`, but is not required: the architecture and labels are unchanged,
and the pilot already learned useful color/spatial features. Keep the earlier
`wide-real/best.pt` for a direct comparison rather than initializing from the
failed adaptation run.

### Remove obsolete generated outputs

Run from the repository root, after retaining any reports you still want. These
commands delete the failed run and old *corrupted* synthetic corpora; they do not
touch recordings, joined real data, clean corpora or the two useful checkpoints.
Do not run cleanup while training/generation uses those directories. If you need
the old synthetic test corpus for a pending comparison, leave that directory out.

```sh
# Explicit old output paths only; no broad checkpoint/data wildcards.
rm -rf -- checkpoints/optical-decoder/wide-adapt
rm -rf -- local-data/synthetic-train local-data/synthetic-validation local-data/synthetic-test
```

Preserve `checkpoints/optical-decoder/wide-pilot/`,
`checkpoints/optical-decoder/wide-real/`, `local-data/real-*`,
`local-data/clean-*`, and the original sender/receiver JSON files. The new run
uses different, empty output paths below. Check any additional old run's contents
before deleting it; directory names alone do not establish that its weights are
replaceable.

### Reconstruct clean targets and generate paired corruptions

```sh
node scripts/decoder-dataset.mjs targets \
  --dataset local-data/real-train \
  --out local-data/target-clean-train

python experiments/optical-decoder/synthetic.py \
  --clean local-data/target-clean-train \
  --out local-data/target-synthetic-train \
  --bursts-per-target 16 --frames 9 --severity .65 --seed 31
```

`targets` reconstructs canonical clean matrices from the sender-derived bytes
and frame metadata already in the joined dataset. It verifies regenerated
headers, physical labels and masks against the source, and makes one clean
example per receiver session/payload/frame/phase, regardless of how many real
captures show it. **No noisy camera pixels are treated as clean truth.** The
sender/receiver identities, whole-payload identity and split are preserved;
recording files remain unchanged. Multiple joined source directories can be
passed as a comma-separated `--dataset` list. This command supports the capture
app’s single logical frame with palette phases; it rejects incomplete multipart
payloads rather than inventing missing target bytes.

Sixteen independent corruption bursts with nine frames each provide **144
synthetic observations per clean target**. Independent bursts vary optical
conditions; nine frames of one burst are correlated. The generator reports the
actual target/sample count and estimates its disk budget before writing. This
avoids duplicating the same canonical clean raster for every real capture.

Generate training targets **only from `real-train`**. Do not add validation or
test target matrices to training, including through a pilot checkpoint. Synthetic
derivatives of a real target remain in that target's split; changing a seed or
folder name does not make them independent held-out data.

The matched examples remove target-content differences between the real and
synthetic parts of an update. They do not by themselves make camera artifacts
identical or prove domain alignment; the new corruptions and held-out real
results test that hypothesis. The clean references are offline supervision,
not an additional model input or a browser decoding oracle.

### Train the paired model

Reuse the existing direction sidecar. If it does not exist, create it once:

```sh
python experiments/optical-decoder/session_groups.py \
  --pc-to-phone local-data/phone/*decoder-captures*.json \
  --phone-to-pc local-data/pc/*decoder-captures*.json \
  --datasets local-data/real-train local-data/real-validation \
  --out local-data/session-groups.json
```

The paths follow the existing capture organization: captures recorded on the
phone are PC→phone, and captures recorded on the PC are phone→PC. Verify those
assignments; filenames/UUIDs are not used to infer direction. The helper refuses
to overwrite an existing sidecar. The mapping is optional for generic training,
but the following run uses it to balance and report both directions.

```sh
python experiments/optical-decoder/train.py \
  --train local-data/real-train local-data/target-synthetic-train \
  --validation local-data/real-validation \
  --initialize checkpoints/optical-decoder/wide-pilot/best.pt \
  --session-groups local-data/session-groups.json \
  --paired-targets --real-fraction .5 \
  --lr .00005 --steps 32000 --eval-every 1000 --validation-samples 0 \
  --device cuda --out checkpoints/optical-decoder/wide-paired
```

Each effective update pairs every chosen real capture with one synthetic of its
exact session, payload, frame and phase, with identical label/mask/expected-byte
hashes. Pairing applies across the complete optimizer update, so shuffled
microbatches still have equal real/synthetic totals. Missing targets, mismatched
labels, unrelated synthetic targets and incompatible split identities fail
rather than silently falling back to unrelated images. Use only the new paired
synthetics for this run, not a random-payload corpus alongside them.

With effective batch 16 and two mapped directions, an update contains four real
PC→phone captures, four real phone→PC captures, and their eight matched synthetic
counterparts. Real draws remain balanced by group/session/payload/phase; the
sampler avoids duplicate bursts where possible. Both sides of a pair use the
same ordinary classification objective. There is no consistency loss,
reconstruction head, online real augmentation or hidden recipe switch.

The run writes `settings.json`, `history.json`, `baseline-validation.json`,
`final-validation.json`, selected `validation.json` and `best.pt`. History records
source/group CE, source counts, pairing counts, repeated captures, and validation
recovery. Inspect the best checkpoint's per-direction results, not only the
final update. All model selection uses the unchanged real validation set.

### Compare against the retained baseline

```sh
python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/optical-decoder/wide-real/best.pt \
  --session-groups local-data/session-groups.json \
  --max-samples 0 --device cuda \
  --out local-results/optical-decoder/wide-real-grouped-validation.json

python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/optical-decoder/wide-paired/best.pt \
  --session-groups local-data/session-groups.json \
  --max-samples 0 --device cuda \
  --out local-results/optical-decoder/wide-paired-validation.json
```

Report the generation summaries, new `history.json`, and both full validation
reports. The practical objective is better exact recovery in phone→PC without
losing PC→phone performance, with zero wrong accepted frames. Validation is used
for model selection; evaluate untouched real test sessions only after choosing
the model. Synthetic accuracy alone does not establish device improvement.

## Export the selected checkpoint

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/wide-paired/best.pt \
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
