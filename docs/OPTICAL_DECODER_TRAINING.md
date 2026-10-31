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

The default fresh pilot uses **wide**, AdamW at **1e-4**, **32,000** optimizer
updates, microbatches of 4, and accumulation of 4 (effective batch 16).
With `--initialize`, defaults are **5e-5** and **32,000** updates. Use `--steps`
to override either phase's budget. Microbatches share a grid size and
palette; inactive color classes are masked before cross-entropy.

```sh
python experiments/optical-decoder/train.py \
  --train local-data/synthetic-train \
  --validation local-data/synthetic-validation \
  --model wide --device cuda --out checkpoints/optical-decoder/wide-pilot
```

For a quick CPU pipeline check, use `--device cpu --steps 10 --batch-size 1
--accumulate 1 --eval-every 5 --validation-samples 4`, with a small generated
corpus and a fresh run directory. A short smoke run need not recover a payload.

At each evaluation interval, the trainer chooses `best.pt` by validation exact
frame recovery, then by lower data-cell loss to break ties. Exact recovery uses
the existing JS palette mapping, reliability ranking, RS, and CRC, comparing
accepted bytes with truth afterward. Selection can improve loss while exact
recovery remains zero; a saved checkpoint is not automatically a usable decoder.
`history.json` and `validation.json` record these outcomes. An accepted incorrect
payload stops training for investigation.

Fine-tune with real captures and synthetic examples at a lower learning rate:

```sh
python experiments/optical-decoder/train.py \
  --train local-data/real-train local-data/synthetic-train \
  --validation local-data/real-validation \
  --initialize checkpoints/optical-decoder/wide-pilot/best.pt \
  --real-fraction .5 --device cuda \
  --out checkpoints/optical-decoder/wide-real

python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-test \
  --checkpoint checkpoints/optical-decoder/wide-real/best.pt --split test \
  --device cuda --out local-results/optical-decoder/real-test.json
```

Training validation defaults to 128 captures, selected deterministically with
round-robin coverage across receiver sessions, sources, and grid/palette profiles.
Increase `--validation-samples` to cover the intended held-out corpus. Evaluation
reports available/selected/omitted sessions and per-session cell error, phase
counts, and exact recovery, so a poor camera direction cannot be hidden in a total.
Standalone `evaluate.py --max-samples 0` evaluates all examples; there is no
4,096-frame offline ceiling. Subset results are not full-corpus results.
After model selection, evaluate untouched real test sessions once. The loader
rejects overlapping sender sessions, receiver sessions, and payloads, including
checkpoint ancestry. Keep original session IDs intact when moving or joining files.

Reports include cell confusion/error, confidence calibration, raw CRC passes,
ordinary RS and total acceptance, exact frames, wrong accepted frames, profile
counts, and inference timing. Both methods receive one exposure per candidate.
This does not compare temporal windows or measure live goodput; those comparisons
follow when connecting the selected model to the receiver.

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
2% of stored training examples being real. With effective batch 16, 32k updates
draw each real crop about 167 times on average (about 27 by step 5,200). There is
currently no online augmentation of those real crops. `trainLoss` is the most
recent update's mixed-source mean, not an averaged real-only training loss, so its
absolute value cannot be compared directly with real validation CE.

Review found no obvious physical-color/phase, mask, or loss aggregation bug.
Actual alignment and label quality still require inspecting capture tensors.
Synthetic crops clip to nominal RGB while real affine-calibrated inputs can
overshoot to [-1.5,1.5]. Camera processing and interference may differ in other
ways. The log does not identify which mismatch matters most.

## Next training experiments

These are planned, controlled experiments, not newly enabled loss/sampler options.
Keep the wide architecture fixed and compare each change from the same pilot,
with the same splits, source ratio, seeds, and validation budget.

1. **Diagnose the selected model first.** Evaluate all 768 real validation
   captures; inspect session-level error/recovery, the confusion matrix, and
   confidence calibration. Check whether the seven recoveries come from just one
   session. Examine raw rectified inputs beside physical labels and inspect phase,
   clipping, and geometry failure patterns. Add separate rolling real/synthetic
   training metrics and a fixed real-training diagnostic subset before inferring
   failure to fit real data. Keep test sessions untouched.
2. **Control adaptation before increasing capacity.** Add validation-driven LR
   reduction and early stopping, retaining 32k as the maximum budget. Monitor
   exact recovery with CE as the tie-break; save the selected model. Compare a
   lower initial fine-tuning LR (1e-5) with the current 5e-5. The current trainer
   saves the best checkpoint but does not automatically stop or decay LR.
3. **Diversify real updates.** Sample device direction, then session, payload,
   phase, and frame, while avoiding repeated bursts within an update. Directions
   inferred from the pasted paths are 8:4 PC→phone:phone→PC in training and 4:2 in
   validation; verify this against session notes before using direction labels.
   Each bundle currently contributes 128 captures, so session balance alone will
   not change this run's weighting much. Keep 50:50 real/synthetic initially;
   test 75% real separately, since extra repetition can also increase overfitting.
4. **Augment real captures mildly online.** Small residual exposure/color/noise
   variations can preserve measured moiré while increasing variation. Geometric
   augmentation must preserve cell alignment and transform labels/masks together.
   Avoid strong extra moiré or opaque masks on already difficult captures. More
   independent payloads and sessions remain more informative than longer bursts.

For a full validation report using the existing evaluator:

```sh
python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/optical-decoder/wide-real/best.pt \
  --max-samples 0 --device cuda \
  --out local-results/optical-decoder/wide-real-validation.json
```

After those controls, compare these objective/regularization ablations:

| Candidate | Purpose | Initial experiment and limitation |
| --- | --- | --- |
| Mean CE + modest hard-cell CE | Give remaining difficult data cells more influence | Per frame, `0.8 × mean CE + 0.2 × mean CE of the hardest 25% of valid cells`; retain ordinary CE validation. The limited contribution prevents hard examples from receiving the entire objective, but cannot fix wrong labels or destroyed colors. |
| Mild focal loss | Downweight already well-classified cells | Test gamma 1 as an alternative to the hard-cell term, not simultaneously. It can also emphasize unrecoverable captures, so check held-out recovery and confidence. |
| Confidence regularization | Reduce confidently wrong predictions that disrupt erasure ranking | Compare small label smoothing (e.g. 0.02) or a small Brier-score term separately; tune on held-out validation. Check top-class error as well as confidence. |
| Weight regularization | Limit memorization of a small real corpus | AdamW already uses its default weight decay (0.01). Compare stronger decay or a small penalty toward pilot weights; too much anchoring may prevent useful adaptation. |
| Same-phase burst consistency | Encourage stable predictions across captures of the same physical symbols | Keep supervised CE, weight consistency lightly, and exclude mismatched phases/motion failures; shared wrong predictions are not evidence of correctness. |
| RS-aware auxiliary loss, later | Reduce byte errors per actual codeword rather than just mean cell errors | Use the canonical phase mapping, cell packing and interleaving; penalize soft error estimates above the correction budget. This is a surrogate, not a differentiable CRC or a replacement for exact recovery evaluation. |

Start with one objective change. Random payload colors are approximately balanced
by design; arbitrary inverse-frequency class weights or Dice/TV smoothness losses
are not justified without measured imbalance, and neighboring symbols should not
be encouraged to match. Group reweighting should address observed session or
direction failures and retain regularization, not blindly chase the hardest group.

Temperature scaling is a cheap later calibration experiment using held-out
validation: it can change confidence-based erasure choices, but cannot change
the per-cell argmax or repair the current 40.5% mean cell error by itself. The
JS logits converter already accepts a temperature; training/evaluation currently
use 1 and do not fit it automatically. Single-frame evaluation should eventually
be followed by equal-budget multi-frame recovery, matching the intended receiver.

Relevant methods: [focal loss](https://arxiv.org/abs/1708.02002),
[regularization under group shifts](https://arxiv.org/abs/1911.08731), and
[temperature scaling](https://arxiv.org/abs/1706.04599). Their published results
motivate these experiments; they do not establish an optical-decoder benefit.

## Export the selected checkpoint

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/wide-real/best.pt \
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
