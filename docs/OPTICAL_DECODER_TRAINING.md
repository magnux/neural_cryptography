# Optical decoder M2 — capture, train, and evaluate

The software for collecting examples, generating synthetic interference,
training the small U-Net, evaluating recovery, and exporting a checkpoint is
implemented. The full-resolution tube has improved recorded PC→phone recovery.
The next combined model, **attention-tube**, adds a learned global positional
anchor, window attention and variable 5–8-frame observations. Its immediate
phone→PC target is **at least 50% exact recovery**; this remains unachieved by
the reported runs, and live learned transport remains open. Focus on G64/16 colors,
using G64/4 colors as a control. G128 remains a structural export case; train a
separate network with a 128×128 bottleneck if that experiment becomes necessary.

The [design](OPTICAL_DECODER.md) and [roadmap](ROADMAP.md) describe how this feeds
the Neural ZKBoo color-authentication POC. M2 now supports single captures and
multiframe learned fusion. Live transport/authentication integration remains
subsequent work; recording files remain unchanged.

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
   the selected interval (100 ms by default; 100/250/500/1000/2000 ms available).
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
   over at least 500 ms before saving. Missing headers reset the guard. At 100 ms,
   five accepted captures span about 400 ms after settling: roughly 900 ms from
   initial detection if every scan succeeds. Slower processing stretches this
   cadence; no concurrent scan jobs or queued frames are created. Payload
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

## Organize recordings and rebuild datasets

Sender and receiver downloads now start with the **eight-digit hexadecimal
CRC32 of the first logical matrix's payload**. This is `info.chunkCRC`, already
present in both JSON files. Palette phases share it; the header checksum changes
between phases and is deliberately not used. The original sequence-ID and
receiver session/timestamp suffixes distinguish downloads. A receiver batch containing only
diagnostics uses `unpaired` because it has no decoded matrix header. If a batch
contains several sequences, its prefix describes the first retained matrix;
the offline join still considers every record's full header.

The prefix is an organizational aid, not a unique identity or authentication
check. Pairing still validates the full header and sender payload. Recording
JSON content, versions, session identities and capture timestamps are unchanged.

Put both devices' exports under these directories:

| Directory | Contents | Dataset split |
| --- | --- | --- |
| `local-data/phone/train/` | Sender and receiver exports saved on the phone | Training |
| `local-data/pc/train/` | Sender and receiver exports saved on the PC | Training |
| `local-data/phone/val/` | Held-out exports saved on the phone | Validation |
| `local-data/pc/val/` | Held-out exports saved on the PC | Validation |

In the current collection, prefixes **6, 7, 8, d, e, f** belong to `val` on
both devices and the other prefixes belong to `train`. Keep this arrangement
when renaming. For future collections, assign splits before removing manual labels. Keep
whole physical sessions and payloads in one split. Filesystem creation dates
and the original manual prefixes do not participate in joining or grouping.

Rename existing downloads using their JSON contents:

```sh
node scripts/rename-decoder-recordings.mjs --dry-run
node scripts/rename-decoder-recordings.mjs --apply
```

The command scans the `phone` and `pc` trees recursively. It replaces only the
leading manual/checksum prefix, preserving the rest of each filename exactly,
including any existing timestamp. Files stay in the same split/device directory;
file bytes are preserved, conflicting destinations are refused, and rerunning
is safe. No recording-format conversion is performed. For example,
`6_decoder-sender-7b37095e6b0768e7.json` becomes
`<crc32>_decoder-sender-7b37095e6b0768e7.json` in the same `pc/val` directory.

Once the current training job has finished, rebuild the standard datasets:

```sh
node scripts/rebuild-decoder-data.mjs --python python --dry-run
node scripts/rebuild-decoder-data.mjs --python python
```

Use the active training virtual environment for `python`. Do not rebuild while
a training/evaluation process is reading these paths. The command checks raw
inputs, pairing, split ownership and dependencies before replacing outputs.
It rejects unmatched captures and conflicting/duplicate records instead of
silently shrinking the dataset. Captures in `phone` map to **PC→phone**; captures
in `pc` map to **phone→PC**, with the sender required on the opposite device.

Only these generated outputs are replaced:

| Output under `local-data/` | Purpose |
| --- | --- |
| `real-train`, `real-validation` | Joined calibrated captures with original burst/session identities |
| `target-clean-train`, `target-clean-validation` | Canonical clean targets reconstructed from sender truth |
| `target-synthetic-train`, `target-synthetic-validation` | Corrupted views paired with those exact targets |
| `session-groups.json` | Current receiver-session direction labels |

Other datasets, raw recordings and checkpoints are left alone. Generation
reuses these names on each run; no numbered or dated dataset copies accumulate.
After preflight, the previous managed outputs are removed to avoid requiring
space for two synthetic corpora. A later processing/disk failure leaves an
incomplete rebuild; fix the cause and rerun before training.

The training corpus uses the current harder recipe, including glare, native
capture-resolution loss and RGB-offset motion blur, with **32 bursts × 12 frames
per clean target**, seed 41. Validation synthetics use **4 bursts × 12 frames**,
seed 42, and remain in the validation split. Model selection should still use
`real-validation`; synthetic validation is an additional diagnostic. The command
exposes `--frames`, `--bursts-per-target`, `--validation-bursts-per-target`,
`--seed`, severity and output-budget controls in `--help`. Training `--frames 5` remains
independent of the number of generated exposures per burst.

Existing real burst IDs are preserved; neither renaming nor rebuilding makes
temporally correlated captures into independent physical sessions. The model
continues to draw random distinct captures inside each burst. Newly added
sessions enter the next rebuild without any handwritten sender/capture lists.

## Join labels offline manually

The receiver never loads the sender manifest. The rebuild command above uses
the canonical JavaScript codec. For an individual custom join, run:

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
**For the recommended attention-tube run, skip this optional pilot generation
and use the [target-paired run below](#recommended-next-training-run).** It
starts fresh combined-model weights using the current paired synthetics. Existing
tube and longtube checkpoints remain available for their own evaluation or continuation.

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
python experiments/optical-decoder/train.py \
  --train local-data/synthetic-train \
  --validation local-data/synthetic-validation \
  --frames 5 --device cuda --out checkpoints/optical-decoder/tube-pilot
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
With explicit real groups, fixed-frame checkpoint selection uses mean exact
recovery across those groups, then mean CE as the tie-break; otherwise it uses
global metrics. Attention-tube averages the per-count selection metrics across
K=5/6/7/8, balancing real groups within each K when the sidecar is supplied.
Read the per-direction results too: mean recovery alone can improve while a
still-failing direction stays at zero.

`--validation-samples 0` evaluates all validation captures and is the default.
Capped evaluation uses a fixed session/profile-balanced subset and must include
every mapped real group. Standalone `evaluate.py --max-samples 0` also evaluates
all examples. Reports include per-session/group CE, error and recovery,
confidence calibration, phase identities, raw/hard/erasure recovery, wrong
accepted frames, and inference timing. Multiframe reports count complete stacks,
not individual exposures: `inputCaptures` records consumed observations and
`burstGrouping` reports discarded leftovers/insufficient groups. For fixed-frame
models, 768 captures in 24 phase pools of 32 give 144 five-frame candidates,
consuming 720 captures and leaving 48 unused. Attention-tube instead forms
eight-member groups and reports 96 candidates for each nested K in this example. Actual grouping depends
on burst boundaries. Never compare the raw candidate denominator directly with
768 single-exposure candidates. Classical recovery averages logical scores over
exactly the same T observations;
expected bytes are only checked after independent RS/CRC acceptance. Offline
stack recovery is not live throughput.

The evaluation draw is a fixed seed-7 permutation within each burst, followed
by disjoint grouping at the model capacity. Attention-tube evaluates nested
5/6/7/8 subsets of each eight-member group. It is independent of timestamps, directory paths
and input-file order. Existing checkpoints can be reevaluated without retraining;
use a new report filename because previous chronological reports used different
input groups. `selection` and `burstGrouping` report the method and seed. Training
settings/history also record the policy.

Whole-session and whole-payload split checks, including checkpoint ancestry,
remain mandatory. Keep sender/receiver identities intact when generating paired
targets. Use untouched test sessions only after selecting the model and receiver
settings on validation.

## Current models and latest result

The default **`normalized-tube`**, residual **`longtube`** and new
**`attention-tube`** all use 32-channel feature streams and retain the full
256×256 crop for G64: **256→128→64→128→256**, followed by **256→128→64** at the
outlet. The bottleneck matches the target grid. A future G128 attention model
needs separate design/training; this combined model is G64-only.

| Profile | Input observations | Parameters | Hidden structure |
| --- | --- | ---: | --- |
| `normalized-tube` (default) | Fixed T=1…64 | 144,432 at T=5 | Two normalized convs per resolution block |
| `longtube` | Fixed T=1…64 | 279,792 at T=5 | Four convs per outer block, eight at bottleneck, residual pairs |
| `attention-tube` | Variable K=5…8, capacity 8 | 432,848 | Longtube body + learned global anchor + two window-attention blocks |

The fixed-frame models accept `[N,3T,4G,4G]`. Attention-tube accepts
**`[N,32,256,256]`: 24 RGB planes followed by eight binary presence planes**.
All emit 16 physical-color logits per cell. RGB is clipped to [-1,1] before
learned processing; stored captures retain [-1.5,1.5] calibration overshoot.
Missing slots are masked inside the new model, so their values cannot influence
its output. The input layout is explicit in checkpoints and exported metadata.

The new global positional tensor has shape **`[1,32,64,64]`**, initialized from
normal(0,.02), expanded bilinearly to 256×256 with `align_corners=False`, and
added **immediately after the inlet**. This is a learned starting reference for
the whole network, not only a spatial hint inside attention. It adds 131,072
parameters while keeping image features full resolution. A full independent
32×256×256 positional tensor alone would exceed two million parameters.

Two attention blocks follow the 64×64 residual bottleneck. Each has four heads,
32 feature channels, 8×8 windows, a 32→64→32 ReLU feed-forward layer, pre-normalization
and residual additions. The second uses a four-pixel window shift with an
explicit nonwrapping boundary mask. All longtube convolutional blocks retain
their current depth and normalization. No fixed sinusoidal positions or
chronological frame-slot positions are added. See the
[architecture diagrams](OPTICAL_DECODER.md#anchored-attention-tube).

The existing tube and longtube checkpoint tensors remain unchanged and loadable.
Cross-profile initialization is rejected: train attention-tube fresh. An explicit
single→T initialization still applies to the existing fixed-frame profiles;
attention-tube always has capacity eight. Omitting `--frames` chooses eight for
attention-tube and one for a fresh fixed-frame model. The model's eight-slot input
uses 8 MiB per example before activations/runtime buffers; fewer valid observations
do not make the exported graph smaller.

The submitted full-resolution tube run used five frames, LR `1e-4`, 32,000
updates, and **batch 16 × accumulation 2 = effective batch 32**. Selection chose
step 29,000:

| Validation group | Selected 29k CE | Cell error | Exact recovery | Final 32k CE / cell error / exact |
| --- | ---: | ---: | ---: | --- |
| All | .4661 | 17.31% | 68/144 (47.22%) | .4709 / 17.34% / 68/144 |
| PC→phone | .0851 | 2.49% | 68/96 (70.83%) | .0846 / 2.44% / 68/96 |
| Phone→PC | 1.2282 | 46.94% | 0/48 | 1.2436 / 47.14% / 0/48 |

Classical fusion recovered zero groups with the same five observations. The
result is encouraging on PC→phone, while phone→PC remains the limiting direction.
The final interval's phone→PC training CE is .8144, compared with 1.2436 on
validation; real/synthetic training CE is .4898/.0840. Depth and more realistic
color-motion artifacts are the next hypotheses, not established remedies.
Earlier runs used effective batch 16, so the improvement cannot be attributed
solely to architecture. Historical results are in [VALIDATION.md](VALIDATION.md).
The user subsequently reported the ongoing longtube run reaching **72/144**
validation recoveries before 16,000 of 32,000 updates, with no phone→PC exact
recoveries yet. This is a user progress report, not a completed attached result.
The next experiment combines the input, positional-anchor and attention upgrades;
there is no broad ablation plan. Phone→PC **50% exact recovery** is the target,
not a result claimed for the new model. Its matched nested validation groups
change the candidate denominator; live transport must still be tested separately.

## Generate the updated paired synthetic corpus

The rebuild command already performs this step. For a manual regeneration,
reuse `target-clean-train`, or reconstruct it once with the first command below.
Generate the RGB-offset corpus into an empty `target-synthetic-train` directory. This keeps seed 41 and the existing resolution/glare settings while
adding the new color-motion effect. Original recordings and joined real data
remain unchanged.

```sh
# Reuse target-clean-train if already generated; otherwise run:
node scripts/decoder-dataset.mjs targets \
  --dataset local-data/real-train \
  --out local-data/target-clean-train

python experiments/optical-decoder/synthetic.py \
  --clean local-data/target-clean-train \
  --out local-data/target-synthetic-train \
  --bursts-per-target 32 --frames 12 --seed 41 \
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
passed as a comma-separated `--dataset` list. This command supports the capture
app’s single logical frame with palette phases; it rejects incomplete multipart
payloads rather than inventing missing target bytes.

Thirty-two independent corruption bursts with twelve frames each provide **384
synthetic observations per clean target**. Frames of one burst remain correlated.
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

## Recommended next training run

**Train `attention-tube` from scratch**, with capacity eight and randomly selected
5–8 observations at `1e-4` for 32,000 updates. Use the current `real-train`,
`target-synthetic-train`, `real-validation` and direction sidecar. Twelve-frame
synthetic bursts already support this run; **no synthetic regeneration is needed
solely for the model change**. Each real and synthetic training burst must have
at least eight distinct observations, and every sample must be G64. Add recordings
through the [rebuild workflow](#organize-recordings-and-rebuild-datasets) after
active training stops if necessary.

Keep ordinary masked cross-entropy and mild real augmentation. Use **batch 8 ×
accumulation 4 = effective batch 32**; batch 4 × 8 is an alternative if activation
memory is limiting. Attention and the positional anchor add parameters and
working memory, so measure actual GPU usage. The new output directory is
`attention-tube8`. Do not pass `--initialize` from a tube/longtube checkpoint.
The current longtube run can finish independently; its code and weights remain
supported and its output directory is not reused.

The rebuild command creates `session-groups.json` from receiver locations and
validates both directions. Use the regenerated sidecar when adding sessions.
For custom datasets, `session_groups.py --help` describes explicit assignments.

```sh
python experiments/optical-decoder/train.py \
  --train local-data/real-train local-data/target-synthetic-train \
  --validation local-data/real-validation \
  --model attention-tube --frames 8 \
  --session-groups local-data/session-groups.json \
  --paired-targets --real-fraction .5 \
  --batch-size 8 --accumulate 4 \
  --real-noise-std .008 --real-tint-stops .06 --real-exposure-stops .08 \
  --lr .0001 --steps 32000 --eval-every 1000 --validation-samples 0 \
  --device cuda --out checkpoints/optical-decoder/attention-tube8
```

Every selected real example is paired with a synthetic example of the same
session, payload, frame and phase, with matching label/mask/expected-byte hashes.
The pair receives the same available count K, drawn uniformly from 5, 6, 7 or 8.
Actual members are sampled without replacement within their own bursts, then
randomly assigned to the fixed eight slots. They never cross phases, payloads,
sessions or independent synthetic trajectories. Real photometric augmentation
is applied to the actual RGB observations **before zero padding**. Presence
planes are not augmented. The model does not need to infer missing frames from
mid-gray RGB values.

With effective batch 32 and two mapped directions, an update has eight real
PC→phone examples, eight real phone→PC examples and their sixteen synthetic
partners. The number of actual exposures varies with K; interval accounting
records it instead of treating all eight slots as observations. Synthetic
`--frames 12` continues to mean generated exposures per independent burst;
training `--frames 8` is this model's capacity, with actual counts 5–8.

The run writes `settings.json`, `history.json`, `baseline-validation.json`,
`final-validation.json`, selected `validation.json` and `best.pt`. Validation
reports have a **`byFrameCount`** map for K=5/6/7/8 with complete per-count CER,
loss, classical/learned recovery, session and direction metrics. `frameCapacity`
is eight, `actualFrameCounts` lists the four tested counts, `uniqueCandidates`
counts the shared groups, and `observationCases` sums their evaluations across K. The top-level
selection combines the per-count scores: equal weights for K and, with the
sidecar, equal weights for real directions (`frame-count-group-balanced-real`).
Loss is the tie-break. It does not select whichever K happens to look best.

History retains `validationByFrameCount`, `phoneToPCGoal` and the training
`trainFrameCounts` histogram alongside source/group statistics. The
`phoneToPCGoal.targetExactRate` is .5, with actual recovery and `achieved`
reported for each K. Its overall `exactRate` is the mean across K, while
`achievedForAllFrameCounts` requires every tested K to meet the target; missing
direction mappings do not count as success. The progress display summarizes
these metrics; full results remain in JSON. Inspect the five-frame result as
well as the eight-frame result, and check both directions. A shortfall from 50%
is an experiment result, not a reason to report success on aggregate recovery alone.

### Evaluate the four observation counts

```sh
python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-validation --split validation \
  --checkpoint checkpoints/optical-decoder/attention-tube8/best.pt \
  --session-groups local-data/session-groups.json \
  --max-samples 0 --device cuda \
  --out local-results/optical-decoder/attention-tube8-validation.json
```

The evaluator ranks each burst with its fixed seed-7 random permutation, forms
disjoint groups of eight, and evaluates nested first-5/6/7/8 subsets with fixed
slot assignments. Each K therefore sees the same candidate targets and a subset
of the same exposures. Canonical classical fusion and neural recovery receive
only the K actual member IDs; zero padding is never supplied as an observation.
Remainders and insufficient bursts are reported, never duplicated into a group.

For the previously reported 24 phase pools of 32 captures, there are **96
candidates per K**, compared with 144 in the fixed-five-frame evaluation. The
four per-count evaluations reuse candidates, so their combined count is not a
count of independent matrices. Compare per-count/per-direction rates, not raw
success totals across these different grouping policies. Existing fixed-frame
tube/longtube evaluations retain their original grouping behavior.

For this fresh run, `baseline-validation.json` evaluates random step-zero weights;
it is diagnostic and contains no parent checkpoint. Share selected
`validation.json`, `history.json` and the standalone validation report. Keep
untouched test sessions for the selected model/settings. Phone→PC recovery of at
least 50% is the immediate target; zero wrong accepted payloads and the existing
4-color control remain required checks before live integration.

### Existing fixed-five-frame longtube run

The earlier command remains supported, with its own output directory:

```sh
python experiments/optical-decoder/train.py \
  --train local-data/real-train local-data/target-synthetic-train \
  --validation local-data/real-validation \
  --model longtube --frames 5 \
  --session-groups local-data/session-groups.json \
  --paired-targets --real-fraction .5 --batch-size 8 --accumulate 4 \
  --lr .0001 --steps 32000 --eval-every 1000 --validation-samples 0 \
  --device cuda --out checkpoints/optical-decoder/longtube5
```

This is a reference for the current run, not another requested ablation. Do not
start it again over an existing run directory. Its fixed-five-frame reports
retain the historical 144-candidate grouping for the same 24 pools.

## Export the selected checkpoint

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/attention-tube8/best.pt \
  --out experiments/optical-decoder/artifacts/selected
npm ci --prefix experiments/optical-decoder/browser-runtime
npm run prepare:decoder -- --artifacts experiments/optical-decoder/artifacts/selected
npm run test:decoder:wasm -- --artifacts experiments/optical-decoder/artifacts/selected
```

Attention-tube export produces **G64 only**, with fresh PyTorch numerical fixtures.
Tube and longtube continue to export both G64/G128 using the selected weights;
the latter is a structural/parity check, not validated cross-size recognition.
The current `optical-decoder` manifest records trained status, checkpoint hash,
required `inputLayout`, `framesPerInput` capacity, `minFramesPerInput`, allowed
`frameCounts` and channel dimensions,
required capture-normalization and `networkInputClip: [-1,1]` metadata,
and bounded public training provenance.
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
