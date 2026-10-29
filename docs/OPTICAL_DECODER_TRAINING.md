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
4. Set **Maximum per phase (m)** to **32** and **Keep at random (n)** to **16**.
   Press **Start / resume collection** once. Each new matrix/phase must have
   consecutive identical valid headers over at least 500 ms before saving starts.
   Missing headers reset the guard. Payload CRC failures still produce useful
   training examples. Collection waits indefinitely while the page is visible;
   pause it or stop the camera when finished.
5. At 32 samples the phase pool stops growing but alignment keeps running. To
   vary observations, make a small angle/distance adjustment while keeping the
   matrix visible, then press **Keep 16 at random & resume**. This replaces the
   current phase's pool with a uniform random subset and fills the free slots
   with new captures after a fresh stability guard. Other phase pools are intact.
   Maximum choices are 8/16/32/64; retention choices are 4/8/16/32, strictly below
   the maximum. Changing the maximum never silently deletes existing samples;
   use the thinning button if a pool already exceeds the new setting.
6. The **Mean RGB distance** updates automatically for the current phase. Note
   the value after the initial fill and after two or three thin/refill cycles.
   The previous pre-thinning value remains visible for comparison. Do not mix
   phases to raise the metric. It is diagnostic, not a quality target or an
   automatic near-duplicate rejection rule (see below).
7. When ready, advance the sender manually. Collection remains armed and starts
   a separate pool after the new phase passes the guard. Pause collection before
   **Save receiver captures**, then **Clear downloaded batch** before resuming.
   Clearing keeps the physical session ID. Download once per final batch: saving
   overlapping copies and joining both creates duplicate capture IDs.
8. Generate a second fresh sequence, save its sender manifest, and repeat.
   Include easy captures and mild motion, focus/white-balance changes, different
   distances, and moiré. Keep each displayed phase static while collecting.
9. Record device/direction/brightness/conditions in the notes. Repeat with roles
   reversed when alignment permits, and collect the 4-color control. Start a
   **New physical session** when changing camera/device arrangement or returning
   later; rear and front cameras should have distinct session IDs. Keep entire
   sessions together in a split. Stopping the camera or hiding the page pauses
   collection and preserves the samples; reopening does not restart the camera.

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

Random thinning is deliberately independent of pixels, CRC outcome, and labels.
It can retain near-duplicates and can lower or raise the mean. Blur, registration
errors, and strong noise can also raise distance, so do not maximize it blindly.
A stable value across several cycles is a useful observation, not evidence that
all device conditions are covered. Capture both easy and difficult conditions.
Receiver exports include current collection settings, per-phase distance summaries
(with a `current` flag), and the most recent 100 thinning events. These metadata
never influence decoding or offline labels.

For the next device check, report camera/direction, grid/palette, scan interval,
chosen m/n, retained counts, distance after the initial fill and after each refill,
and whether the green overlay remains easy to see. Include failures to acquire
markers/headers separately; this iteration improves collection, not the detector.

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

```sh
python -m pip install -r experiments/optical-decoder/requirements.txt

node scripts/decoder-dataset.mjs generate --out local-data/clean-train \
  --count 16 --size 64 --colors 16 --phases 4 --seed 100 --split train
node scripts/decoder-dataset.mjs generate --out local-data/clean-validation \
  --count 4 --size 64 --colors 16 --phases 4 --seed 10000 --split validation
node scripts/decoder-dataset.mjs generate --out local-data/clean-test \
  --count 4 --size 64 --colors 16 --phases 4 --seed 20000 --split test

python experiments/optical-decoder/synthetic.py --clean local-data/clean-train \
  --out local-data/synthetic-train --frames 3 --severity .65 --seed 7
python experiments/optical-decoder/synthetic.py --clean local-data/clean-validation \
  --out local-data/synthetic-validation --frames 3 --severity .65 --seed 8
python experiments/optical-decoder/synthetic.py --clean local-data/clean-test \
  --out local-data/synthetic-test --frames 3 --severity .8 --seed 9
```

`--count` counts fresh payloads; phases and burst frames multiply the resulting
samples. Keep all outputs of one payload in the same split. Choose disjoint seed
ranges for clean corpora; changing a folder name is not a split. Add separate
4-color corpora with fresh seeds to both training and validation; the training
CLI accepts multiple directories. Use the same pattern for G128 when ready.

The generator applies an RGB stripe carrier before blur/resampling, curved and
chirped channel-dependent moiré, bounded subcell jitter, short motion blur,
color/exposure/gamma changes, clipping, and shot/read noise. Disturbance evolves
across a burst rather than drawing unrelated images. This is a **calibrated-crop
approximation**; it does not simulate the full optical sensor or header failures.
Real held-out examples determine whether it covers the device's artifacts.
Synthetic classical baselines are recomputed from the same corrupted crop;
real baselines use the independently captured current-exposure samples.

## Train and evaluate

The default pilot uses AdamW at 1e-3, 2,000 optimizer updates, microbatches of 4,
and accumulation of 4 (effective batch 16). Microbatches share a grid size and
palette; inactive color classes are masked before cross-entropy.

```sh
python experiments/optical-decoder/train.py \
  --train local-data/synthetic-train \
  --validation local-data/synthetic-validation \
  --device cuda --steps 2000 --out checkpoints/optical-decoder/pilot
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
  --initialize checkpoints/optical-decoder/pilot/best.pt \
  --real-fraction .5 --lr .0001 --device cuda --steps 2000 \
  --out checkpoints/optical-decoder/real

python experiments/optical-decoder/evaluate.py \
  --dataset local-data/real-test \
  --checkpoint checkpoints/optical-decoder/real/best.pt --split test \
  --device cuda --out local-results/optical-decoder/real-test.json
```

Validation is bounded by `--validation-samples` (128 by default); set it to cover
the intended validation sessions and profiles, and inspect the report's selected
counts. Synthetic validation is only a pipeline/pilot check. After choosing the
model on validation data, evaluate the untouched real test sessions once.
The loader rejects overlaps of sender sessions, receiver sessions, and whole
payloads, including checkpoint ancestry when fine-tuning. Keep original session
IDs intact when moving or joining files.

Reports include cell confusion/error, confidence calibration, raw CRC passes,
ordinary RS and total acceptance, exact frames, wrong accepted frames, profile
counts, and inference timing. Both methods receive one exposure per candidate.
This does not compare temporal windows or measure live goodput; those comparisons
follow when connecting the selected model to the receiver.

## Export the selected checkpoint

```sh
python experiments/optical-decoder/export.py \
  --checkpoint checkpoints/optical-decoder/real/best.pt \
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
