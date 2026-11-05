# Multires recordings: collect by size and upload to the demo server

The collector supports **8, 16, 32, 64, 128 and 256 cells per side**, with new
captures rectified to **2×2 pixels per cell**. The intended `multires` training
profile uses **16 colors and four palette phases**. It records camera evidence;
the receiver never receives sender payloads or labels. Sender manifests provide
the labels during the offline join.

The collector now uses the header-free [Neural-QR layout](NEURAL_QR.md): three
standard finders, an asymmetric bottom-right fiducial, two repeated phase-cue
pairs and fixed color references. There is no monochrome byte header. The
receiver knows the grid size from its settings and reads the phase optically.
The demo server automatically shares the active collection code, grid profile and
dataset split with the receiver. This small session descriptor contains no clean
pixels or labels. Static hosts still support manually entering the collection code.

The [multires architecture](NEURAL_CORRECTOR.md#multires-next-architecture) is
specified and prototyped; production training/browser integration is pending.
The established 4×4-cell training commands must not be used on these new
2×2-cell Neural-QR data. The existing authentication demo remains a separate
OBM5/QR-ACK baseline until the new model and packet codec are integrated.

## Start the upload-enabled collector

From the repository root:

```sh
npm run prepare:apps
python apps/color-optical-bench/serve.py --port 8000 \
  --recordings-dir local-data/recordings
```

Open `http://localhost:8000/capture.html` on that computer. For a phone, use the
same server over trusted HTTPS, with your existing trusted certificate:

```sh
python apps/color-optical-bench/serve.py --host 0.0.0.0 --port 8443 \
  --https --cert certs/dev.pem --key certs/dev-key.pem \
  --recordings-dir local-data/recordings
```

Both devices open `https://YOUR-CERTIFICATE-HOST:8443/capture.html`. Uploads are
enabled only when `--recordings-dir` is supplied. A static host still supports
downloads, but cannot accept uploads. Collection and correction remain in the
browser; the Python server only stores the JSON explicitly submitted by Upload.

## Collection plan

Use **four phases** for every size, including small control matrices. Each
phase applies a different known palette permutation to the same logical payload.
Capture all four before advancing to a fresh payload. Keep each phase static
while recording; this is dataset acquisition, not timed live transmission.

| Grid size | Crop | 200 MiB budget | 400 MiB budget (default) | 800 MiB budget |
| --- | --- | --- | --- | --- |
| 8 / 16 / 32 / 64 | 16 / 32 / 64 / 128 per side | 32/phase | 32/phase | 32/phase |
| 128 | 256×256 | 32/phase | 32/phase | 32/phase |
| 256 | 512×512 | 8/phase | 16/phase | 32/phase |

The budget dropdown controls the maximum encoded recording size. These are
recommended per-phase maxima across all four phases, within the unchanged
128-record limit. Pools can be smaller. Change the budget while the batch is
empty; the application preserves it when clearing a saved batch.

A G256 capture stores about **5 MiB** of base64-encoded RGB input and cell/reference
samples. Four phases with 16 captures each require about **320 MiB**, and 32 each
require about **640 MiB**, plus metadata. Thus 400 MiB fits 16 per phase; use 800 MiB
for 32. G128 with 32 per phase is about 160 MiB plus metadata and fits 200 MiB.
Actual browser memory exceeds encoded file size. If a phone cannot comfortably
retain the larger pool, select a lower budget and save more batches.

Upload and download serialization build JSON one record at a time, avoiding one
whole-file JavaScript string. The offline join also reads capture JSON in pieces.
The format remains ordinary JSON with the same `neural-qr-captures` version.
**Already recorded smaller pools remain valid and need no conversion.**

Start with **three training sessions, one validation session and one test session
per direction**, with **three fresh logical payloads per size in each session**.
This is an initial recording budget, not a generalization guarantee. The phone's
front and rear cameras are separate arrangements. Favor new payloads and changed
physical conditions over hundreds of nearly identical views of one matrix.

For each size, collect easy, typical and difficult views: vary distance, angle,
brightness and small natural motion without losing the complete matrix. Collect
both phone→PC and PC→phone. G256 may require a larger displayed area and a camera
resolution that preserves sufficient cell detail; an upsampled crop does not
restore detail that the camera never captured. Record acquisition failures too; they remain bounded diagnostics rather than labeled model inputs.

Keep an entire physical session in **one split**, including every size, payload
and phase from that arrangement. Do not put different phases or neighboring
bursts of the same session into training and validation. The UI locks the split
for an active session, and the offline join checks physical-session and payload
ownership across splits.

## Device workflow

1. Run the upload-enabled server and reload `capture.html` on both devices.
   On the sender choose the split and grid, then create a random matrix.
2. The sender automatically uploads its manifest as `<filename>.json.tmp`.
   Only after it is stored does the server publish `collection_session.json`.
   It contains an opaque collection-session ID, sender physical-session ID,
   public collection code, profile, split and creation time—no clean matrix.
3. The receiver reads the active session automatically. Its code, grid and split
   are filled from that descriptor. Select the budget/pool, start the camera and
   press Record. The receiver retains its own physical-session identity.
4. Keep each displayed phase static while its pool fills; advance all four phases
   manually. Full pools continue balanced replacement. Pause when finished.
5. Upload the receiver recording. The server first saves it durably, then finalizes
   the matching sender by removing `.tmp`, and finally removes the active
   collection-session metadata. The sender can then create the next matrix.
6. Clear the saved receiver batch and continue. The next active assignment is
   picked up automatically. Changing the physical arrangement or camera starts
   a new physical session; changing split requires a new physical session too.

An active assignment cannot relabel a populated batch. If the server changes or
ends that assignment while collection is underway, recording pauses; existing
samples remain on the device. A different split is never silently applied to an
already-used physical session. Acquisition still identifies phase optically; the
server's descriptor supplies association, not pixel truth or phase labels.

A failed upload preserves local samples, the staged sender and active metadata
for retry. Identical receiver retries are idempotent. If an inactive-session retry
can confirm only that the exact receiver file is saved, it reports that fact
without claiming sender finalization. Staging a different matrix while one is
active is refused. The explicit abandon action clears the active descriptor but
keeps its `.tmp` sender; it does not delete recordings.

The server restores the active descriptor after restart. Downloads and manual
code/grid selection remain available without an upload-enabled server; manually
saved current recordings can still be joined as before. A direct upload of an
existing recording without the optional collection-session ID remains a normal
file upload and does not finalize an unrelated active session.

## Storage and automatic metadata

The server derives names from the recording identities and stores:

```text
local-data/recordings/collection_session.json                 # active only
local-data/recordings/train/g64/sender/<code>_neural-qr-sender-<session>.json.tmp  # pending
local-data/recordings/train/g64/sender/<code>_neural-qr-sender-<session>.json      # complete
local-data/recordings/train/g64/receiver/<code>_neural-qr-captures-<session>-<time>.json
local-data/recordings/validation/g256/receiver/...
```

Diagnostic-only captures go under `<split>/unknown/receiver/`. The storage
directory must be outside the served app; recordings have no public download
endpoint. The upload token/origin checks protect the local collector workflow;
this is not a multi-user account or authorization service.

Metadata includes a locally generated browser-installation device ID, available
browser/platform information, inferred device class, screen/viewport dimensions,
pixel ratio and optional device label. Camera metadata records available track
label, selected device/group IDs, facing mode, actual dimensions and frame rate.
The per-frame camera settings remain attached to the evidence after the camera
stops. Sender and receiver environments are both retained during offline joining.

The local device ID identifies this browser storage, not a globally verified
physical device. Camera identifiers are browser/origin scoped and may change;
exact hardware model names are not always available. A manual label can improve
human grouping without inventing missing information. No high-entropy device
fingerprint or separate hardware permission is requested.

Recordings now use `neural-qr-sender` and `neural-qr-captures`, version 1.
The receiver stores `pixelsPerCell: 2`, its public assignment and observed phase,
normalized CHW input, raw RGB cell/reference samples, acquisition diagnostics and
environment/split metadata. Sender files store random logical symbols and their
hash for offline labeling. There is no fake or reconstructed OBM5 header.

## Admission and balanced retention

The first filter is geometry and phase agreement: all finder geometry and the
asymmetric orientation must be readable, and the phase pairs at the top and
bottom must confidently agree. An uncertain phase or transition exposure is
skipped. The 500 ms settling guard restarts when a corner moves by more than
4% of the current matrix diagonal. Green means acquired geometry/phase; orange means tentative geometry
without accepted cues. Neither color asserts an exact payload or correct manual
collection assignment.

Within a valid phase, retain an initial pool, then freeze a robust reference
from that warm-up. A trimmed medoid gives a representative captured image;
median radial distance and MAD provide its scale, with a small scale floor.
Freezing the reference avoids shrinking the acceptance region repeatedly as
central samples replace other samples. A new batch establishes a new reference.

The selector targets roughly **70% core, 25% shoulder and 5% moderate tail**
using three robust distance bands. Quotas are rounded to the actual pool size,
so a six-capture G256 pool has no dedicated far-tail slot. Observations beyond
the outer band are rejected; isolated poor warm-up samples are replaced first.
Near duplicates are skipped. Within a filled band, a randomized refresh permits
new evidence without rewarding ever larger differences. Empty bands are not
fabricated, and the result is not a mathematically fitted normal distribution:
RGB distances are nonnegative and their distribution depends on the scene.

The representative image is an agreement estimate, not clean ground truth.
A consistently misaligned or wrong matrix can still form a coherent cluster.
If the arrangement changes substantially, save the batch and start another;
do not move far away solely to increase the displayed mean distance. Vary
physical conditions across sessions while keeping each batch mostly typical.
Selection never receives sender symbols, payload CRC outcomes or model labels.

The collection code is the CRC32 of the logical matrix, validated against its
SHA-256 sender manifest during joining. It is supplied by the server descriptor
(or entered manually), not decoded from the camera. **Verify the displayed matrix
belongs to the active assignment before recording,
and keep the same matrix through all four phases.** A wrong assignment cannot
be ruled out from marker detection or capture similarity alone. Review suspect
sessions offline rather than assuming the code authenticates the image.

## Join uploaded recordings by size

After completing receiver uploads:

```sh
node scripts/corrector-dataset.mjs join-uploads \
  --recordings-root local-data/recordings \
  --out local-data/multires-joined
```

The output must be a new or empty directory. Pending `.json.tmp` senders and the
root `collection_session.json` are not joined. This command discovers completed recordings,
checks public code/profile/phase matches and split ownership, and writes datasets under
`local-data/multires-joined/<train|validation|test>/g<size>/`. The joined manifest is `neural-qr-dataset`, version 1. Each sample retains
its sampling density, `[3,2G,2G]` input shape, device pair, camera metadata and
physical-session grouping. Source recordings are never deleted or overwritten.
Unmatched captures must be resolved by uploading the missing sender manifest;
the tool does not guess labels from filenames.

The existing four-pixel-per-axis synthetic/training pipeline does not accept
these Neural-QR datasets. The old rebuild entry point stops before cleanup and
directs new recordings to `join-uploads`; it does not delete datasets. Keep the
new joined output separate from existing training inputs. Multires dataset generation
and training are the next implementation step. Existing 4×4-cell datasets and
their documented training recipe remain usable.

Browser API references: [track settings](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/getSettings),
[camera labels](https://developer.mozilla.org/en-US/docs/Web/API/MediaDeviceInfo/label),
[browser client hints](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/userAgentData).
