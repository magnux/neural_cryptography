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
A public collection code, copied from the sender, associates the recording with
its manifest offline. It does not contain clean pixels or labels.

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

| Grid size | Crop | Recommended retained captures per phase | Captures across four phases |
| --- | --- | --- | --- |
| 8 | 16×16 | 32 | 128 |
| 16 | 32×32 | 32 | 128 |
| 32 | 64×64 | 32 | 128 |
| 64 | 128×128 | 32 | 128 |
| 128 | 256×256 | 16 | 64 |
| 256 | 512×512 | 6 | 24 |

These are per-upload retention budgets, not dataset-size limits. Smaller pools
of 16 per phase are a reasonable faster start for G8…G64. At G128, the default
pool is 16; at G256, six. G256 evidence is about 5 MiB per capture including
the dense input and classical sample arrays, so eight captures in each of four
phases would exhaust 160 MiB before JSON metadata. Six leaves useful headroom.
The UI caps the selected pool using the detected grid and phase count. More
captures can be collected in subsequent batches under the same physical session.

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

1. Choose the same split on both devices before beginning the physical session.
   An optional device label helps when a browser exposes only a generic model.
2. On the sender choose a grid and create a random matrix. Colors and phases are
   fixed at 16 and four. **Upload sender manifest** before creating the next
   matrix. The sender displays an eight-character collection code.
3. On the receiver enter that collection code and select the same grid, then
   start the camera, choose the pool, and press Record. Code and grid are fixed
   for the batch. Automatic alignment validates the asymmetric marker and
   repeated phase cues; a 500 ms settling guard precedes retention. Retained
   phase counts, mean distance and replacement counters stay next to the camera.
4. Once a phase has enough captures, manually advance the sender. Full pools
   continue balanced replacement instead of stopping or maximizing distance.
   Keep most captures close to the typical view, with some moderate variation.
5. Pause and **Upload receiver captures**. Wait for the saved-path confirmation.
   A failed upload keeps the captures and permits retry; it never clears them.
6. Clear the saved batch and continue. Changing grid or collection code requires a fresh batch, but
   can retain the physical session when the device arrangement is unchanged.
   Changing camera or physical arrangement requires a new physical session.
7. Repeat with a new payload, then the next size. Download remains available when
   uploads are disabled or when you want an additional local copy.

An upload snapshot is reused for a retry, including its timestamp. The server
deduplicates identical retries and refuses to overwrite a different recording
with the same filename. Uploads are limited to 160 MiB each and two in flight on
the server. Captures retain the existing 128-record browser limit; encoded-file
budgets are smaller than total browser memory use.

## Storage and automatic metadata

The server derives names from the recording identities and stores:

```text
local-data/recordings/train/g64/sender/<code>_neural-qr-sender-<session>.json
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
SHA-256 sender manifest during joining. It is manually assigned on the receiver,
not decoded from the camera. **Verify the visible code and grid before recording,
and keep the same matrix through all four phases.** A wrong assignment cannot
be ruled out from marker detection or capture similarity alone. Review suspect
sessions offline rather than assuming the code authenticates the image.

## Join uploaded recordings by size

After uploading both sides:

```sh
node scripts/corrector-dataset.mjs join-uploads \
  --recordings-root local-data/recordings \
  --out local-data/multires-joined
```

The output must be a new or empty directory. This command discovers recordings,
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
