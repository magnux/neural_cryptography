# W-tube recordings: collect Neural-QR by size

The current collector uses **Neural-QR version 2**, with **G16, G32, G64 and
G128**, 16 colors, four palette phases and grid-8 references. It rectifies the
camera image directly to **4×4 pixels per cell**. The receiver reads the grid
size and phase from optical cues; sender labels are used only during offline
joining.

This implements the collection milestone for the planned **W-tube**. The current
Python `multires` model is still the earlier 2×2-cell experiment. W-tube training,
conversion of earlier recordings, stronger paired synthetics and browser model
integration are separate next steps. Do not feed these recordings to the current
`multires` training command.

Reload both devices after updating. Current sender/capture/session files and
joined Neural-QR datasets use version 2. Earlier recordings remain untouched,
but the current reader does not accept them. Keep them for the planned offline
import and use a separate directory for new recordings.

## Start the upload-enabled collector

From the repository root:

```sh
npm run prepare:apps
python apps/color-optical-bench/serve.py --port 8000 \
  --recordings-dir local-data/recordings-wtube
```

Open `http://localhost:8000/capture.html` on that computer. For a phone, use the
same server over trusted HTTPS, with your existing trusted certificate:

```sh
python apps/color-optical-bench/serve.py --host 0.0.0.0 --port 8443 \
  --https --cert certs/dev.pem --key certs/dev-key.pem \
  --recordings-dir local-data/recordings-wtube
```

Both devices open `https://YOUR-CERTIFICATE-HOST:8443/capture.html`. Uploads are
enabled only when `--recordings-dir` is supplied. A static host supports downloads
and manual collection assignment. The server serves assets and stores uploaded
recordings; camera processing remains in the browser worker.

## Geometry and collection settings

Four fixed finders determine the perspective transform. Asymmetric bottom-right
patches determine orientation and handedness. Two copies of a coarse black/white
size code identify G16/G32/G64/G128, independently of the expected collection
profile. Repeated color cues identify the current phase. See the exact
[Neural-QR layout](NEURAL_QR.md).

The active collection descriptor supplies the expected profile, collection code
and dataset split. It does not supply pixels or phase labels. The receiver
cross-checks the optically observed size against that assignment; it never
silently changes the assigned matrix because a different size entered view.
Manual assignment on static hosts uses the same cross-check.

| Grid | Rectified crop | 200 MiB budget | 400 MiB budget (default) | 800 MiB budget |
| --- | --- | --- | --- | --- |
| G16 | 64×64 | 32/phase | 32/phase | 32/phase |
| G32 | 128×128 | 32/phase | 32/phase | 32/phase |
| G64 | 256×256 | 32/phase | 32/phase | 32/phase |
| G128 | 512×512 | 8/phase | 16/phase | 32/phase |

These conservative per-phase recommendations cover all four phases within the
128-record limit. Change the byte budget while the batch is empty. A smaller
pool is valid; the current format does not require every session to have the
same sample count.

A G128 record contains about **4.25 MiB** of base64-encoded float RGB and
cell/reference samples. Four phases at 16 samples each take about **272 MiB**,
or **544 MiB** at 32 each, plus metadata and optional PNGs. G64 at 32 samples per
phase is about **136 MiB** plus metadata. Browser memory use exceeds file size;
use smaller pools if the device struggles.

Upload/download serialization emits JSON one record at a time, avoiding a
whole-recording JavaScript string. The offline join also reads capture files in
pieces. The file is still ordinary JSON with base64-encoded float tensors.

The displayed camera-pixels-per-cell estimate measures actual image coverage.
It is distinct from the fixed four rectified pixels per cell. Enlarging a crop
cannot recover detail that the camera never captured; note this estimate when
comparing recordings.

## Device workflow

1. Reload `capture.html` on both devices. On the sender choose the split and grid,
   then create a random matrix.
2. The sender automatically uploads its manifest as `<filename>.json.tmp`.
   After storage succeeds, the server publishes `collection_session.json` with
   the assignment ID, sender physical-session ID, public collection code,
   expected profile, split and creation time. It contains no clean matrix.
3. The receiver loads that assignment automatically. Select budget and pool size,
   start the camera and press Record. Its optical size must match the assignment.
4. Keep each phase static while its pool fills, then advance the four phases
   manually. Full pools continue balanced replacement. Pause when finished.
5. Upload the receiver recording. The server saves it durably, finalizes the
   matching sender by removing `.tmp`, then removes the active descriptor.
6. Clear the saved batch and continue with the next matrix. A changed physical
   arrangement or camera starts a new physical session; changing the dataset
   split requires a new physical session too.

An assignment cannot relabel a populated batch. If it changes or ends during
collection, recording pauses and existing samples remain local. A failed upload
preserves samples and the staged assignment for retry. Identical receiver retries
are idempotent. Abandoning an unfinished assignment clears its descriptor but
retains the `.tmp` sender. The server restores a valid active descriptor after a
restart. No recordings are deleted by these actions.

Keep every size, phase and payload from one physical session in one split.
Neighboring captures or resized copies must not become independent training and
validation sessions. The UI locks the split after use; joining checks session
and payload ownership.

## Admission and representative retention

Green means the current geometry, orientation, size and phase cues passed.
Orange marks tentative geometry with uncertain cues or a mismatched expected
profile. Neither state asserts that payload symbols are correct. Uncertain size
codes, disagreement between repeated codes, or wrong assigned sizes do not enter
the saved pool.

The collector waits at least 500 ms and two observations of the assigned phase
before retaining evidence. A corner displacement exceeding 4% of the matrix
diagonal restarts that guard. A valid phase then fills an initial pool; a robust
reference from that warm-up is held fixed across pause/resume.

Retention targets approximately **70% core, 25% shoulder and 5% moderate tail**
around that reference. Quotas are rounded to pool size. Outliers and near
duplicates are skipped, while randomized replacement refreshes filled bands.
It seeks representative variation, not maximum distance or a mathematically
fitted normal distribution. Selection uses captured pixels, never sender labels.

A wrong matrix can still have coherent geometry and phases. The collection code
is supplied by the assignment, not read optically. Verify the displayed matrix
belongs to the active assignment and keep it unchanged through all four phases.
Size agreement does not identify the payload. Review suspect sessions offline.

## Recording plan and quick device checks

Before a long recording session, check all four grids in both directions:

1. Confirm the displayed detected G agrees with the sender and the crop has the
   expected dimensions. Try rotation as well as the normal orientation.
2. Briefly cover one size-code copy or show a matrix with a different G from the
   assignment. Acquisition should report uncertainty/mismatch and retain no
   samples from that exposure. Uncover it and confirm collection resumes.
3. Capture all four phases, upload, and confirm receiver save, sender finalization
   and assignment clearing. Run a small offline join before collecting many files.
4. Record camera resolution, measured camera pixels/cell, direction and any
   exposure/focus or glare difficulty. Report the stage that remains orange.

Collect fresh physical sessions, payloads and lighting conditions rather than
many near-duplicate views. Prioritize **G64 PC→phone** and **G128 in both
directions**. Front/rear cameras are distinct arrangements. Include easy,
typical and difficult captures without deliberately filling pools with unreadable
frames. Keep an untouched test split when possible.

## Storage, metadata and offline join

The server derives file names from recording identities. For example:

```text
local-data/recordings-wtube/collection_session.json
local-data/recordings-wtube/train/g64/sender/<code>_neural-qr-sender-<session>.json.tmp
local-data/recordings-wtube/train/g64/sender/<code>_neural-qr-sender-<session>.json
local-data/recordings-wtube/train/g64/receiver/<code>_neural-qr-captures-<session>-<time>.json
local-data/recordings-wtube/validation/g128/receiver/...
```

Diagnostic-only exports use `<split>/unknown/receiver/`. The directory stays
outside the served app, with no public recording-download endpoint.

Metadata includes the local browser-installation ID, available browser/platform
information, device class, screen and viewport, optional device label, camera
label/settings and per-frame acquisition measurements. These are browser-scoped
identities and observations, not guaranteed hardware identifiers.

Receiver exports are `neural-qr-captures`, version 2, with `pixelsPerCell: 4`,
normalized CHW crops, raw cell/reference samples, optical cue diagnostics and
assignment metadata. Sender exports are `neural-qr-sender`, version 2, and keep
logical symbols and their hashes for offline labeling.

After completed uploads:

```sh
node scripts/corrector-dataset.mjs join-uploads \
  --recordings-root local-data/recordings-wtube \
  --out local-data/wtube-joined
```

The output directory must be new or empty. Pending `.json.tmp` senders and the
active descriptor are excluded. The join validates code/profile/phase and split
ownership, then writes `<train|validation|test>/g<size>/` datasets. Each joined
`neural-qr-dataset`, version 2, sample retains `[3,4G,4G]` inputs, device metadata
and physical-session grouping. Source recordings are never overwritten.

The full rebuild/training pipeline is not yet ready for these new inputs. Its
managed rebuild is blocked before replacing datasets until the W-tube and
synthetic-pipeline milestone. The upcoming offline importer will handle eligible
older G64 crops and upsampled 2-pixel Neural-QR captures; do not edit their metadata
or mix them into this recording directory manually.
