# Multires recordings: collect by size and upload to the demo server

The collector supports **8, 16, 32, 64, 128 and 256 cells per side**, with new
captures rectified to **2×2 pixels per cell**. The intended `multires` training
profile uses **16 colors and four palette phases**. It records camera evidence;
the receiver never receives sender payloads or labels. Sender manifests provide
the labels during the offline join.

This collector still uses the working OBM5 geometry/header to identify captures.
The header-free [Neural-QR layout](NEURAL_QR.md) remains a separate integration
task. Payload/reference cells and phase grouping supply useful multires data now.
The [multires architecture](NEURAL_CORRECTOR.md#multires-next-architecture) is
specified and prototyped; its production training/browser integration is pending.
The established 4×4-cell training commands must not be used on these 2×2-cell data.

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
restore detail that the camera never captured. Record acquisition failures too.

Keep an entire physical session in **one split**, including every size, payload
and phase from that arrangement. Do not put different phases or neighboring
bursts of the same session into training and validation. The UI locks the split
for an active session, and the offline join checks physical-session and payload
ownership across splits.

## Device workflow

1. Choose the same split on both devices before beginning the physical session.
   An optional device label helps when a browser exposes only a generic model.
2. On the sender choose grid, 16 colors and four phases, then create a random
   sequence. **Upload sender manifest** before creating the next sequence.
3. On the receiver start the camera, choose the pool, and press Record. Automatic
   alignment and the stable-header guard remain active. Retained phase counts,
   diversity and replacement counters stay next to the camera view.
4. Once a phase has enough varied captures, manually advance the sender. Full
   pools keep replacing redundant captures rather than stopping automatically.
5. Pause and **Upload receiver captures**. Wait for the saved-path confirmation.
   A failed upload keeps the captures and permits retry; it never clears them.
6. Clear the saved batch and continue. Changing grid requires a fresh batch, but
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
local-data/recordings/train/g64/sender/<checksum>_decoder-sender-<sequence>.json
local-data/recordings/train/g64/receiver/<checksum>_decoder-captures-<session>-<time>.json
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

The recording format identifiers/version stay unchanged. New recordings add
explicit `pixelsPerCell: 2` and environment/split metadata. Established records
without the sampling field retain their 4×4-cell interpretation. No recording
file is rewritten; the offline join preserves the actual tensor shape.

## Join uploaded recordings by size

After uploading both sides:

```sh
node scripts/corrector-dataset.mjs join-uploads \
  --recordings-root local-data/recordings \
  --out local-data/multires-joined
```

The output must be a new or empty directory. This command discovers recordings,
checks sender/header matches and split ownership, and writes datasets under
`local-data/multires-joined/<train|validation|test>/g<size>/`. Each sample retains
its sampling density, `[3,2G,2G]` input shape, device pair, camera metadata and
physical-session grouping. Source recordings are never deleted or overwritten.
Unmatched captures must be resolved by uploading the missing sender manifest;
the tool does not guess labels from filenames.

The existing four-pixel-per-axis synthetic/training rebuild explicitly rejects
these new 2×2-cell datasets before deleting outputs. Multires dataset generation
and training are the next implementation step. Existing 4×4-cell datasets and
their documented training recipe remain usable.

Browser API references: [track settings](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/getSettings),
[camera labels](https://developer.mozilla.org/en-US/docs/Web/API/MediaDeviceInfo/label),
[browser client hints](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/userAgentData).
