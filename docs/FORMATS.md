# Formats and layer boundaries

Each live layer has one current POC format. Update both devices and regenerate
affected fixtures when changing one. The explicit [offline recording importer](W_TUBE_DATA.md)
converts the two earlier collection rounds into the current dataset contract; it
does not add old-format parsing to the browser or demo server.

**Neural-QR acquisition is now implemented in `capture.html`.** Its
[layout](NEURAL_QR.md) uses asymmetric geometric markers, repeated phase-color
cues, repeated optical size codes and no dense monochrome byte header. Current
collection covers G16/G32/G64/G128 at four rectified pixels per cell.
The collector records random training symbols; the in-band packet codec and
QR-free authentication integration remain pending. The separate runnable
Neural-ZKBoo demo still uses OBM5 data and standard QR ACKs. Neither receiver
has a compatibility fallback to the other format.

| Layer | Current identifier | Canonical documentation / source |
| --- | --- | --- |
| Python optical simulation | `NPM1`, raw symbols, NPZ captures | [Shared Python formats](../shared/python/README.md); separate from browser transport |
| Neural proof statement | `NeuralPicnic-POC-MPC3-v2` | [Crypto API and proof format](../apps/color-optical-bench/CRYPTO_PARITY.md) |
| Compact binary proof | `NPZK0002` | [Python core](../experiments/python-zkboo/README.md), [shared browser core](../shared/browser-crypto/crypto/proof.js) |
| Color optical frame | `OBM5` | [Exact 64-byte header, palette, phases, and FEC](../apps/color-optical-bench/FORMAT.md) |
| Optical content descriptor | Eight bytes, `OC`, version 1, type, length | [OBM5 header](../apps/color-optical-bench/FORMAT.md); descriptor is in the monochrome header |
| Multipart color message | `OLM1` | [Link transport](../apps/color-optical-bench/LINK_TRANSPORT.md), `dist/js/link/protocol.js` |
| QR byte transport | `QRT` | [QR formats](../apps/qr-neural-key/FORMAT.md), `dist/js/protocol.js` |
| Neural-ZKBoo color packet container | `NZM1` | [Turn demo](NEURAL_ZKBOO_DEMO.md), `dist/js/neural-auth/optical.js`; length-delimited OLM1 packet plus random fill, inside OBM5 |
| Neural-ZKBoo authentication envelope | `NZA1`, `neural-zkboo-auth-v1` | [Authentication protocol](../apps/qr-neural-key/AUTHENTICATION.md), [QR formats](../apps/qr-neural-key/FORMAT.md) |
| Optical model bundle | `neural-corrector`, version 1 | [Model/export contract](../experiments/neural-corrector/README.md); untrained or trained status, hashed graphs and fixtures |
| Neural-corrector checkpoint | `optical-decoder-checkpoint`, version 1 | [`checkpoint.py`](../experiments/neural-corrector/checkpoint.py); local state dict and training/split provenance |
| Neural-QR acquisition layout | `neural-qr`, version 2 | [`layout.js`](../apps/color-optical-bench/dist/js/neural-qr/layout.js), [`vision.js`](../apps/color-optical-bench/dist/js/neural-qr/vision.js); optical G16/G32/G64/G128 size codes, 16 colors, four phases, grid-8 references, 4×4 pixels/cell |
| Capture sender / receiver exports | `neural-qr-sender` / `neural-qr-captures`, version 2 | [Multires collection](MULTIRES_COLLECTION.md); public collection code and profile, phase cues, independently saved/uploaded evidence and sender labels |
| Joined Neural-QR evidence | `neural-qr-dataset`, version 2 | [`corrector-dataset.mjs`](../scripts/corrector-dataset.mjs); 4×4-cell CHW inputs, offline labels and tagged origins; [imports/synthetics](W_TUBE_DATA.md) ready, W-tube training pending |
| Established OBM5 training dataset | `optical-decoder-dataset`, version 1 | [`data.py`](../experiments/neural-corrector/data.py), [`corrector-dataset.mjs`](../scripts/corrector-dataset.mjs); hashed NCHW inputs, physical labels, mask, baseline, expected bytes |

## Collector session coordination

`collection_session.json` is temporary server coordination metadata, separate
from the optical format and recording schema. Its public descriptor uses
`neural-qr-collection-session`, version 2, and contains only an opaque ID,
collection code, sender session ID, profile, dataset split and creation time.
No sender symbols, payload bytes or labels are returned to the receiver.

Managed receiver exports include the optional `collectionSessionId`; manually
assigned current-format files may omit it. Successful managed upload saves receiver evidence,
finalizes the matching `.json.tmp` sender, then removes the active descriptor.
All current Neural-QR endpoints use version 2. Earlier recordings can be read by
the explicit offline importer, not by these live endpoints. Use
a new recording directory when updating both devices. Recordings may be up to
800 MiB, within 128 retained captures; current-format pools may contain fewer
samples. JSON serialization and capture-file reading avoid
one whole-file JavaScript string.

## Neural-QR collection association

The sender's collection code is eight hexadecimal CRC32 characters over its
logical training matrix; the full SHA-256 and logical symbols remain in the
sender manifest. The receiver records that code and profile from the server’s active session
(or manual assignment), then observes orientation, grid size and phase from the
camera. An observed grid must match the assignment. Joining checks these values and
manifest consistency, including split ownership. It does not establish that a
manually assigned code matches the pixels actually photographed. This collection
mechanism is not a packet identifier, authenticator or cryptographic key.

The input is normalized RGB evidence at `[3,4G,4G]`, sampled directly from the
camera at the selected size. Joined labels are physical
palette indices; the known phase map relates them to logical symbols. Reference
cells have a separate loss mask. Current training matrices contain random
logical symbols and fixed reference cells; no packet framing, ECC or payload CRC
is implied. The future Neural-QR packet layer must add those checks before
accepting bytes or advancing authentication. The present Python `multires`
model uses the separate earlier 2×2-cell contract. W-tube training is pending;
the offline data pipeline already combines native, resampled and original G64
payload crops with explicit origin tags, clean targets and paired synthetics.

## Content and transport

The color bench's typed content supports raw bytes, UTF-8, PNG, JPEG, and PCM WAV.
Its monochrome header carries the content descriptor; the color payload carries
the content bytes plus the selected coding overhead. A single OBM5 sequence
contains 1–8 logical frames, each of which may have palette phases. The OLM1
multipart layer extends this to larger messages; phase count is not frame count.

QR transport uses standard QR **Byte mode** and obtains decoded binary bytes.
It does not Base64-encode proof bytes or interpret them as UTF-8. The application
protocol segments those bytes into transport packets and provides acknowledgments,
retry limits, and complete-message integrity. Generic phone QR scanners do not
implement this reassembly protocol.

## Cryptography

Proof circuit side/round count, proof repetitions, and optical size/palette are
independent. A larger optical matrix changes channel capacity, not the proof
statement. Both browser apps use the compact v2 proof engine. For the default
side-8, 16-circuit-round profile, 219 repetitions produce 119,249 proof bytes;
32 repetitions produce 17,474 bytes, before application and transport framing.

The public key's minimum repetition count and the verifier's requested count
both constrain proof acceptance. Expiry, replay prevention, pairing, and an exact
expected challenge are session responsibilities; they are not supplied merely
by parsing a valid proof or a CRC-valid transport packet.

## Reports and fixtures

Reports are JSON diagnostics, not wire packets or proof credentials. Consult
the relevant [color testing guide](../apps/color-optical-bench/TESTING.md),
[Crypto Lab guide](../apps/color-optical-bench/CRYPTO_LAB.md), or
[QR testing guide](../apps/qr-neural-key/TESTING.md). Current validation conclusions live in [VALIDATION.md](VALIDATION.md); generated
benchmark reports are ignored local output.

The Python/browser parity manifest has `version: 2` and a `cases` array; its
binary `.bin` files are compact proofs with deterministic **public test secrets**.
See [Python reference and fixtures](../experiments/python-zkboo/README.md).
Only the current fixture encoding is retained in `shared/fixtures/zk/`.

