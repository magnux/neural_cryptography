# QR Neural Key — authentication and transport labs

A browser-only POC combining **standard black-and-white QR transport** with the
compact neural ZKBoo proof engine. The authentication view generates ephemeral
keys locally, exchanges public identities, requests a fresh proof, and verifies
it on the receiving device. The independent transport lab remains available for
camera alignment, throughput comparisons, and arbitrary-byte transfers.

- [Authentication entry point](dist/auth.html) — serve `dist/`, then open `/auth.html`.
- [QR transport lab entry point](dist/index.html) — serve `dist/`, then open `/`.
- [Wire formats and diagnostic reports](FORMAT.md).

The previous hosted test sites intentionally serve blank pages. Running this
checkout does not depend on those deployments. The earlier color-matrix bench
and its crypto lab are in [the sibling app](../color-optical-bench/).
See the repository [source inventory](../../docs/SOURCE_INVENTORY.md) for recovered
sources and gaps in the original standalone Python experiments. This integration
uses the current compact v2 browser proof engine without changing its circuit or
proof format. The server serves static app files; keys and proofs are processed
locally.

## Start an authentication test

1. Open **Authentication** (`auth.html`) on both devices and allow camera access.
2. Select **Key holder** on one device and **Verifier** on the other. Choose
   **32 proof repetitions on both devices** for a short development run. Set the
   phone to **Phone**, the computer to **Desktop (Fast)**, and request 720p capture.
   These profiles control the QRs each device displays, independently of its role.
3. Start both cameras, then start the key holder's attempt before the verifier's.
   Each device generates its own ephemeral key, then exchanges public identities
   optically.
4. Compare the 24-hex-character pairing fingerprint on both screens and confirm
   it on both devices. Keep each camera pointed at the other screen.
5. If a DATA QR is too dense for the opposing camera, reduce the local outgoing
   chunk size and click **Apply outgoing settings**. The current message can
   restart with smaller chunks without generating another key or proof. Wait for
   the challenge, proof transfer, local verification, and result acknowledgment.
   The verifier's local result is authoritative; the key holder
   displays an unauthenticated peer report.
6. Repeat at the default **219 repetitions**, then reset and exchange roles to
   authenticate in the opposite direction.

Each generated public identity uses the selected repetition count as its minimum.
The verifier requests the higher of its own selection and the peer minimum; a
short run therefore requires selecting 32 on both devices.

The default proof contains 119,249 bytes; the 32-repetition proof contains 17,474
bytes, before transport framing. The selectable 10-, 20-, or 30-minute attempt
deadline includes pairing and transfer; the default is 20 minutes. It is fixed
when the attempt starts. Reset or reload discards ephemeral keys. Repetition
counts are an experimental speed/strength setting, not an audited security-level
selector.
See [AUTHENTICATION.md](AUTHENTICATION.md) for the protocol and
[TESTING.md](TESTING.md) for the device checklist.

## Independent settings for each optical direction

Authentication sends application messages in both directions: the verifier sends
its identity, challenge, and result; the key holder sends its identity and proof.
**Every device therefore needs an outgoing DATA profile**, even when it is the
verifier. A local profile does not configure incoming data; the receiver learns
the peer's chunk and window settings from its transport packets.

| Authentication preset | Useful DATA bytes | DATA correction | Chunks before ACK | QR interval | ACK display hold | Display width |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Phone | 128 | Q | 1 | 350 ms | 750 ms | 100% |
| Desktop (Fast) | 1,024 | M | 4 | 200 ms | 500 ms | 65% |

Initial selection uses viewport size as a convenience, not the cryptographic
role; confirm the intended profile on each device. The physical display width is
independently adjustable from 40% to 100%. Making the PC QR narrower can let the
phone move closer to the PC webcam while keeping the PC code in its own camera's
field of view. Making a DATA chunk smaller reduces its module count and lets the
phone show larger modules. These solve different parts of the alignment problem.
Keep the entire white quiet zone visible in both directions.

**ACKs remain 28-byte control packets at Q correction** regardless of DATA chunk
size. Reducing the chunk does not make an ACK smaller. If even the ACK is
unreadable, adjust physical QR width, distance, focus, lighting, or ACK hold time.

The authentication view can apply outgoing settings during an attempt. A smaller
chunk restarts the same application message with a new transport ID, discarding
partial transfer progress but retaining the exact payload and any generated
proof. Larger chunks take effect at the next message. DATA correction and window
changes also wait for that message unless applied with a smaller-chunk restart. QR interval and ACK hold changes apply immediately.
The selected authentication deadline does not restart.

**Adapt outgoing speed to ACK progress** starts enabled. After three retry intervals without newly
acknowledged DATA bytes, it can reduce the chunk, use one chunk per ACK, and
lengthen display timing. It stops at 64 bytes and permits at most six fallbacks
per message. An unreadable START prompts timing adjustments; reducing DATA
cannot make START smaller. After eight clean windows it may cautiously raise
capacity for the next message, never beyond the locally configured ceiling.
A Phone ceiling of 128 bytes therefore remains 128 even if the reverse direction
uses Desktop (Fast). This is transport adaptation; it does not change proof repetitions
or cryptographic policy. See [AUTHENTICATION.md](AUTHENTICATION.md) for handoff
behavior and [TESTING.md](TESTING.md) for the focused device tests.

## Start a transport-only test

1. Open the app on both devices over HTTPS and allow camera access.
2. In **Scan check**, display the **28 bytes · ACK-sized** sample on the phone.
   Check that the computer webcam reads it repeatedly. Reverse the direction.
3. In **Transfer**, select **Receive** on the phone and start the receiver first.
4. Select **Send** on the computer. For a new camera arrangement, start with the
   **Slow** preset and 1,024 test bytes. For an arrangement already known to work,
   start the speed comparison at **Medium**, using the built-in 4 or 16 KiB payload.
5. Keep each camera pointed at the other screen. The phone must keep displaying
   its acknowledgment after receiving the bytes, until the computer confirms.
6. Export a JSON report on each device. See [TESTING.md](TESTING.md) for the
   diagnostic sequence, including failures.

For facing screens, the phone's **front camera** is often the practical choice:
it can see the computer while the computer sees the phone display. A rear camera
may need a different physical arrangement. Test this geometry before a transfer.

The large QR display includes its white quiet zone. Use **Expand display** if
needed; keep the entire white border visible. QRs use integer pixel scaling.
The camera preview helps alignment; it is not a decoded copy of the payload.

## Three transport lab experiments

| Tab | What it does |
| --- | --- |
| Transfer | Text, deterministic test bytes, or a local file, with acknowledgments and retries. Received bytes can be downloaded; valid UTF-8 can be read in the content box. |
| Scan check | Static text, ACK-sized, or data-sized QRs, without a transfer session. Diagnose each optical direction independently. |
| Simulation | Render and decode real QR images locally, with seeded packet drops, duplicate observations, and corrupted packet bytes. |

Simulation corruption is applied **before QR encoding**. A valid QR therefore
contains a damaged transport packet that its CRC must reject. This exercises
protocol recovery, not camera blur, moiré, glare, or motion. Simulated link time
and local execution time are not phone-to-computer throughput measurements.

## Speed presets and QR capacity

Transfer and Simulation share the following presets. These are settings to
compare, not measured speed ratings. **Fastest** makes the densest code and may
be slower, or unreadable, on a particular camera. Changing an individual setting
switches the preset to **Custom**.

| Preset | Useful bytes per DATA QR | Data correction | Chunks before ACK wait | Minimum QR interval | Retry interval |
| --- | ---: | --- | ---: | ---: | ---: |
| Slow | 128 | Q | 1 | 350 ms | 2,500 ms |
| Medium | 512 | Q | 1 | 250 ms | 2,000 ms |
| Fast | 1,024 | M | 4 | 200 ms | 2,000 ms |
| Faster | 2,048 | M | 8 | 150 ms | 2,000 ms |
| Fastest | 2,935 | L | 8 | 100 ms | 1,500 ms |

Two controls directly below the sender preset allow independent tuning:

- **Chunks before waiting for ACK:** choose any value from 1 to 8. The sender
  sends that many chunks, then waits for their acknowledgment before beginning
  the next group. Missing chunks are retried within the current group. With 1,
  every chunk must be acknowledged before the next is displayed.
- **Milliseconds between QRs:** type a whole number from 50 to 2,000, such as
  375 or 500. This is a minimum display time, including with one chunk per ACK.
  The sender advances only when both the interval and required ACK are satisfied.
  An ACK arriving later than the interval adds no further hold. Reobserving or
  retrying an unchanged QR does not restart its display timer.

Changing either control selects **Custom** while retaining the chosen chunk
size and correction. A longer interval automatically raises a shorter retry
interval to a valid offered value, with an adjustment message. The same controls
are available in Simulation. Display and decode timing depend on the device.
The live scan loop targets a 50 ms interval
and the transfer pump runs every 25 ms; these are scheduling targets, not
guaranteed capture or decoding rates.

The receiver learns chunk and window settings from START. **It does not need the
same preset as the sender.** Data correction is independent of control correction:
ACKs and other control packets stay at Q by default, including with Fastest.
Their small size lets the return path retain stronger correction without making
the DATA QR denser.

All four standard correction levels are available. The limits below are for
binary Byte mode in a maximum-size, version 40 QR (177 × 177 modules):

| Correction | Maximum QR bytes | Maximum useful DATA bytes |
| --- | ---: | ---: |
| L | 2,953 | 2,935 |
| M | 2,331 | 2,313 |
| Q | 1,663 | 1,645 |
| H | 1,273 | 1,255 |

Each DATA packet uses 18 bytes for transport framing. Chunk options respect the
selected correction level; if a correction change lowers capacity, the app
reduces an oversized chunk and shows the adjustment. **Scan check** includes
**Current sender chunk + header** and **Maximum size at selected correction**
samples for comparing dense codes before starting a session. The maximum sample
uses the whole QR capacity, while the current-chunk sample includes the DATA
framing overhead. Choose the sender preset in Transfer, then select matching
correction in Scan check when comparing that preset's density.

For a useful comparison, send the same 4 or 16 KiB payload three times per preset.
Compare successful end-to-end goodput and retries, keeping position, resolution,
lighting, and direction fixed. JSON reports include the selected preset and actual
settings. See [TESTING.md](TESTING.md) for the short device test sequence.

## Transport

The sole current wire format is `QRT`, a binary packet format with a transfer ID
and CRC-32. There are no compatibility readers for older experiments.

| Packet | Role | Default size / QR at Q correction |
| --- | --- | --- |
| START | Payload length, chunk/window settings, full SHA-256 | 60 bytes / version 5, 37 × 37 modules |
| DATA | Chunk index and bytes | 146 bytes for a 128-byte chunk / version 10, 57 × 57 modules |
| ACK | Window bitmap, status, and digest prefix | 28 bytes / version 3, 29 × 29 modules |
| RESTART | Previous/new transfer IDs, same payload hash, strictly smaller chunks | 68 bytes; sent when shrinking an active authentication message |
| POLL / CANCEL | Request current acknowledgment / stop the transfer | Small control packets |

Module counts exclude the four-module white border on every side. Higher error
correction can increase code density; H is an option to test, not an assurance
of better camera decoding. Each device chooses correction for its own display,
with separate settings for DATA and control packets.

The Slow and Medium presets hold one DATA QR until acknowledged and its minimum
display interval has elapsed. Larger groups send 2–8 data QRs at the configured
interval and use cumulative bitmap ACKs. The receiver may emit ACKs during a
group; the sender must confirm that group before advancing. Only missing chunks
are resent. Receiver delivery happens once, after the
**full SHA-256** matches; retained completion receipts allow repeated final ACKs
if the first was missed. The sender's compact ACK carries an eight-byte digest
prefix. These checks detect accidental corruption: **they do not authenticate
a peer or protect against forged messages**.

State, retry counts, and receive buffers are bounded. The transfer limit is
8 MiB and 65,535 chunks; an 8 MiB file needs at least 129 bytes per chunk, so use
256 bytes or more from the offered settings. DATA chunks can reach 2,935 bytes
at L correction.
The UI defaults to a 60-minute absolute deadline and also offers 15 or 30 minutes;
use the same deadline on both devices.
The transport defaults to 20 retries without completing the current window.
One receiver accepts one transfer at a time. Cancellation, expiry, and retry
exhaustion end the attempt; start a new transfer to retry. Live state is transient
and does not survive a page reload. The simulation separately caps payloads at
1 MiB and bounds its iterations and transmissions.

## Run and inspect

Run these commands from `apps/qr-neural-key/`. Serve `dist/` from any static HTTPS
server. No Flask, build step, CDN, or data
relay is required. Source and tests use native JavaScript modules.

For a **same-computer localhost** check, Python's standard library is sufficient:

```sh
python3 -m http.server 8000 --directory dist
```

Open `http://localhost:8000`. A phone visiting a computer's ordinary LAN HTTP
address generally cannot use its camera; use a static HTTPS deployment or a trusted
local HTTPS server for two-device work.

With Node.js available:

```sh
npm test
npm run benchmark
npm run benchmark:speed
```

The benchmark can also be run with `node scripts/benchmark.mjs`. It measures
synthetic local QR processing and protocol recovery, not physical-camera speed.
`npm run benchmark:speed` runs all five presets with 16 KiB, seed 1234, and the
default packet faults, saving `validation/qr-speed-presets.json`. Payload and seed
are fixed, but presets use different packet counts, so fault locations do not
match byte-for-byte. This checks recovery across configurations; its virtual
link timing does not predict which preset will be fastest on real devices.

| Source | Responsibility |
| --- | --- |
| `dist/js/protocol.js`, `session.js` | Packet validation, bounded multipart state, ACKs, retries, integrity |
| `dist/js/profiles.js` | Shared speed presets, QR capacities, and configuration validation |
| `dist/js/qr.js`, `qr-worker.js`, `qr-client.js` | Standard QR encoding/decoding and background camera decoding |
| `dist/js/simulation.js`, `simulation-worker.js` | Reproducible local fault experiment |
| `dist/js/app.js`, `dist/index.html`, `dist/style.css` | Transport lab camera controls, display, received content, reports |
| `dist/js/auth/`, `dist/auth.html` | `AuthSession` message ordering and authentication state, `OpticalLink` transport handoffs, and role views |
| `dist/js/crypto/`, `dist/js/crypto-worker/` | Compact v2 neural circuit and proof engine; `CryptoClient` controls the worker that owns the secret |
| `tests/` | Codec, packet, session, and adversarial tests |

QR dependencies are bundled locally: **Nayuki QR Code generator 1.8.0** (MIT)
and **jsQR 1.4.0** (Apache-2.0). Exact sources, hashes, and wrapper changes are
recorded in `dist/vendor/manifest.json`; licenses ship alongside the code.
The proof engine also bundles its SHAKE dependency and license under
`dist/js/crypto/vendor/`. QR Byte mode carries the original bytes without Base64. A generic phone scanner
may show unreadable text for these binary packets; use this app's receiver.

Files and camera frames are processed locally. JSON reports omit payload content
and filenames. Optical codes remain visible to nearby cameras. The app must be
loaded before disconnecting its server; this version does not promise offline
reload or persistent transfers. Device performance and reliability need the
physical tests described in [TESTING.md](TESTING.md).
