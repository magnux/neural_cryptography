# Reliable optical messages · step 3.3a

The **Live Demo** tab (`link.html`) adds a multipart byte transport above the
existing optical decoder. It supports a local fault simulation and a real
two-device camera transfer. It does not generate keys, pair devices, or perform
neural proofs yet. All computation remains in the browsers; any static HTTPS
host can serve the app. No Flask, enrollment endpoint, or server relay is needed.

## Quick start: local simulation

1. Open **Live Demo → Local simulation**.
2. Keep **165,000 content bytes**, window **4**, and noise seed **1234**. The
   default fault probabilities are **0.15 drop**, **0.15 duplicate**, and
   **0.05 corrupt**.
3. Select **Run simulation**. Success requires complete receiver delivery,
   byte-for-byte equality with the original, and a completion acknowledgment
   received by the sender.
4. Inspect the counts, digests, and report. Repeat with another seed, then try a
   high loss rate or a short session limit to exercise clean failure.

The worker runs the real sender and receiver state machines and the OBM5
monochrome header, Reed–Solomon, palette phase, and symbol codec. It drops entire
display cycles, duplicates decoded packets, and damages headers or payloads.
It does **not** simulate camera images, moiré, motion, color classification, or
neural cryptography. Optical time advances virtually according to advertised
phase holds; wall-clock execution time is a separate measurement. A simulation
that executes in seconds does not predict a seconds-long camera transfer.

The simulation is capped at **1 MiB**. The reusable transport and physical
transfer support messages up to **8 MiB**. Deterministic test bytes are diagnostic
data; protocol identifiers still use browser cryptographic randomness.

## Two-device camera test

Both devices need a screen and a camera. Unlike a one-way bench test, the sender
must also see the receiver's screen to read acknowledgments. For a laptop and
phone, front/built-in cameras with the screens facing each other are a useful
starting arrangement. Check both alignment previews. Keep all finder markers
visible, the pages in the foreground, and the display brightness stable.

1. Open the same app release on both devices over HTTPS. Select **Live Demo →
   Optical transfer** on each.
2. On the receiver, select **Receiver**, choose the camera and requested
   resolution, and start it first. The actual camera resolution may differ from
   the requested resolution.
3. On the sender, select **Sender**, use deterministic test bytes, and set
   **1,024 content bytes**. Select the camera and start the transfer.
4. Confirm the receiver reports a matching complete-message SHA-256 and the
   sender reports delivery after its acknowledgment. Use **Expand display** if
   the matrix needs more screen area.
5. Repeat with **165,000 bytes**. Start with the default 128×128/16-color data
   profile. Use 64×64 if the larger matrix is unreliable; smaller matrices need
   more chunks. At 64×64 with eight four-second phases, allow at least 70 minutes
   before retries and raise the session limits on both devices (for example,
   to 120 minutes). A 256×256 option is experimental, not demonstrated handheld
   capability.
6. Try a brief camera obstruction, then restore alignment. The receiver should
   preserve accepted chunks and the sender should retry missing ones. Also test
   a persistent obstruction: the transfer must stop at its retry limit or
   absolute deadline rather than wait forever.
7. Swap sender and receiver and repeat. Stop and restart before changing the
   matrix profile or camera.

Use a local binary file when you want an independent byte comparison. After a
successful reception, download the received bytes and compare them with the
source, for example with `cmp original.bin received.bin`. The receiver releases
the downloadable payload only after its SHA-256 matches the advertised digest.
That digest is an integrity check, not proof of sender identity.

**Download both reports, especially after failures.** Include device and browser
versions, screen/camera arrangement, requested and actual resolution, matrix
profile, phase count/hold, window size, content size, and any intentional
obstruction in the notes. Useful measurements are the final state, sender and
receiver digests, received/acknowledged chunks and bytes, retries, elapsed time,
and the reported error. A receiver can finish while the sender times out if the
final acknowledgment is never seen; keep both reports to distinguish that case.
Reports exclude raw payload bytes and file names.

## Profiles, capacity, and timing

| Setting | DATA packets | Control packets |
|---|---|---|
| Default matrix | 128×128, 16 colors | 32×32, 4 colors |
| Reference grid / RS parity | 8 cells / 32 bytes | 8 cells / 32 bytes |
| Palette phases | 8 permuted phases | 1 phase |
| Hold per phase | 4 seconds | 1 second |
| Packet capacity | 5,472 bytes | 150 bytes |
| Link content | 5,428 bytes per DATA packet | Largest control is 88 bytes |

Each DATA packet has 44 bytes of link overhead. A 165,000-byte message therefore
needs **31 DATA packets** at the default profile. All eight phases of all 31
packets take 992 seconds, before control exchanges: roughly **17 minutes** for a
full no-loss display. The sender displays complete palette cycles before moving
to the next packet; allow additional time for camera acquisition and retries.
Palette phases add observations, not content capacity.

The default window is four logical chunks. The sender displays a window, polls
for its receipt bitmap, and resends only missing chunks. DATA and acknowledgment
traffic have separate profiles so small control messages stay easy to read.
The UI also exposes a retry margin, retry limit, and total session limit. The
session deadline is absolute: receiving another packet cannot extend it.

## Current wire format

**OLM1 is the only link format.** Its packets are carried as raw content inside
unchanged **OBM5** optical frames. Every link packet occupies one OBM5 logical
frame, potentially displayed in several palette phases. Multipart assembly is
done by the link layer; it does not increase OBM5's eight-frame sequence limit.
There are no previous-version readers, migrations, or fallback parsers. Update
both devices together.

Every packet has a kind, message type, 16-byte session ID, 8-byte message ID,
length, and CRC32. Identifiers are lowercase hex in the JavaScript API and binary
on the wire. Multibyte integers are big-endian. Reserved bytes must be zero.

| Packet | Additional fields | Purpose |
|---|---|---|
| `start` | `totalBytes`, `chunkBytes`, `chunkCount`, `windowSize`, SHA-256 `digest` | Advertise a bounded message and request readiness |
| `data` | Chunk `index`, `data` bytes | Carry one chunk |
| `poll` | Window `base` | Request the current receipt bitmap |
| `ack` | `base`, `bitmap`, `status`, `digest` | Report readiness, progress, or verified completion |
| `cancel` | Numeric `reason` | End a transfer, best effort |

The wire encodes an ACK bitmap in 32 bits, but negotiated windows are limited to
**1–8 chunks**. Message types are `raw`, `hello`, `proof`, `finish`,
`fast-challenge`, `fast-response`, and `result`. Only raw transfer is exposed in
this demo; the other type labels do not implement authentication protocols.

The receiver accepts one active message. It rejects foreign-session packets,
conflicting START metadata, malformed packets, and invalid chunk lengths. A
duplicate chunk cannot overwrite accepted bytes. It delivers only after every
chunk is present and the complete-message SHA-256 matches. The sender accepts
completion only for its own message and digest, after all chunks were sent.

Assembly retains at most an **8 MiB body plus a bounded chunk-seen array**
(at most 131,072 bytes). Camera decoder evidence, displayed frame buffers,
temporary hashing buffers, and the UI's completed download are separate memory
users. Terminal receipts are bounded by count and lifetime; they allow repeated
packets to receive the same completion response without another delivery.
Exactly-once delivery applies within that retained session history, not after
state deletion or restart. This is not a cryptographic replay policy.

CRC and SHA-256 detect damage. Session identifiers route packets. None of these
authenticate a peer or prevent an adversary from constructing packets. The
later handshake must add authenticated context and action-level replay rules.

## Reusable modules and API

| Module | Main exports |
|---|---|
| `dist/js/link/protocol.js` | `encodePacket`, `decodePacket`, `randomId`, `sha256`, `LIMITS`, `MESSAGE_TYPES`, `DATA_OVERHEAD`, `CONTROL_MAX_BYTES` |
| `dist/js/link/session.js` | Async `createSender(payload, options)` and `Receiver` |
| `dist/js/link/optical.js` | `DATA_PROFILE`, `CONTROL_PROFILE`, `getChunkBytes`, `getControlCapacity`, `normalizeProfile`, async `packetToFrames` |
| `dist/js/link/payload.js` | `diagnosticPayload`, repeatable test bytes |
| `dist/js/link/simulation.js` | Async `runSimulation(options, onProgress)` |

`createSender` takes a `Uint8Array` snapshot. Options include `sessionId`,
`messageId`, `messageType`, `chunkBytes`, `windowSize`, `retryMs`, `maxRetries`,
`timeoutMs`, and starting `now`. Choose `chunkBytes` with
`getChunkBytes(profile)` and select retry timing appropriate for the optical
profile; the core state machine has no knowledge of camera timing.

Drive the sender with `next(now)`, transmit a returned packet when non-null,
and feed incoming packets through `receive(bytes, now)`. Use `snapshot()` for
progress and `cancel(now)` to obtain a best-effort cancellation packet.

Create a `Receiver` with timeout, message-size, and terminal-history limits.
Await `receive(bytes, now)` serially. Its result contains `accepted`, an optional
`reason`, `outbound` response packets, and, on first successful completion, a
`delivery` containing `bytes`, `digest`, and the message identifiers. Call
`tick(now)` while idle so an abandoned message reaches its deadline. The
receiver also exposes `snapshot()` and `cancel(now)`. Supply a monotonic
millisecond clock to both state machines.

`packetToFrames` returns renderable OBM5 phase frames. A hash of the exact packet
bytes supplies the optical sequence identifier, so repeated copies can reuse
decoder evidence while different packets remain separate. That identifier is
a routing discriminator, not authentication.

## Verification and next increment

```sh
npm run test:link
npm test
node scripts/link-benchmark.mjs
```

Automated tests cover wire validation, selective retry, duplicate handling,
deadlines, integrity failure, optical codec integration, and fault simulation.
They do not establish physical camera reliability. The two-device tests above
are the remaining acceptance check for the user's devices.

The [current POC roadmap](../../docs/ROADMAP.md) uses this byte channel for local
identity exchange, explicit pairing confirmation, a fresh challenge, the existing
compact Neural ZKBoo proof, and a result/completion acknowledgment. Reuse the QR
authentication session through a color-link adapter and integrate the
[planned learned receiver](../../docs/OPTICAL_DECODER.md). Begin with one-direction
authentication and reverse roles in a new attempt. Mutual proofs in one session,
ephemeral key agreement, fast authentication, and persistent identities are
deferred. They are not implied by a successful transport result.
