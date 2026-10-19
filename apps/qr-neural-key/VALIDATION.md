# QR authentication and transport validation

## Physical authentication milestone — user report, 2026-10-17

The user completed a real PC-verifier / phone-key-holder authentication at
**32 repetitions**. The verifier report reviewed in the project conversation
recorded a **17,474-byte proof**, successful verification and result delivery,
approximately **15 minutes 31 seconds** to verification, and **30.7 ms** for the
verification computation itself. The phone's final incoming profile at the PC
used **64-byte chunks and one chunk per ACK**. This establishes a working device
exchange, but its throughput was impractical. The report alone does not establish
the phone's complete profile-change history.

The hosted test sites were subsequently replaced with blank pages; their source
remains runnable here. This paragraph records the user-provided result, not a
new device test performed during repository organization. The original attached
report is not bundled in this checkout. Earlier sections below retain the scope
and status of their respective software validation runs.

## Independent outgoing profiles and adaptation — 2026-10-17

The final `npm test` run passed **223 tests**, with **0 failures and 0 skipped**,
in **5.9498 seconds**. Complete output is saved in
`validation/directional-tests.txt`. `git diff --check` also passed.

The authentication view now configures each device's outgoing DATA separately
from incoming traffic and from its authentication role. Phone and Desktop
presets include different chunk sizes, windows, display timing, ACK holds, and
physical QR widths. Outgoing settings can be applied during an attempt, including
restarting the same message with smaller chunks. The selected 10-, 20-, or
30-minute deadline remains fixed; 20 minutes is the default.

| Coverage | Evidence |
| --- | --- |
| Transport restart | 12 cases exercise the current 68-byte RESTART format, same-payload replacement, strict size reduction, retained deadlines, and completion receipt handling. |
| Adaptive link | Seven new cases cover bounded outgoing fallback, manual updates, next-message growth, and the configured local ceiling. Fallback waits for a completed outgoing window and three retry intervals without additional acknowledged DATA. |
| Authentication integration | 13 cases, including four new cases, cover independent 128-byte phone / 1,024-byte PC directions, real QR image decoding, and a 32-repetition proof transfer. An adaptive 1,024→128-byte case completes without regenerating its proof; a lost completion ACK followed by restart uses the stored receipt. |
| Authentication controller | 18 cases, including six new cases, cover independent device defaults, mid-attempt changes, report fields, physical width, and the fixed configurable deadline. |
| Crypto regression | All 47 crypto and worker regression cases still pass. The neural circuit and compact proof format remain unchanged. |

These checks exercise software behavior using actual QR images and packet-level
faults where specified. They do not establish physical camera throughput or
model the reported screen-size mismatch, moiré, focus changes, or viewing angles.
**Rendered-browser visual QA and physical device authentication have not been
run for this iteration.** The earlier device authentication attempt did not
complete; this update provides directional controls and bounded recovery for
the next measurement, not a claim that the device issue is resolved.

Follow the updated `TESTING.md` checklist at 32 repetitions first: Phone on the
phone, Desktop (Fast) on the computer, 720p capture, and both full QR borders in
view. Export both JSON reports, including configured and effective outgoing
profiles, incoming settings, adaptation events, and final result acknowledgment.

## Compact proof authentication — 2026-10-17

The final `npm test` run passed **194 tests**, with **0 failures and 0 skipped**,
in **5.154 seconds**. The complete test output is saved in
`validation/auth-tests.txt`. `git diff --check` also passed.

The integration adds a separate `auth.html` view while preserving the transport
lab. Its 13 crypto and worker source files are byte-identical to the preserved
compact v2 implementation; see `CRYPTO_SOURCE.md` for provenance. The circuit and
proof format are unchanged.

| Coverage | Evidence |
| --- | --- |
| Crypto regression | 47 copied crypto cases, including five exact Python proof-byte parity fixtures and an actual worker generating a 219-repetition proof while the main thread continues its heartbeat. |
| Authentication session | 14 cases cover explicit pairing confirmation, pinned identities, the higher repetition policy, wrong session/binding/nonce, changed proofs, identity substitution, expiry, cancellation, and duplicate/replay handling. |
| Optical link handoffs | Seven cases cover staged and simultaneous transfers, retained completion receipts, repeated-scan fairness, bounded queues, absolute deadlines, and asynchronous application processing. |
| Full authentication | Nine end-to-end cases use the real proof engine, including a 32-repetition exchange through rendered and decoded QR images and a 219-repetition exchange through multipart packets. |
| Authentication controller | 12 minimal-DOM cases cover reset and late operations, worker failures, stale camera results, invalid settings, expiry, QR presentation invariants, and distinct local-verdict/peer-report states. |

The full-size multipart case verified a **119,249-byte** proof. The QR-image case
verified a **17,474-byte** proof and completed all five application messages:
HELLO, IDENTITY, CHALLENGE, PROOF, and RESULT. A recovery case dropped completion
ACKs for three simulated seconds at each of those five handoffs; the exchange
still completed with one proof generation and one verification.

Negative end-to-end cases reject an altered proof even when its transport hashes
and CRCs are valid, reject a proof for a different nonce, prevent a completed
proof from triggering another verification, and discard a valid verification
result that finishes after cancellation or expiry. These distinguish proof
validation from successful byte delivery.

QR-image tests render and decode actual standard QR pixels. They do not model
physical screens, camera movement, moiré, autofocus, glare, or camera processing.
Controller checks use a minimal DOM harness; **rendered-browser visual QA and
physical device authentication remain unverified in this environment**. No
phone throughput or production cryptographic security claim follows from these
software checks. Use the authentication checklist in `TESTING.md` for the next
device run, starting at 32 repetitions on both devices.

## ACK grouping and typed QR timing — 2026-10-17

The final automated suite passed **105 tests**, with no failures or skipped
tests, in 4.05 seconds. This includes 22 UI controller integration checks.

The sender and simulator expose every ACK group size from 1 through 8 and
accept integer QR intervals from 50 through 2,000 ms. Tests confirm that
manual edits preserve payload size and correction, select Custom immediately,
validate before creating a session, and visibly raise the retry interval when
needed.

Transport tests verify that every group size waits for its complete ACK before
starting the next group, partial ACKs retry only missing current-group chunks,
and non-power-of-two groups deliver the final partial group exactly. UI checks
cover early ACKs respecting the minimum display time, late ACKs advancing
without another delay, identical retries preserving the display timer, and
absolute deadlines taking precedence over that timer.

`git diff --check` also passed. Controller checks use a minimal DOM harness;
rendered-browser layout and physical camera performance still require device
testing. See `TESTING.md` for a focused 2,048-byte-chunk trial at 500 ms with
one chunk per ACK.

## Speed presets — 2026-10-17

The final automated suite passed **87 tests**, with no failures or skipped
tests, in 4.01 seconds. This includes 13 UI controller integration checks.

The updated encoder supports all four correction levels. Actual QR raster
round trips pass at their maximum binary capacities: L 2,953 bytes, M 2,331,
Q 1,663, and H 1,273. DATA framing consumes 18 bytes, leaving a maximum
2,935-byte application chunk at L. Invalid capacity combinations reject before
allocating the transfer payload.

Each preset passes a real QR image-path transfer across a window boundary,
including 26,432 bytes with Fastest. DATA and control packets use independent
correction levels; control defaults to Q. UI controller checks cover preset
selection, manual Custom settings, capacity adjustment, and display pacing.

`npm run benchmark:speed` also recovered exactly 16,384 bytes with the final
ACK for every preset, using seeded drops, duplicates, and pre-encoding CRC
damage. Reports are in `validation/qr-speed-presets.json`.

| Preset | QR transmissions | Dropped | CRC-damaged | Exact + final ACK |
| --- | ---: | ---: | ---: | --- |
| Slow | 376 | 58 | 19 | Yes |
| Medium | 84 | 7 | 2 | Yes |
| Fast | 38 | 3 | 1 | Yes |
| Faster | 20 | 1 | 0 | Yes |
| Fastest | 16 | 1 | 0 | Yes |

Different packet counts lead to different fault locations even with the same
seed. These results establish software recovery, not a measured device speed
ranking. The simulator also charges a display slot to each control packet; its
virtual clock is not a model of the live simultaneous display/camera pipeline.
Visual browser and physical camera testing remain unverified in this environment.

## Initial transport release — 2026-10-16

At this initial transport-only release, the archived neural circuit and ZKBoo
proof implementation were not part of the app.

### Software checks

The final `npm test` run passed **63 tests**, with no failures or skipped tests,
in 4.69 seconds. It covers the binary packet parser,
bounded receive buffers and receipts, selective retransmission, final-ACK loss,
deadlines, cancellation, forged/stale control packets, and full-payload digest
failure. A partial-ACK retry deadlock discovered during window-four simulation
was fixed and retained as a regression case.

QR tests encode and decode every byte value and exercise M/Q/H correction,
small modules, mild blur/noise, perspective and rotation. Worker tests exercise
the actual decoder and simulation entrypoints, including failure recovery.
Distribution checks verify module syntax, local asset paths, UI element
references, and the bundled dependency hashes/licenses.

UI lifecycle checks use a minimal DOM harness. They are not rendered-browser
tests. Visual layout, browser-specific camera behavior, and real devices have
not been verified in this environment.

### Full QR image-path benchmark

Recorded by `npm run benchmark` under Node.js v24.19.0. Full reports are in
`validation/qr-simulation.json`. Every surviving observation passes through
standard QR encoding, raster rendering, and QR decoding before transport
processing; the decoder is not replaced by direct byte delivery.

| Payload | Window | QR transmissions | Dropped | CRC-damaged | Exact + final ACK |
| --- | ---: | ---: | ---: | ---: | --- |
| 1,024 bytes | 1 | 22 | 1 | 0 | Yes |
| 4,096 bytes | 4 | 100 | 20 | 2 | Yes |
| 165,000 bytes | 1 | 3,694 | 553 | 145 | Yes |

The large case also included 500 duplicated observations and delivered the
payload exactly once. Its local computation took 27.30 seconds. Its modeled
channel time was **45.27 minutes** with 128-byte chunks, 350 ms observation
slots, 2.5-second retries, 15% packet loss, and 5% packet corruption. Neither
time predicts camera throughput. Corruption is injected before QR encoding,
so these faults test the transport CRC rather than optical damage correction.

### What device testing must establish

First test the phone's static 28-byte ACK-sized QR against the PC webcam.
Then test the reverse direction, text and 1,024 bytes, and only then larger
payloads and optional windows. See `TESTING.md` for the order and measurements.
The smaller return QR and standard decoding are promising changes to test,
but this build does not yet establish better moiré tolerance or faster transfer
than the parked color bench.

Integrity checks detect accidental damage; they do not authenticate devices.
That initial phase performed no proof generation, key exchange, or encryption.
The current authentication integration is described in `AUTHENTICATION.md`.
