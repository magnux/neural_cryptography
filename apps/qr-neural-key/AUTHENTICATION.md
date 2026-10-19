# QR neural-key authentication POC

This demo connects the compact browser ZKBoo proof engine to the tested QR
multipart transport. Open `auth.html` on both devices. `index.html` remains the
independent QR transport lab for alignment, capacity, and recovery experiments.
Everything after loading the app runs in the browser. The server supplies static
files; it does not generate, distribute, or verify keys.

For exact packet layouts, message fields, compact proof serialization, and JSON
report identifiers, see [FORMAT.md](FORMAT.md).

## What an attempt establishes

One device is the **Verifier** and the other is the **Key holder**. Each generates
an ephemeral neural key locally. After the users compare and confirm a matching
pairing fingerprint, the verifier requests a fresh proof from the key holder.
A successful local verification establishes that the peer supplied a valid POC
proof for the pinned public identity and this attempt's exact challenge.

This is one-direction authentication. The key holder receives the verifier's
result, but that message is an **unauthenticated peer report**, not a proof of the
verifier's identity. Reset both devices and exchange roles to test the opposite
direction. The fingerprint comparison associates public identities with the two
screens you intend to pair; it is an explicit part of the experiment.

No payload encryption, persistent enrollment, remembered trust, or short-session
credential is implemented. Nearby cameras can observe the optical messages.
Transport CRCs and digests detect damage; they are not authentication.

## Browser and transport parts

| Part | Responsibility |
| --- | --- |
| Authentication view | Role selection, pairing confirmation, progress, result, and cancellation |
| Authentication session | Strict message order, pinned peer identity, fresh challenge, expiry, and one-use verification |
| Crypto worker | Ephemeral secret, compact v2 proof generation, and proof verification |
| QR link | Multipart bytes, pacing, acknowledgment display, retransmission, and completion receipts |
| QR codec and camera | Standard black-and-white QR generation and camera decoding |

The secret remains inside the crypto worker. **Reset / discard key** or reloading
discards it. Backgrounding the page also cancels the attempt and stops its camera;
return to the foreground and explicitly restart.
The browser may retain a device UUID; this identifies an installation for the
experiment and is not a credential. Proof bytes travel directly in binary QR
packets rather than as Base64 text. Dependencies are bundled locally.

The single application format is `neural-qr-auth-v1`: an `NQA1` prefix, a bounded
canonical JSON header, and a raw body only for PROOF messages. Headers are limited
to 4,096 bytes and complete envelopes to 512 KiB. The underlying `QRT` packet
format now includes an explicit RESTART packet for reducing DATA chunks during
a message. Only the current formats are accepted; there are no compatibility
readers or migrations.

## Optical exchange

| Step | QR message | What happens locally |
| --- | --- | --- |
| 1 | Verifier → key holder: HELLO | A fresh session identifier and verifier public identity start the attempt. |
| 2 | Key holder → verifier: IDENTITY | The key holder returns its public identity for that session. |
| 3 | Human comparison | Both screens show the same 24-hex-character pairing fingerprint. Confirm it on both devices. |
| 4 | Verifier → key holder: CHALLENGE | The verifier chooses fresh randomness and the required repetition count. |
| 5 | Key holder → verifier: PROOF | The worker creates a compact proof; QR multipart transfer carries its raw bytes. |
| 6 | Verifier → key holder: RESULT | The verifier checks against its saved challenge and pinned public identity, then reports success or failure. |

The pairing binding includes the session, both public identities, and their
ordered roles. It is included in the proof challenge. The verifier keeps its own
expected challenge and repetition policy; received proof data cannot replace
them. A challenge is consumed before verification begins, so the same attempt
cannot accept it twice. Deadlines and cancellation also invalidate late results.
A new attempt uses a new session and nonce.

The attempt deadline is selectable as 10, 20, or 30 minutes, with 20 minutes as
the default. It is fixed when the attempt starts. Each side retains transport
completion receipts across application messages: starting the next message must not discard
the final acknowledgment still needed by the preceding sender. Duplicate scans
must not start another proof or repeat an authentication decision. Queues,
message sizes, retries, and parsing are bounded. A transport failure can end an
attempt without producing a cryptographic verdict; do not treat it as a proof
rejection.

## Repetitions and proof size

For the default 8 × 8 circuit with 16 circuit rounds, compact proof size is
`58 + ceil(repetitions / 4) + 544 × repetitions` bytes:

| Proof repetitions | Raw proof bytes | Intended comparison |
| ---: | ---: | --- |
| 32 | 17,474 | Quick development and alignment test |
| 64 | 34,890 | Intermediate transport test |
| 110 | 59,926 | Intermediate transport test |
| 219 | 119,249 | Default full-size experiment |
| 512 | 278,714 | Upper supported repetition bound |

These are raw proof sizes; the authentication envelope and QR transport add
framing bytes. Circuit rounds and proof repetitions are different parameters.
Increasing proof repetitions repeats the MPC proof experiment; it does not deepen
the neural circuit.

The idealized per-attempt soundness term is `(2/3)^repetitions` for this style of
protocol. It is not a measured security level for this POC. Fiat–Shamir use,
implementation details, the neural relation, repeated attempts, and the complete
protocol also matter. Low counts are useful for development but weaken the
proof; optical proximity is not a substitute for repetitions. The verifier's
request and the public identity's minimum policy both constrain acceptance.

The presets are 32, 64, 110, and 219; Custom accepts integers from 1 through 512.
Each page uses its selection as the minimum for the key it generates. The verifier
requests the higher of its selected count and the peer's public minimum. Select
32 on **both** devices for a 32-repetition run. Reset before changing roles or
repetition policy.

Adaptive repetition counts based on recent authentication are not implemented.
The proposed chaining of old and new challenges, including an XOR construction,
is recorded as a future research idea only.

## Outgoing settings and adaptation

Transport settings belong to the device that **displays** the QR, independently
of whether it is the verifier or key holder. Both roles send application data and
acknowledge incoming data. A phone may need small outgoing chunks while receiving
large chunks from a computer. The peer reads incoming settings from the transport;
the two profiles do not need to match.

Use these starting points with requested 1280 × 720 capture:

| Device preset | DATA | Window | QR interval | ACK hold | Physical QR width |
| --- | --- | ---: | ---: | ---: | ---: |
| Phone | 128 bytes, Q correction | 1 | 350 ms | 750 ms | 100% |
| Desktop (Fast) | 1,024 bytes, M correction | 4 | 200 ms | 500 ms | 65% |

The initial preset is selected by viewport size, not authentication role. Select
it explicitly when testing. The width control accepts 40%–100% and preserves the
white quiet zone. Reducing PC display width may allow the phone to move closer
to the PC webcam while still seeing the entire PC code. Reducing phone chunk
size makes its DATA QR less dense. Neither setting changes the fixed 28-byte ACK
payload; ACKs keep Q correction. A longer ACK hold can help a slower camera read
the existing control code.

**Apply outgoing settings** is available during an attempt:

| Change | Effect |
| --- | --- |
| Smaller chunk | Restart the current application message with the same bytes and smaller chunks; partial transport progress is discarded. |
| Larger chunk | Use the new value for the next application message. |
| DATA correction or window size | Use the new value at the next message, or with a smaller-chunk restart; the restarted window cannot exceed its predecessor. |
| QR interval or ACK hold | Apply immediately to local display scheduling. |
| Physical width | Resize the QR without changing its bytes. |

Restarting transport does not generate a new key, challenge, or proof and does
not reset the selected attempt deadline. The 68-byte RESTART control packet
identifies the previous and new transfer IDs, the unchanged full payload hash,
and strictly smaller chunks. The receiver rejects an unrelated replacement,
retains the original deadline, and preserves completion receipts. If the first
transfer had already completed but its ACK was lost, the retained receipt permits
completion acknowledgment without delivering the application message twice.
These checks protect ordering and duplicate handling; transport packets remain
unauthenticated and can be disrupted by an active observer.

**Adapt outgoing speed to ACK progress** is enabled by default. Three retry intervals without additional
acknowledged DATA bytes trigger a smaller chunk, a one-chunk window, and longer
display holds (at least 350 ms for DATA and 750 ms for ACKs). Reductions follow a
bounded size ladder down to 64 bytes, with at most six fallbacks per message.
If START has never been acknowledged, the link adjusts timing rather than
shrinking DATA; that would not make the START code easier to read. Eight clean
windows can qualify a cautious increase for the next message, capped at the
user's local ceiling. The phone's 128-byte ceiling is independent of the PC's
1,024-byte ceiling. Reports record configured, effective, and incoming settings
and adaptation events so physical runs can distinguish the directions.

Adaptation cannot diagnose whether a missing ACK means unreadable DATA, an
unreadable return code, or temporary misalignment. It is a bounded recovery
experiment, not a guarantee of better throughput. It leaves proof repetitions,
identity binding, one-use challenges, and cryptographic verification unchanged.

## First device run

Use **Phone** on the phone and **Desktop (Fast)** on the computer. Start at requested
1280 × 720 camera resolution and record the actual value. Keep each camera
pointed at the other display, with the entire QR quiet zone visible. A front-facing
phone camera is often easier to arrange opposite a computer webcam. Use the PC's
physical width control before moving the phone so close that it can no longer
see the PC code.

Select cameras and click **Start camera** on each device. Start the key holder
first with **Generate key & wait for verifier**, then start the verifier with
**Generate key & start verifier**. Use 32 proof repetitions on both devices for the
first complete exchange. Compare the entire fingerprint and click
**The codes match · confirm pairing** on both screens. Keep both cameras running
through the result acknowledgment. Repeat with 219
repetitions after the short exchange works. **Export JSON report** on both devices
before resetting; reports contain status, settings, public fingerprints, and
timing/camera/link metrics, not secret keys or proof bytes. See
[TESTING.md](TESTING.md) for the recovery checks and measurements to report.

The automated checks exercise protocol behavior and cryptographic processing.
They do not establish physical camera performance, browser-specific layout, or
production cryptographic security. This remains an experimental implementation.
