# QR transport, authentication, proof, and report formats

This is a source-derived index of the current experiment formats. The encoder,
decoder, and tests linked below are authoritative. Each layer accepts one current
format; there are no compatibility readers. Transport integrity is not peer
authentication. See [AUTHENTICATION.md](AUTHENTICATION.md) for the trust model and
[TESTING.md](TESTING.md) for device scenarios.

| Layer / use case | Identifier | Source |
| --- | --- | --- |
| Arbitrary-byte multipart optical transfer | `QRT` | [protocol.js](dist/js/protocol.js), [session.js](dist/js/session.js) |
| Paired neural-key authentication messages | `neural-qr-auth-v1`, binary prefix `NQA1` | [envelope.js](dist/js/auth/envelope.js), [session.js](dist/js/auth/session.js) |
| Public identities, challenges, and compact proofs | `NeuralPicnic-POC-MPC3-v2`, proof magic `NPZK0002` | [proof.js](dist/js/crypto/proof.js), [circuit.js](dist/js/crypto/circuit.js), [bytes.js](dist/js/crypto/bytes.js) |
| Live QR transport / scan diagnostics | `schema: "qr-optical-report-v1"` | [app.js](dist/js/app.js) |
| Synthetic QR transport experiment | `schema: "qr-link-report-v1"` | [simulation.js](dist/js/simulation.js) |
| Live authentication diagnostics | `format: "QR-Neural-Key-auth-report-v2"` | [auth/app.js](dist/js/auth/app.js) |

## QRT packets

QRs use binary Byte mode without Base64. All packet multibyte integers are
unsigned and big-endian. Common framing:

| Byte offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 3 | ASCII `QRT` |
| 3 | 1 | Packet kind |
| 4 | 8 | Random transfer identifier |
| 12 | Variable | Kind-specific body |
| Packet length − 4 | 4 | CRC-32 over every preceding packet byte |

CRC uses reflected polynomial `0xedb88320`, initial value `0xffffffff`, and final
XOR `0xffffffff`. The encoded CRC itself is big-endian. Offsets in the next table
are relative to the body at byte 12.

| Kind | Body fields in order | Total packet bytes |
| --- | --- | ---: |
| START = 1 | `totalBytes:u32`, `chunkBytes:u16`, `chunkCount:u16`, `windowSize:u8`, three zero reserved bytes, SHA-256 of complete payload (32 bytes) | 60 |
| DATA = 2 | `index:u16`, chunk bytes | 18 + chunk length |
| ACK = 3 | `base:u16`, `bitmap:u8`, `status:u8`, first eight bytes of full payload SHA-256 | 28 |
| POLL = 4 | `base:u16` | 18 |
| CANCEL = 5 | `reason:u8`, accepted range 0–3 | 17 |
| RESTART = 6 | START body followed by previous transfer identifier (8 bytes); common header carries the new identifier | 68 |

ACK status values are `ready=0`, `progress=1`, `complete=2`, and `stored=3`.
Bitmap bit `j` acknowledges chunk `base+j`. A stored receipt uses zero base and
bitmap and allows a same-payload restart to finish without a second delivery.
RESTART retains the payload digest and attempt deadline, uses strictly smaller
chunks, and cannot enlarge the predecessor's window. See session tests for
ordering and receipt validation.

Limits are 8 MiB per payload, 65,535 chunks, 1–2,935 bytes per chunk, and 1–8
chunks per ACK window. QR error-correction capacity may impose a lower chunk
limit. `chunkCount = max(1, ceil(totalBytes / chunkBytes))`, including empty
payloads. The receiver verifies the complete SHA-256 before delivery.

Transport-only content has no MIME, filename, or content-type wrapper. Text is
encoded as UTF-8; files and test data are raw bytes. The receiving lab offers a
binary download and attempts UTF-8 display. Application framing is independent
of QRT, so other experiments can transmit their own byte formats.

## Authentication envelopes

An envelope is `NQA1` (4 bytes), canonical JSON header length (`u32`, big-endian),
header bytes, then body bytes. The header is limited to 4,096 bytes and the whole
envelope to 524,288 bytes. JSON must be the exact canonical serialization from
`crypto/bytes.js`: sorted keys, compact separators, ASCII escapes, integer numbers
only. Noncanonical whitespace, duplicate keys, unknown fields, and invalid UTF-8
are rejected.

Every header has `protocol`, `type`, and `sessionId`; protocol is
`neural-qr-auth-v1`, and the session ID is 16 bytes as lowercase hexadecimal.

| Type | Additional exact header fields | Body |
| --- | --- | --- |
| HELLO | `deviceId`, `publicKey` | Empty |
| IDENTITY | `deviceId`, `publicKey` | Empty |
| CHALLENGE | `binding`, `challenge` | Empty |
| PROOF | `binding`, `nonce` | Nonempty raw compact proof |
| RESULT | `binding`, `nonce`, `proofDigest`, `accepted` | Empty |

`deviceId` is a lowercase UUID string. `binding`, `nonce`, and `proofDigest` each
encode 32 bytes as lowercase hex; `accepted` is a Boolean. The pairing binding is
derived from the session, both identities, and their ordered roles. The displayed
pairing code is its first 24 hex characters. The authentication view currently
accepts only the 8×8, 16-round, 256-key-bit circuit.

A public document has exactly `protocol`, `circuit`, `minimum_repetitions`, and
`anchor_outputs`. Its circuit has `name: "binary-rotconv-feistel-v1"`, `side`,
`rounds`, and `key_bits: 256`. Anchor outputs are packed bits expressed as hex;
the authentication profile uses 32 bytes. A challenge has exactly `protocol`,
`nonce`, `lock_id`, `action`, and `required_repetitions`. Its text fields contain
1–256 UTF-8 bytes. In this app, `lock_id` is `qr:` followed by the full pairing
binding, and `action` is `authenticate`.

Both repetition fields allow 1–512. The verifier requests the higher of its own
selection and the peer's minimum; proof acceptance uses the retained challenge
and local policy. RESULT is an unauthenticated peer report at the key holder,
while the verifier's local proof result is authoritative.

## Compact proof bytes

The reusable engine also supports circuit sides 4, 8, and 16 and circuit rounds
2–64. These circuit dimensions are unrelated to QR dimensions. For side `s` and
circuit rounds `r`, define `blockBytes = 2*s*s/8`,
`anchors = ceil(256/(2*s*s))`, and
`traceBytes = r*(anchors+1)*s*s/8`. Let `R` be proof repetitions.

| Field, in serialized order | Bytes |
| --- | ---: |
| ASCII `NPZK0002` | 8 |
| Repetitions, big-endian `u16` | 2 |
| Salt | 32 |
| Challenge response | `blockBytes` |
| Packed Fiat–Shamir party choices | `ceil(R/4)` |
| Hidden commitments | `32*R` |
| Two opened tape seeds per repetition | `64*R` |
| Two opened key shares per repetition | `64*R` |
| Second opened party's trace per repetition | `traceBytes*R` |

Each Fiat–Shamir choice is 0, 1, or 2, packed four per byte with the most
significant pair first. Value 3 and nonzero unused trailing bits are invalid.
The two opened parties are choice `c` and `(c+1) mod 3`; the hidden party is
`(c+2) mod 3`. Arrays are grouped by field, then repetition, then opened party
where applicable. The proof omits opened commitments and output shares that
verification reconstructs. Public documents and challenges are external inputs
to verification rather than duplicated inside the proof.

Total length is `42 + blockBytes + ceil(R/4) + R*(160 + traceBytes)`.
For the authentication profile this is `58 + ceil(R/4) + 544*R`: 17,474 bytes at
32 repetitions and 119,249 bytes at 219. Exact-byte fixtures are under
[`tests/fixtures/zk/`](tests/fixtures/zk/). They are deterministic test material,
not live credentials. This remains an unaudited experimental proof protocol.

## JSON reports and interpretation

Reports are diagnostics, not proof files or messages accepted by the receiver.
Current exporters omit raw payload and proof bytes, secret keys, and filenames.
Public fingerprints, hashes, device/browser camera settings, and timestamps may
still be present.

| Report | Main fields and intended use |
| --- | --- |
| `qr-optical-report-v1` | `exportedAt`, `startedAt`, `elapsedMs`, `tab`, `role`, `profile`, `status`, `session`, `camera`, `cameraRun`, `stats`, `lastDisplayed`, `scanCheck`, `events`; compare physical transport and independent scan checks. |
| `qr-link-report-v1` | `mode`, `model`, `status`, `exact`, `confirmationReceived`, `payloadBytes`, `elapsedMs`, `simulatedMs`, `stats`, `sender`, `receiver`, digests, `options`, `qr`, `events`; reproduce software recovery under seeded packet faults. |
| `QR-Neural-Key-auth-report-v2` | `exportedAt`, `startedAt`, `elapsedMs`, `role`, `configuration`, `authentication`, `timings`, `transport`, `camera`, `events`; distinguish proof computation from bidirectional transport. |

Authentication `transport` includes configured and effective outgoing profiles,
the incoming `receiveProfile`, ACK hold, adaptation events, and sender/receiver
snapshots. The sender snapshot describes the current or latest application
message, not necessarily the whole authentication attempt. Export-time
`elapsedMs` can exceed authentication-completion `timings.elapsedMs`.
Duplicate observations often mean repeated scans of the same displayed QR;
they are not a packet-loss estimate. The simulator's `simulatedMs` is a model
clock, while its `elapsedMs` is local execution time; neither predicts camera
goodput. Export both devices' reports to resolve directional behavior.
