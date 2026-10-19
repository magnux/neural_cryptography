# Formats and layer boundaries

Each layer has one current POC format. There are no compatibility readers for
previous experiment formats. Update both devices and regenerate affected
fixtures when changing one.

| Layer | Current identifier | Canonical documentation / source |
| --- | --- | --- |
| Python optical simulation | `NPM1`, raw symbols, NPZ captures | [Shared Python formats](../shared/python/README.md); separate from browser transport |
| Neural proof statement | `NeuralPicnic-POC-MPC3-v2` | [Crypto API and proof format](../apps/color-optical-bench/CRYPTO_PARITY.md) |
| Compact binary proof | `NPZK0002` | [Python core](../experiments/python-zkboo/README.md), [shared browser core](../shared/browser-crypto/crypto/proof.js) |
| Color optical frame | `OBM5` | [Exact 64-byte header, palette, phases, and FEC](../apps/color-optical-bench/FORMAT.md) |
| Optical content descriptor | Eight bytes, `OC`, version 1, type, length | [OBM5 header](../apps/color-optical-bench/FORMAT.md); descriptor is in the monochrome header |
| Multipart color message | `OLM1` | [Link transport](../apps/color-optical-bench/LINK_TRANSPORT.md), `dist/js/link/protocol.js` |
| QR byte transport | `QRT` | [QR formats](../apps/qr-neural-key/FORMAT.md), `dist/js/protocol.js` |
| QR authentication envelope | `NQA1`, `neural-qr-auth-v1` | [Authentication protocol](../apps/qr-neural-key/AUTHENTICATION.md), [QR formats](../apps/qr-neural-key/FORMAT.md) |

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
