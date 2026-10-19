# Locally bundled QR dependencies

No CDN or package registry is contacted by the app. `manifest.json` records exact
source versions, source and bundled SHA-256 hashes, and ESM wrapper changes.

- **Nayuki QR Code generator 1.8.0**, MIT: the official tagged TypeScript source
  compiled with TypeScript 5.9.3 to ES2018, with a default ESM export appended.
  The QR implementation is unchanged. See `qrcodegen-LICENSE.txt`.
- **jsQR 1.4.0**, Apache-2.0: the official npm distribution wrapped in a local
  CommonJS scope and exported as an ES module. The decoder is unchanged.
  See `jsQR-LICENSE.txt`.

The sender uses QR Byte mode, and the receiver reads `binaryData`, not text.
Arbitrary bytes therefore round-trip without Base64, UTF-8, or URL conversion.
Standard phone scanner apps may display these binary packets as unreadable text;
use this app's receiver to reassemble transfers.
