# Shared browser cryptography

This is the canonical JavaScript implementation used by the color Crypto Lab
and QR authentication app. Edit it here, then run `npm run prepare:apps` at the
repository root. The preparation script copies these modules into ignored
app-local paths so each app's `dist/` directory remains independently servable.
There is no bundler or runtime dependency download.

- `crypto/`: exact binary circuit, compact prover/verifier, byte utilities, and
  locally bundled SHAKE implementation with its original license.
- `worker/`: the worker boundary, client, operations, and public-only reference
  fixture. The color lab UI and reports remain in that app.
- [`../fixtures/zk/`](../fixtures/zk/): deterministic Python↔JavaScript vectors.
  These public test keys and entropy streams are never live credentials.

The only accepted proof format is `NeuralPicnic-POC-MPC3-v2` / `NPZK0002`.
See [the circuit and proof API](../../apps/color-optical-bench/CRYPTO_PARITY.md)
and [format boundaries](../../docs/FORMATS.md). Native Web Crypto supplies live
randomness; no deterministic RNG option is exposed to browser users.

Both app test commands prepare their generated files automatically. Run root
`npm test` for JavaScript regression tests and `npm run test:parity` for the
NumPy reference cross-check. `npm --prefix apps/color-optical-bench run
fixtures:zk` regenerates the canonical fixtures and refreshes that app's copies;
run `npm run prepare:apps` before serving the other app after regeneration.
