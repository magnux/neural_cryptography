# Shared compact proof engine

This app and the color Crypto Lab use one canonical browser implementation in
[`shared/browser-crypto/`](../../shared/browser-crypto/). It contains the exact
binary circuit, compact prover/verifier, worker operations, and worker client.
The authentication integration does not change the circuit or proof format.

Run `npm run prepare:app` in this app directory, or `npm run prepare:apps` at the
repository root. Preparation copies the canonical modules to ignored
`dist/js/crypto/` and `dist/js/crypto-worker/` outputs. The resulting `dist/` is
independently servable without another site, CDN, or runtime package download.
Edit the shared sources, not these generated copies.

`npm test` prepares the app automatically and runs circuit, primitive, proof,
worker, Python fixture parity, and authentication integration tests. The
canonical fixtures in [`shared/fixtures/zk/`](../../shared/fixtures/zk/) include
public deterministic test keys and entropy; they remain outside served `dist/`.
The browser's reference fixture contains public verification material only.
Live key generation and proving use Web Crypto randomness with no deterministic
seed option.

The SHAKE dependency retains its upstream bytes, checksums, and MIT license in
[`shared/browser-crypto/crypto/vendor/noble-hashes/`](../../shared/browser-crypto/crypto/vendor/noble-hashes/).
The Python reference is in
[`experiments/python-zkboo/`](../../experiments/python-zkboo/README.md).
This experimental implementation has not undergone an independent cryptographic
audit. Only the current `NeuralPicnic-POC-MPC3-v2` / `NPZK0002` format is accepted.
