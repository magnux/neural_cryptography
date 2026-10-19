# Compact proof engine provenance

The browser engine in `dist/js/crypto/` and worker components in
`dist/js/crypto-worker/` were copied from the preserved optical bench source
commit `ad7ec85423a023c207d4bf0c5f7a1ef962260b52` for this standalone QR app.
The worker directory was named `crypto-lab` in that project. Its relative engine
imports work unchanged here. The authentication integration does not change the
neural circuit, compact v2 proof encoding, or proof verification algorithm.

The recovered optical bench and Crypto Lab are available in
[`../color-optical-bench/`](../color-optical-bench/). See the repository
[source inventory](../../docs/SOURCE_INVENTORY.md) for the exact recovered sources
and missing original standalone Python files. The source commit above belongs
to the previous hosted app repository, not this repository's commit history.
This checkout bundles the browser implementation so deploying or running it does
not depend on another site.

The copied circuit, primitive, proof, worker, and Python parity tests run with
`npm test`. Test fixtures include deterministic test secrets and entropy streams;
they live outside the served `dist/` directory. The browser's reference fixture
contains public verification material only. Live key generation and proving use
Web Crypto randomness, with no deterministic-seed option.

The SHAKE dependency retains its upstream files, checksums, and MIT license in
`dist/js/crypto/vendor/noble-hashes/`. This experimental implementation has not
undergone an independent cryptographic audit.
