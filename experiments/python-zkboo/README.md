# Python ZKBoo experiment

This directory is the canonical NumPy implementation of the binary rotating-
convolution Feistel circuit and ZKBoo-style proof engine. It provides a Python
API for key creation, challenges, proof generation, and verification, plus the
reference used to check the browser implementation byte for byte. It is an
unaudited research POC, not the Picnic standard or an established cryptographic
primitive.

## Run a small round trip

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r experiments/python-zkboo/requirements.txt
cd experiments/python-zkboo
python - <<'PYTHON'
import secrets
import neural_zkboo_reference as zk

secret = secrets.token_bytes(32)
circuit = zk.ZKBinaryCircuit(side=8, rounds=16)
public_key = zk.zk_public_key(secret, circuit, repetitions=32)
challenge = zk.zk_new_challenge("demo-door", "unlock", required_repetitions=32)
proof = zk.zk_prove(secret, public_key, challenge)
assert zk.zk_verify(public_key, challenge, proof)
assert len(proof) == zk.zk_proof_size(circuit, 32)
print(f"Verified {len(proof):,}-byte proof")  # 17,474 bytes
PYTHON
```

Thirty-two repetitions are a development profile, not a claimed security level.
The `repetitions` argument in `zk_public_key` establishes the enrolled minimum.
`zk_prove` defaults to the greater of that minimum and the challenge requirement.
A caller may request more repetitions, but never fewer than either bound.
Circuit sides are 4, 8, or 16; circuit rounds are 2–64; proof repetitions are
1–512. Circuit dimensions are independent of the optical transport grid, which
can be much larger.

For another Python program, add this directory to the module search path, for
example `PYTHONPATH=experiments/python-zkboo python your_program.py` from the
repository root, then `import neural_zkboo_reference as zk`. The module is an API;
running it directly does not launch a CLI. There is no PyTorch dependency.

## Source provenance and scope

`neural_zkboo_reference.py` is the recovered crypto-only extraction of
`neural_zkboo_experiment.py`. The extraction record identifies its original
complete source with SHA-256:

```
fc7633f08f5aa89127fa7f740bb4c4fc0f8a3156f8bc4f098e76d1eabd8f1a7e
```

The recovered extraction has SHA-256
`f42cbb0b63b647c02a5d4415a51c7a9d8496706426d82e7cd1e351bf96dffb86`; it was moved
into this directory without changing its bytes.

The extraction record states that the constants, helpers, `ZKBinaryCircuit`,
public-key and challenge validation,
statement and tape generation, commitments, Fiat–Shamir challenges, proof
serialization, prover, verifier, and `zk_new_challenge` matched that
experiment. The module documentation and imports were narrowed to the
extracted code. Camera, optical framing, command-line orchestration, file-based
key provisioning, and nonce-consumption policy are intentionally outside this
reference; no readers for earlier formats were added. The current protocol is
`NeuralPicnic-POC-MPC3-v2`, with compact `NPZK0002` proofs.

The original standalone CLI wrappers (`neural_zkboo_experiment.py`,
`neural_key_experiment.py`, `neural_optical.py`, and `test_neural_pocs.py`) and
the original neural-waves notebook were not present in the recovered sources
after workspace maintenance. They are not recreated or represented as included
here. Their earlier descriptions and results are retained as historical context.
This extraction covers the binary proof core, not the original floating-point
CNN experiment.

The reference depends only on Python's standard library and NumPy. Runtime
proof randomness still comes from `secrets.token_bytes`. Fixed randomness is
installed temporarily only by the optical app's `scripts/zk-fixtures.py` while
creating public test vectors, and restored in a `finally` block. The generated fixture secrets
and entropy seeds are public test material and must never be used as real keys.

## Browser parity fixtures

From the repository root, regenerate or verify the checked-in vectors with:

```sh
python apps/color-optical-bench/scripts/zk-fixtures.py generate
python apps/color-optical-bench/scripts/zk-fixtures.py verify apps/color-optical-bench/tests/fixtures/zk/manifest.json
cd apps/color-optical-bench
npm run test:parity
```

The default fixture profile uses 219 repetitions and produces 119,249 bytes.
The compact layout omits reconstructible commitments and output shares, retains
every circuit lane, and binds the actual repetition count into the statement.
Challenges require `required_repetitions`; verification enforces the greater of
that request and the enrolled `minimum_repetitions`. See
[CRYPTO_PARITY.md](../../apps/color-optical-bench/CRYPTO_PARITY.md) for exact
layout and API details. Smaller profiles are development vectors only; this
extraction does not assert a security level for the custom primitive or protocol.

The manifest has `version: 2` and a `cases` array. Each generated case contains
its identifier, public test secret and entropy seed in hexadecimal, enrolled
`public` key, `challenge`, relative `proofFile`, proof SHA-256, and intermediate
expected values. The latter include canonical JSON bytes, public circuit tables,
anchors, response/context, the first repetition's seeds/shares/tapes,
commitments and output shares, AND-trace hashes, and all Fiat–Shamir trits.
Small cases also contain the first repetition's complete packed AND traces.

Intermediates are independently reconstructed from the fixture entropy and
checked against the actual `zk_prove` proof, including its packed challenge
trits, hidden commitments, opened seed/share, and supplied AND trace. Full
commitment and output arrays are independently reconstructed. Entropy is the
continuous stream
`SHAKE256(UTF8("NeuralPicnic parity fixture v2/" + case_id))`; splitting the
stream across `token_bytes` calls does not restart it. Regeneration contains no
timestamps or output-directory-dependent fields.

For JavaScript-to-Python cross-verification, `verify` also accepts a minimal
manifest:

```json
{
  "cases": [
    {
      "id": "javascript-proof",
      "public": {"...": "enrolled public key fields"},
      "challenge": {"...": "expected challenge fields"},
      "proofFile": "javascript-proof.bin",
      "expectValid": true
    }
  ]
}
```

`expectValid` defaults to `true`; set it to `false` for rejection tests. Proof
paths are relative to the manifest. The command returns a compact JSON result
with `ok`, `cases`, `verified`, and `rejected`, and exits nonzero if any actual
verification result differs from its expectation. It uses only public inputs;
the verifier does not load a secret from the manifest. Session-level challenge
expiry or one-time nonce consumption is intentionally not tested by this
mathematical proof verifier.
