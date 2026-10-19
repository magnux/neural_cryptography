# Python ZKBoo experiment

This experiment has one canonical NumPy proof engine and a command-line wrapper:

- `neural_zkboo_reference.py`: exact binary rotating-convolution Feistel circuit,
  key creation, challenge validation, compact proof generation, and verification.
- `neural_zkboo_experiment.py`: key files, challenge/prove/verify commands,
  one-time nonce consumption, reports, and optional Python optical simulation.

The CLI was recovered from the supplied project archive and updated to import
the current reference. Its previous embedded proof implementation is not kept.
The current protocol is **NeuralPicnic-POC-MPC3-v2**, with compact **NPZK0002**
proofs and challenge-specific repetition requirements. No earlier proof reader
or format migration is included. This is an unaudited research POC, not the
Picnic standard or an established cryptographic primitive.

## Quick demo

From the repository root, with a virtual environment active:

```sh
python -m pip install -r experiments/python-zkboo/requirements.txt
python experiments/python-zkboo/neural_zkboo_experiment.py --repetitions 32 --camera
python experiments/python-zkboo/neural_zkboo_experiment.py --self-test
python -m unittest discover -s tests/python -v
```

NumPy is the only external dependency. PyTorch is needed only for the separate
neural-dynamics tests; those tests are skipped when PyTorch is unavailable.
A default 8×8 circuit with 16 rounds and 32 proof repetitions produces **17,474
bytes**. The default repetition profile is 219, producing **119,249 bytes**.
These repetition counts are development settings, not claimed security levels.

The default output directory is `neural_zkboo_results/`. It contains the private
and public keys, challenge JSON, `proof.bin`, `summary.json`, and generated
challenge/proof HTML, PNG, and NPZ displays. Keep the private key local. Reports
contain public parameters and timings, not the private key.

## Separate prover and verifier commands

```sh
# Enrollment creates private_key.json and public_key.json; existing keys are not overwritten.
python experiments/python-zkboo/neural_zkboo_experiment.py \
  --zk-action keygen --minimum-repetitions 32 --out local-data/key

# The verifier creates and retains its expected challenge.
python experiments/python-zkboo/neural_zkboo_experiment.py \
  --zk-action challenge --required-repetitions 64 --out local-data/key

# The prover reads its private key, enrolled public key, and received challenge.
python experiments/python-zkboo/neural_zkboo_experiment.py \
  --zk-action prove --out local-data/key

# The verifier needs only the enrolled public key, its challenge, and the proof.
python experiments/python-zkboo/neural_zkboo_experiment.py \
  --zk-action verify --out local-data/key
```

These commands use one local directory for convenience; on separate devices,
copy only the public key, challenge, and proof as required. `--secret-key`,
`--public-key`, `--challenge`, `--proof`, and `--replay-cache` override individual
paths. The verifier must obtain its enrolled public key and outstanding challenge
from trusted local state. It does not read the private key.

The `verify` command atomically records an accepted nonce in `consumed_challenges/`.
The same nonce cannot be accepted again for that lock, even under another action.
The mathematical API verifier does not consume nonces or enforce expiry. The
CLI demonstrates nonce consumption but does not implement challenge expiration,
peer enrollment, or the browser authentication state machine.

## Repetition policy

| Option | Effect |
| --- | --- |
| `--minimum-repetitions` | Enrolled floor for a new key; otherwise `--repetitions`, or 219 |
| `--required-repetitions` | Requirement for a newly generated challenge; otherwise `--repetitions`, or the new-key floor |
| `--repetitions` | Explicit proof count; when omitted while proving, resolve the greater of enrolled and challenge requirements |

Every count is 1–512. Explicit proof counts below either floor are rejected.
The actual proof count is bound into the statement, and changing the challenge
requirement invalidates an existing proof. For a short standalone demo,
`--repetitions 32` sets the new-key floor, challenge requirement, and proof count
together. Existing demo key files must match the requested enrollment policy;
use a fresh output directory when changing that policy.

## Circuit and optical parameters

| Option | Meaning |
| --- | --- |
| `--state-size` | Binary circuit half-side: 4, 8, or 16; default 8 |
| `--zk-rounds` | Circuit rounds: 2–64; default 16 |
| `--size` / `--matrix-size` | Python optical packet side: 8–512; default 256 |
| `--colors` | Python packet palette: 2/4/8/16; default 16 |
| `--frames` | Maximum packet frames: 1–8; default 8 |
| `--captured FILE.npz` | Verify a decoded Python symbol-capture sequence instead of `proof.bin` |

The optical dimensions do not change the proof circuit. The circuit uses a
256-bit secret bias/key vector, public sparse XOR convolutions, AND nonlinear
units, exact quarter-turn rotations, Feistel rounds, and public permutations.
A proof binds the same key to the enrolled anchor outputs and the response to a
fresh nonce, lock identity, action, and required proof count.

The Python NPM1 packet simulator reserves 32 bytes per frame, supports reordering
and identical repeats, and detects corruption using CRC32. It does not implement
browser OBM5, QR framing, real camera decoding, or error correction. See
[shared Python optical formats](../../shared/python/README.md). `--camera` is an
aligned noisy-cell simulation only; `--captured` accepts symbol NPZs, not photos.

## Python API

From the repository root:

```sh
PYTHONPATH=experiments/python-zkboo python - <<'PYTHON'
import secrets
import neural_zkboo_reference as zk

secret = secrets.token_bytes(32)
circuit = zk.ZKBinaryCircuit(side=8, rounds=16)
public = zk.zk_public_key(secret, circuit, repetitions=32)
challenge = zk.zk_new_challenge('demo-door', 'unlock', required_repetitions=32)
proof = zk.zk_prove(secret, public, challenge)
assert zk.zk_verify(public, challenge, proof)
assert len(proof) == zk.zk_proof_size(circuit, 32)
print(f'Verified {len(proof):,}-byte proof')
PYTHON
```

Keys, proof salts, views, and tapes use OS randomness. `--seed` affects only the
camera simulator. Public deterministic fixture entropy must never be used for
runtime proofs or credentials.

## Browser parity

The reference is the same engine used by the browser parity tooling:

```sh
python apps/color-optical-bench/scripts/zk-fixtures.py generate
npm run test:parity
```

See [CRYPTO_PARITY.md](../../apps/color-optical-bench/CRYPTO_PARITY.md) for exact
compact layout, public-key/challenge schemas, fixture entropy, and APIs. The
current shared fixture manifest has `version: 2` and a `cases` array. Each case
records a public test secret and entropy seed, public key, challenge, relative
proof file, SHA-256, and expected circuit/protocol intermediates. The verifier
reconstructs omitted commitments and output shares; every circuit lane remains
present in the proof statement.

The fixture verifier also accepts a minimal manifest of cases containing `id`,
`public`, `challenge`, relative `proofFile`, and optional `expectValid` (default
true). It exits nonzero on a disagreement. Session expiry and nonce consumption
belong to the application and are outside mathematical parity verification.
