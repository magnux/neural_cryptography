# Browser cryptography: compact proof format

The JavaScript engine reproduces the Python `NeuralPicnic-POC-MPC3-v2`
implementation, including complete proof bytes when given the same test entropy.
It is a port of the existing binary rotating-convolution Feistel circuit and
ZKBoo-style proof. The floating-point CNN experiment remains a separate POC.

This step supplies browser-compatible ES modules and cross-language tests.
[Crypto Lab](CRYPTO_LAB.md), added in step 3.2, runs these modules in a dedicated
worker. Provisioning, session policy, and optical proof exchange remain integration
work. The current optical bench keeps its OBM5 format.
Only the current proof format is accepted; there is no migration or old-format
reader.

## Modules

| File | Responsibility |
|---|---|
| `dist/js/crypto/bytes.js` | Python-compatible canonical JSON, domain-separated framing, MSB-first bit packing, strict byte/hex validation |
| `dist/js/crypto/hashes.js` | Native Web Crypto SHA-256 and random bytes; bundled SHAKE-256 |
| `dist/js/crypto/circuit.js` | Public circuit tables, exact rotating taps, clear evaluation/inversion, three-party simulation, two-view replay |
| `dist/js/crypto/proof.js` | Public-key/challenge validation, statements, commitments, Fiat–Shamir selection, prover, verifier |
| `dist/js/crypto/vendor/noble-hashes/` | Unmodified SHAKE dependency closure from `@noble/hashes` 2.4.0, with license, integrity, and source checksums |
| [`../../experiments/python-zkboo/neural_zkboo_reference.py`](../../experiments/python-zkboo/neural_zkboo_reference.py) | Canonical matching NumPy reference; setup and provenance are in its [README](../../experiments/python-zkboo/README.md) |
| `scripts/zk-fixtures.py` | Deterministic public fixtures and Python verification of external proof files |
| `scripts/zk-cross-check.mjs` | Python regeneration check and bidirectional verification with rejection cases |

The browser modules have no Node, DOM, network-fetch, or CDN dependencies. SHA-256
and randomness require native Web Crypto; there is no fallback RNG. Use HTTPS or
localhost in browsers. SHAKE is bundled because the port needs the exact SHAKE
stream used by Python. No ML framework, GPU, build step, or runtime package
installation is required.

## Core API

```js
import { BinaryCircuit } from './dist/js/crypto/circuit.js';
import { PROTOCOL, hex } from './dist/js/crypto/bytes.js';
import { randomBytes } from './dist/js/crypto/hashes.js';
import { publicKey, prove, verify, proofSize } from './dist/js/crypto/proof.js';

// Local demonstration. The later app will separate key provisioning and roles.
const circuit = await BinaryCircuit.create(8, 16);
const secret = randomBytes(32);
const enrolledPublic = await publicKey(secret, circuit, 219);
const challenge = {
  protocol: PROTOCOL,
  nonce: hex(randomBytes(32)),
  lock_id: 'demo-door',
  action: 'unlock',
  required_repetitions: 219,
};

const controller = new AbortController();
const proof = await prove(secret, enrolledPublic, challenge, {
  signal: controller.signal,
  onProgress: ({ phase, completed, total }) => {
    // phase is 'simulate' or 'complete'. Crypto Lab forwards this from its worker.
  },
});
const accepted = await verify(enrolledPublic, challenge, proof);
const expectedBytes = proofSize(circuit, 219); // 119249
```

`prove` returns a `Uint8Array`, throws on invalid input, and honors cancellation
between repetitions. By default its repetition count is the greater of the
enrolled public key's `minimum_repetitions` and the verifier challenge's
`required_repetitions`. Its optional `repetitions` setting may request a higher
actual count, but cannot fall below either minimum.
`verify` takes only public information and returns `false` for malformed or
inconsistent proofs. It accepts 1–512 proof repetitions, subject to both minimums.
Callers must supply the trusted enrolled key and
expected challenge; neither should be adopted from an incoming proof packet.

Challenges have exactly five fields: protocol, a 32-byte lowercase-hex nonce,
lock ID, action, and integer `required_repetitions` from 1 through 512. Lock ID
and action contain 1–256 UTF-8 bytes. Canonical JSON
matches Python's sorted, compact, ASCII-escaped encoding, including astral
characters. Lone Unicode surrogates are rejected in challenge text. Proving and
verification copy caller-owned key/proof bytes and public documents before
asynchronous work, including Node Buffer inputs.

The core proves a statement; it does not remember used nonces, expire challenges,
identify devices, or authorize a lock action. Those checks belong to the later
session controller. A valid proof can still be verified again by this stateless
API. Temporary secret buffers are cleared where practical, but JavaScript does
not guarantee secure erasure or constant-time execution.

## Circuit and proof bytes

Circuit side is 4, 8, or 16 and rounds range from 2 to 64. Circuit dimensions are
independent of the optical transport's matrix size, palette, phases, and frame
count. An optical matrix transports serialized proof bytes; enlarging it does
not change the circuit or proof statement.

For side `s` and circuit rounds `d`:

- A circuit block contains `2*s*s` bits, or `B = 2*s*s/8` bytes.
- The number of anchor blocks is `A = ceil(256/(2*s*s))`.
- An execution has `L = A+1` lanes: the anchors and the challenge block.
- Each repetition records `G = d*L*s*s` AND gates and `G/8` trace bytes per party.

Only exact Boolean operations are used. The three rotated convolution branches,
public state permutation, and key selectors match Python. Three simulated
parties use XOR shares of one common key across all lanes. Fiat–Shamir selects
two adjacent views per repetition. The verifier reconstructs the first opened
AND trace and uses the adjacent supplied trace. It reconstructs both opened
commitments and their output shares. The hidden party's output is the XOR of
the expected public output and the two opened outputs. Public key, challenge,
response, and the actual proof repetition count are bound into the statement
hash; all three commitments and output shares are bound into the Fiat–Shamir
challenge hash. The verifier checks that the reconstructed transcript produces
exactly the serialized trits.

Proof fields, in serialization order:

| Field | Bytes |
|---|---:|
| Magic `NPZK0002`, repetition count `t`, salt | 8 + 2 + 32 |
| Challenge response | B |
| Fiat–Shamir trits, packed two bits each | ceil(t/4) |
| Hidden party's commitment per repetition | 32*t |
| Two opened seeds per repetition | 64*t |
| Two opened key shares per repetition | 64*t |
| Second opened AND trace per repetition | (G/8)*t |

Integers in framing and the proof header are big-endian; packed bits are MSB
first. Trit values are 0, 1, or 2; the encoding rejects 3 and requires unused
padding bits to be zero. Seeds and salt are freshly generated for every proof.
Allocation is bounded by the profile and the 8 MiB proof limit. Simulation runs one repetition
at a time and retains packed traces, avoiding a full unpacked repetition tensor.

The general size is `42 + B + ceil(t/4) + t*(160 + G/8)`. For side 8 and
16 circuit rounds, this becomes `58 + ceil(t/4) + 544*t`. The default 219-repetition
proof is **119,249 bytes**, 27.62% smaller than the previous 164,746-byte encoding.
It retains all three circuit lanes, including the challenge-derived response;
this change removes reconstructible transcript data without removing circuit
work or repetitions. There is no reader for the previous proof format.

| Proof repetitions | Default-circuit proof bytes | Idealized repeated-protocol soundness term |
|---:|---:|---:|
| 32 | 17,474 | about 2^-18.7 |
| 64 | 34,890 | about 2^-37.4 |
| 110 | 59,926 | about 2^-64.3 |
| 160 | 87,138 | about 2^-93.6 |
| 219 | 119,249 | about 2^-128.1 |

The last column is only `(2/3)^t`, not a security rating for this custom neural
primitive or its Fiat–Shamir instantiation. Fewer repetitions make development
faster and also weaken the soundness amplification. Circuit rounds control
the neural computation; proof repetitions control repeated MPC-in-the-head
checks. They are separate parameters.

## Per-challenge repetition policy

The enrolled minimum is a hard floor. A verifier can request a higher count in
each fresh challenge; the prover cannot lower that count, and changing it
invalidates a previously generated proof. Using a lower challenge count with
the same key is possible only if that count remains at or above the enrolled
minimum. Raising or lowering that enrolled floor changes the public key document
and therefore requires an explicit new enrollment decision.

There is no automatic recency policy in this iteration. A later session
controller could choose challenge counts from trusted verifier-side history.
It should key its history by the trusted enrolled public key and verifier and
retain the time of the last successful **full-strength** verification;
successful short proofs must not refresh that timestamp indefinitely. Missing
history, expiry, or a clock rollback should require the full configured count.
A previous strong proof does not strengthen a subsequent short proof: the short
proof still has its own weaker forgery bound. This is not equivalent to a short
authentication tag under an established secret session key.

Every proof is generated afresh for its challenge and actual count. Do not trim,
append to, or reuse earlier proofs to change their repetition count. Fresh nonce
generation, challenge expiry, one-time acceptance, and policy history remain
session-layer responsibilities. Optical transport provides no assumed secrecy.

## Verification

From the repository root, using Node 22+:

```sh
npm test
npm run test:crypto
npm run test:parity
```

`npm test` includes the optical regression tests and all JavaScript crypto tests.
It reads committed Python fixtures and does not need Python or NumPy. The full
cross-language gate needs Python with NumPy; select an interpreter with `PYTHON`
if needed. It generates fresh files in the OS temporary directory, compares them
to committed fixtures, creates JS proofs with native randomness, asks Python to
verify them, and checks rejection of changed nonces, changed actions, and damaged
traces. Temporary files are removed after the run. Set `TMPDIR` to a writable
directory in restricted environments.

Four public fixture profiles cover all supported circuit sides and Unicode:

| Side | Rounds | Repetitions | Proof bytes |
|---|---:|---:|---:|
| 4 | 2 | 3 | 635 |
| 8 | 4 | 5 | 1,340 |
| 16 | 3 | 4 | 1,515 |
| 8 | 16 | 219 | 119,249 |

Tests compare canonical JSON, circuit tables, anchors, outputs, MPC traces,
random tapes, commitments, statement hashes, all Fiat–Shamir choices, and the
entire serialized proof. Additional checks exercise malformed schemas, proof
component corruption, wrong contexts and repetition policies, fresh randomness,
cancellation, and mutation of caller-owned inputs during asynchronous execution.
Deterministic entropy exists only in test scripts, never as an option in the
served prover. All fixture keys and entropy are public and unsuitable for use
as credentials.

The checked-in `validation/crypto-step31.json` records this iteration's actual
test results. Timings are development-host Node measurements, not mobile browser
benchmarks. Step 3.2 adds real worker tests and a browser interface for collecting
device measurements; see `CRYPTO_LAB.md` for their scope. Phone responsiveness and
camera proof transfer require device testing. Byte parity demonstrates faithful implementation; it
does not establish security of the custom primitive or the proof construction,
and no post-quantum security level is claimed.
