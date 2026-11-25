# Contributing to Neural Cryptography

Pick a question you can answer with the equipment you have. A library example
that runs in both languages, a phone that consistently fails to read a matrix,
or a model that turns out no better than the baseline are all useful results.
You don't need to reproduce the entire authentication demo to contribute.

## Start with a known result

From the repository root, choose one starting point:

| What you have | What to try | What it establishes |
| --- | --- | --- |
| Node.js 22+ | `npm run test:demo` | Runs the standalone app's software checks without a camera or weights. Simulated reads do not measure optical performance. |
| Node.js 22+, Python 3.10+ and NumPy | In a Python virtual environment, run `python -m pip install -r requirements.txt`, then `npm run test:parity`. | Runs the two proof implementations against the same public fixtures and checks cross-language verification. |
| Two camera devices and a trained export | Follow the [mobile demo setup](docs/NEURAL_AUTH_DEMO.md). Start with G64 DATA and G16 receipts in both directions. | Measures a real exchange, including acquisition, inference, retries and the final acknowledgment. Save a report from each endpoint. |

Before editing, read [AGENTS.md](AGENTS.md) and the [development guide](docs/DEVELOPMENT.md).
The [source inventory](docs/SOURCE_INVENTORY.md) identifies the maintained
modules; files staged by `prepare:apps` are generated copies.

## Open research questions

### 1. Can another application use Neural-ZKBoo without importing a demo?

Design a small JavaScript library for browser/Node and a Python library around
the [canonical JavaScript core](shared/browser-crypto/README.md) and the
[Python reference](experiments/python-zkboo/README.md). A good first PR is a
pair of runnable examples: Python produces a proof for a given statement and
challenge, JS verifies it, then the roles reverse. Include a changed challenge
that both reject. Use the [public vectors](shared/fixtures/zk/manifest.json)
for byte-for-byte tests; fresh live proofs use fresh randomness and need not
be identical. Keep camera, UI, enrollment and challenge-consumption policy out
of the proof API. The libraries are a proposal; the cores already exist.

### 2. What is the best spacing for the Neural-QR color references?

The current G64 layout reserves rows and columns for known colors, leaving
1,568 bytes before packet framing and error correction. Without those internal
references the ceiling is 2,048 bytes. Do the extra 480 bytes survive the camera?
Compare the current grid-8 layout with wider spacing (for example grid-16)
and with no internal grid. Keep the outer palette strips, finders, four phases
and G16 receipt layout fixed. Turning off the existing **local color correction**
switch is a useful control, but it leaves the pilots visible to the model.

For a layout change, update both endpoints and their fixtures, packing and label
masks together. Train or adapt a corrector for each layout. Compare the same
payload, device placement and error-correction budget on held-out sessions in
both directions. Report verified payload bytes per second **including ACKs and
retries**, plus failed acquisitions and exact packet recovery. See the
[layout and capacity table](docs/NEURAL_QR.md#capacity-symbols-camera-pixels-and-bytes)
and [receiver calibration](docs/NEURAL_QR.md#color-calibration-before-inference).

### 3. Which discrete circuits make good use of a proof byte?

Start with the [exact Boolean circuit](shared/browser-crypto/crypto/circuit.js).
Change one design choice, such as the spatial permutation or placement of
nonlinear gates, and compare it to the current circuit with the same key size,
challenge size and proof repetitions. Report input-bit diffusion and output bias
alongside AND-gate count, serialized proof size and verification time.
The floating-point [dynamics experiment](experiments/neural-dynamics/README.md)
is a source of hypotheses, not the circuit the verifier executes.

### 4. Can training discover a useful keyed transformation?

Train a small network with an explicit secret-key input, then freeze and publish
the learned weights. Compare it with an untrained model of equal size on **new
keys as well as new messages**. Report the distribution of bit flips and
per-output bias, not just the average on the training set. This asks whether
learning improves the transformation; it is distinct from training a camera
to recognize colors. If proposing it for the proof, specify an exact discrete
forward pass that both verifiers could reproduce.

### 5. Can continuous neural dynamics produce reproducible bits?

Return to the [recurrent CNN experiment](experiments/neural-dynamics/README.md).
Hold weights and inputs fixed; vary precision, depth and the rule that turns a
real-valued activation into a bit. Where does a small input change spread while
the *same* input still gives the *same* bits across implementations? Report
whole-response agreement as well as diffusion. One unstable output bit matters
if two devices must agree on the exact response.

### 6. What does adversarial training actually teach a neural model?

Study a toy communication task with generated keys and messages, following the
[Alice/Bob/Eve literature](docs/RESEARCH_CONNECTIONS.md#learning-encryption-and-learning-to-analyze-ciphers).
Compare a jointly trained model with one trained without an adversary. Freeze
both, then evaluate on fresh keys and messages with an independent evaluator
that did not participate in training. Report authorized reconstruction error,
the evaluator's score and a simple baseline at the same budget. This is a
confidentiality experiment; success against one evaluator does not establish
security or answer whether the demo authenticates a key.

## Two other contributions that would help now

- **A matched QR comparison.** Send the same proof bytes on the same hardware
  over standard QR and Neural-QR. Report both directions, completed and failed
  runs, time through final ACK, and median and range over repeated trials.
  [Existing device results](README.md#status-and-scope) come from different setups.
- **A device replication.** Repeat the G64 baseline on another camera pair
  before moving to G128. Record camera choice, measured pixels per cell and
  where each run spent its time. This can distinguish slow acquisition from slow
  inference without retraining the model. See the [measurement guide](docs/ROADMAP.md#measurement-and-device-tests).

## Share a result that someone else can reproduce

Open a [GitHub issue](https://github.com/magnux/neural_cryptography/issues) for
a question or result, or a pull request for code, a fixture or a documentation
change. A useful report fits in five lines:

1. **Question:** what did you change, and what did you expect?
2. **Baseline:** which commit, configuration and comparison did you use?
3. **Reproduction:** exact command; public fixture/seed or device and camera setup.
4. **Outcome:** metric, sample count and failures, split by direction or held-out key where relevant.
5. **Interpretation:** what did this test show, and what is still unknown?

If using a coding agent, include the command it actually ran and distinguish
measured output from a prediction. Share public fixtures or aggregate results;
keep real credentials and private captures out of Git. The [format guide](docs/FORMATS.md)
is the contract if you change a serialized message. A transport success, a
50% avalanche rate and a proof that verifies are three different results:
report the one your experiment measured.
