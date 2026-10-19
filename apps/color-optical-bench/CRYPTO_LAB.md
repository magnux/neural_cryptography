# Crypto Lab · compact proofs and configurable repetitions

Crypto Lab is a separate page, `crypto.html`, beside Optical Bench. The optical
page retains its Live link and Image lab views. Each page owns its workers and
state. Leaving the optical page stops the camera; leaving Crypto Lab destroys
its temporary key. The shared navigation and visual styles do not couple the
two experiments.

All proof computation runs in a dedicated browser module worker. The canonical
engine and worker are in [`shared/browser-crypto/`](../../shared/browser-crypto/);
`npm run prepare:app` copies them into ignored app-local `dist/js/crypto/` and
`dist/js/crypto-lab/` paths. The UI and reports remain app-specific.
There is no Flask or Python runtime requirement for the app. Serve the prepared
`dist/` over HTTPS, or use the optional `serve.py` for localhost development.
Python with NumPy remains useful for the independent reference tests.

## First run

1. Open Crypto Lab over HTTPS or localhost. Leave the default profile at circuit
   side 8, 16 circuit rounds, key minimum 219, and requested proof repetitions 219.
2. Choose **Run round trip**. The worker generates a key if needed, creates a
   fresh challenge, generates a proof, and verifies it. The verification should
   be accepted and the proof should contain **119,249 bytes**.
3. Run the Python reference check. It verifies a shipped public Python fixture
   and its expected hash in this browser. The fixture contains no secret key or
   deterministic entropy; the full cross-language suite remains available via
   `npm run test:parity`.
4. Run tamper checks. The unchanged proof should pass; modified inputs should
   fail. The checks cover public context changes and damaged proof bytes.
5. Run the benchmark, initially with three repetitions of the measurement. Each
   measurement generates a fresh challenge and proof using the current key.
   This benchmark count is separate from the proof's 219 cryptographic
   repetitions.
6. Add device/browser notes and export the JSON report.

Individual key, challenge, proof, and verification controls are also available
for inspecting the steps. Generating a new key or challenge invalidates previous
results. Advanced circuit settings are experiments, not advertised security
levels. Circuit size is independent of the optical matrix dimensions.

The public test sites are intentionally blank. To use this working copy locally:

```sh
npm run prepare:app
python serve.py --port 8000
```

Open `http://localhost:8000/crypto.html`. Keep that server local unless you
deliberately configure HTTPS for another device; no hosted deployment is needed
for the proof work.

## Adjust proof repetitions without changing the key

The UI separates three controls that have different meanings:

| Control | Meaning |
|---|---|
| Circuit rounds | Number of rounds in the binary neural circuit; changing the circuit requires a new key session. |
| Key minimum repetitions | Enrolled proof floor, set in the advanced profile before generating a key. Default 219. |
| Requested proof repetitions | Per-challenge verifier request, from 1 through 512. Changing it preserves the current key and clears its old challenge and proof. |

For a fast development session, set **Key minimum repetitions** to **32** before
generating the key. Then select requested counts of **32**, **64**, **110**,
**160**, and **219** in turn. Each round trip uses that same key with a fresh
challenge and a fresh proof. These presets describe experiment counts, not
certified security levels. The UI also accepts a custom integer count.

The lab rejects requests below the enrolled floor rather than silently weakening
it. A default key with floor 219 cannot produce a 32-repetition lab proof simply
by moving the requested-count control; explicitly change the key profile and
generate a new key for that experiment. The core API accepts an actual count at
least as large as both the floor and the challenge request.

The display estimates exact proof bytes for the selected circuit and repetition
count. Its idealized soundness exponent is `repetitions * log2(1.5)`, derived
from the repeated protocol's `(2/3)^repetitions` term. It is not a guarantee of
security for this experimental neural construction. For the default circuit,
32 repetitions produce 17,474 bytes; 219 produce 119,249 bytes. The compact
encoding retains the same number of circuit evaluations and proof repetitions.

This is the foundation for a future verifier-owned difficulty policy; it does
not implement recent-device history or automatic count selection. In such a
policy, an earlier full proof would not strengthen a later short proof. The
time of the last full verification should remain separate from subsequent short
successes, and lost history, expiry, or a clock rollback should require the full
count. Each proof must still use a fresh challenge; optical access does not
replace cryptographic soundness. See [CRYPTO_PARITY.md](CRYPTO_PARITY.md) for the
wire format, count binding, and policy limits.

For a repeatable command-line comparison of the same five counts on one key:

```sh
npm run benchmark:crypto
```

The JSON output records actual proof bytes, local engine timings, acceptance,
and rejection under a changed challenge request. It measures neither phone
performance nor optical goodput.

## Key and job lifetime

Secret keys are generated with secure randomness and retained only in the
worker's memory. They are never posted to the page, persisted, or included in
downloads. A public-key fingerprint identifies the current experiment.

Only one job runs at a time. Progress messages come from the worker. Cancel,
reset, navigation, a worker failure, or a job timeout terminates that worker and
clears its key session. Generate a new key before proving again. Public-only
verification needs no key session and can run after a reset. Worker cancellation
is immediate termination, so it also works during verification.

The page stays available for scrolling, progress, and cancellation. Reports
observe animation-frame gaps on the page while a job runs. Hidden/background
intervals are flagged; they are not silently counted as evidence of slow crypto.
Long-task observations are included only where the browser supports them.
These measurements help assess UI responsiveness, but do not directly measure
touch-input latency, battery use, temperature, or the full browser memory cost.

Prove and verify timings measure the corresponding engine call. Verification
uses the same timing boundary in individual runs and benchmarks. The report's
operation duration additionally includes worker startup, input preflight, and
result hashing. Progress updates are limited to about 20 per second plus phase
boundaries so reporting does not overwhelm the main thread.

## Verify on another device

After a successful proof, download both the binary proof (`.bin`) and its public
context (`.json`). The current context format contains the enrolled public key
and expected challenge. Move both files to the other device using your normal
file-sharing method. Open Crypto Lab there, import the context and proof, and
choose **Verify proof**. No secret key needs to be generated or transferred.

Do this once laptop → phone and once phone → laptop. In each direction, the
receiving device should accept the original proof and reject the tamper cases.
The file workflow isolates cross-browser proof compatibility from the optical
transport. It is not enrollment: in the final app, the verifier must already
trust its enrolled public key and its own outstanding challenge.

Context files use one strict current schema. Proof uploads are limited to the
engine's 8 MiB limit. No older context/proof formats are recognized. Proof and
context downloads contain public transcript data; they contain no secret key.

## Device tests to report

Use the same default profile on the laptop and phone. Avoid changing profiles
between timing comparisons.

| Test | Expected result | Report back |
|---|---|---|
| Python reference check | All checks pass | Browser/version and any failure text |
| Round trip, three separate runs | All accepted; 119,249 bytes each at the default profile | Prove/verify times and whether controls remain responsive |
| Same key, floor 32, requests 32/64/110/160/219 | Every proof accepted; key fingerprint unchanged | Proof bytes, prove/verify times, and both repetition counts |
| Request below the enrolled floor | Visible rejection; key preserved | Any unexpected acceptance or key change |
| Three-run benchmark | Every generated proof verifies | Exported JSON, median times, frame-gap measurements |
| Tamper checks | Original accepted; each changed case rejected | Any unexpected acceptance or rejection |
| Cancel during a larger benchmark | Job stops; key session clears; next round trip works | Delay you observe and any stuck controls |
| Public files in both directions | Receiver accepts without a secret key | Device pair, proof hash, acceptance result |
| Switch between labs | Each opens and runs independently | Any camera, worker, or layout issue |

Include phone/laptop model, OS, browser/version, and whether the phone is in a
power-saving mode. Keep the tab visible during timing runs. If you change tabs
or lock the phone, annotate the report and repeat a visible run. Note visible
stutters even when the average timing looks good. The lab's measurements are
actual observations, not estimates derived from a desktop benchmark.

The report retains a bounded recent history of operation outcomes, profiles,
proof sizes/hashes, checks, timing samples, responsiveness, and environment
metadata. It excludes secret keys, proof contents, and full challenge/public-key
documents. User notes are exported as entered, so avoid putting secrets there.
The v2 report separates `keyMinimumRepetitions`, `requestedRepetitions`, and
`usedRepetitions`; the benchmark measurement count remains a separate field.
Cancelled, failed, and timed-out worker requests retain an outcome and a bounded
error-class identifier; raw error messages and stacks are not exported. Reset
clears the report as well as the key session.

## Completion goals and remaining work

The five original steps are outcome guidelines. Static browser code can satisfy
the POC goals; Flask and server-generated enrollment keys are not mandatory.

| Goal | Current status |
|---|---|
| Recover real-device optical messages | Demonstrated by user testing, including 128×128 / 16 colors |
| Transfer roughly 165 KB with drops, duplicates, retransmission, and timeouts | Still an integration/validation goal; successful dense matrices alone do not establish this |
| Exact Python/JavaScript cryptography and tampering rejection | Automated parity and rejection suites |
| Worker execution and measured responsive phone UI | Worker/UI implemented; collect phone evidence with this lab |
| Enrollment, optical handshake, proof transfer, verification, result acknowledgment | Next integrated protocol work |
| Authenticate after server disconnection; reject stale challenges; recover or time out | End-to-end tests still to implement |

Loading this static page does not send keys or proofs to a server. A complete
offline restart/asset-caching guarantee is separate work. Likewise, the core
proof verifier is stateless: it can verify the same valid proof again. Challenge
expiry and one-time acceptance must be implemented in the final session layer.

This is a custom unaudited research POC. Passing parity, tamper cases, or browser
benchmarks establishes functional behavior, not a security level.
