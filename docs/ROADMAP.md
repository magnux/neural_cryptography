# Plan: Neural ZKBoo authentication over color matrices

Planning baseline: **19 October 2026**. All work so far is the exploratory phase
of this research. The immediate goal is a repeatable phone/computer POC that
generates and verifies our existing Neural ZKBoo proofs, with the complete
exchange carried by **color-matrix optical transport**. This is a functional
research demonstration, not a claim of production cryptographic security.

The original five development steps were guidelines. Reuse the current static
browser apps, local key generation, crypto worker, compact proof format, and
authentication flow. No Flask service, server-held keys, new cryptographic
primitive, or comprehensive attack bench is required to reach this POC.

## Planning documents

- [Optical decoder design and training](OPTICAL_DECODER.md): model contract,
  capture data, synthetic examples, training, export, and browser integration.
- [Research connections](RESEARCH_CONNECTIONS.md): relevant neural cryptography,
  cryptanalysis, and proof-friendly circuit literature; later experiments.
- [Use cases](USE_CASES.md), [formats](FORMATS.md), and
  [development](DEVELOPMENT.md): current runnable components and their boundaries.

## What we can reuse

| Component | Current evidence | Role in the next POC |
| --- | --- | --- |
| Neural dynamics experiments | Recurrent CNN, rotating kernels, normalization, Feistel, and surrogate probes | Preserve as research/reference; no architecture redesign on the delivery path |
| Python/browser crypto | Exact binary circuit, compact NPZK0002 proof, configurable repetitions, workers, parity and tamper tests | Keep one canonical proof engine |
| Color optical bench | Detection, rectification, reference colors, palette phases, bounded temporal fusion, RS recovery, and CRC | Add learned classification at the receiver; preserve framing and recovery |
| Color multipart link | OLM1 sender/receiver, selective retries, completion receipts, deadlines, and local fault simulation | Carry authentication envelopes and raw proof bytes |
| QR authentication | Local identities, explicit pairing confirmation, fresh challenge, proof, result, and optical acknowledgments | Reuse the application flow and tests through a color-link adapter |
| Learned decoder | M1 untrained model, ONNX export, dense input/score adapter, and separate worker lab implemented; device checks pending | Establish phone/PC feasibility, then capture and train in M2 |

Single-matrix reception does not establish reliable multipart communication.
The particularly difficult direction is often **phone display to PC webcam**,
including acknowledgments. The learned decoder and integration must be measured
on both directions from the start.

## Smallest useful end-to-end demonstration

1. Load the static application and model assets on both devices; disconnect the
   server/network before starting the attempt.
2. Generate ephemeral keys locally in workers, exchange public identities over
   color matrices, and compare/confirm the pairing fingerprint on both screens.
3. The verifier creates a fresh challenge with its required proof repetitions.
4. The key holder generates the existing compact proof and sends its bytes through
   color multipart transport. No diagnostic payload truth enters decoding.
5. The verifier checks the proof against its saved challenge and pinned identity,
   consumes the challenge once, and sends the result; transport completion is
   acknowledged. Record both devices' outcomes and timings.

The initial scope is **one-direction authentication per attempt**, with roles
reversed in another attempt. The key holder's received result remains a peer
report, not a proof of the verifier's identity. Persistent enrollment, simultaneous
mutual proofs, session-key agreement, and OTPs are later work.

Start with the existing side-8, 16-circuit-round crypto configuration and **32
proof repetitions on both devices**: 17,474 raw proof bytes. This is explicitly a
development profile. Then validate the 219-repetition profile, 119,249 raw bytes,
without changing the circuit. Envelopes and transport add overhead. Keep circuit
rounds, proof repetitions, and optical settings separate; the verifier's policy
and the identity's minimum still govern acceptance.

Use **64×64/16-color DATA** as the first learned-decoder target and evaluate
128×128 when it helps. Keep small **32×32/four-color control matrices** and
independent outgoing settings for each device. Use the existing conservative
phase/hold settings initially, then shorten them based on measured recovery.
256/512 grids, fewer phases, and a particular throughput number are not gates
for the first completed authentication.

## Implementation order and completion criteria

As of **20 October 2026**, M1 software is implemented; its real phone/PC
feasibility gate remains open. See [Decoder Lab](DECODER_LAB.md) and the
[validation record](VALIDATION.md). M2–M5 remain planned. M2 and M3 can proceed
in parallel after M1; integration should not wait for a polished model or dataset.

| Milestone | Work | Completion criterion |
| --- | --- | --- |
| **M1 — Interfaces and browser feasibility** | Define the rectified-raster/model/score contract. Export a tiny untrained 2D U-Net and run it in a worker on the real phone and PC. Identify the reusable authentication-session boundary. | Browser/Python outputs agree within recorded tolerance; model loads locally with WebGPU or WASM; latency, memory, and UI responsiveness recorded at 64/128 targets. Unsupported operators are replaced before training. |
| **M2 — Capture and first trained decoder** | Extend the existing image/live bench with a bounded labeled-capture mode; build synthetic bursts, train a single-frame classifier in plain PyTorch, fine-tune on real captures, export a pinned model. | Failed payloads are retained with independently verified labels. Session-disjoint evaluation improves exact CRC-valid recovery or recovery time over the current decoder at equal observation budgets. Inspect overconfident failures, not only mean cell accuracy. |
| **M3 — Authentication integration** | Extract the reusable QR authentication session/envelope code into one shared component as needed; add a color OLM1 adapter and a simple authentication view. Preserve the QR lab as a reference. | In-memory/fault tests complete identity exchange, pairing, challenge, proof, verification, result, and completion receipts. Test cancellation, expiry, duplicates, invalid proof, and stale challenge. Physical 1 KiB transfers complete with acknowledgments in both directions. |
| **M4 — First color authentication** | Connect the trained decoder to existing symbol recovery and multipart transport; use conservative device-specific profiles. | Complete the 32-repetition exchange on the actual phone and PC, with every application/control message using color matrices. Repeat with roles swapped; export both reports, including failures. A model-assisted receive path is exercised and its use recorded. |
| **M5 — Repeatability and larger proof** | Tune phases, holds, confidence, and retries. Add five-frame fusion only if the single-frame model leaves a measured limitation. | Three consecutive 32-repetition authentications in each role arrangement; one 219-repetition run in each arrangement. Obstruction recovers or reaches a bounded timeout, stale/invalid proofs fail, and an attempt succeeds after server disconnection. Publish device-specific timing and settings. |

If the classical receiver can complete an exchange first, record that useful
integration result while model work continues. If an exported model regresses,
keep an explicit classical-decoder selection available; do not silently claim
that an untested learned path produced the result. Reduce grid size or use a
single-frame model when that reaches M4 sooner. No new cryptographic design is
needed to compensate for optical failures.

### Work allocation and planned source boundaries

| Work | Location / execution |
| --- | --- |
| Model, generator, training, evaluation, export | Model/export exist in `experiments/optical-decoder/`; generator/training/evaluation follow on the project owner's RTX 4090 |
| Capture and replay UI | Extend `apps/color-optical-bench/`; use its existing renderer, detector, and image lab |
| Learned inference and score adapter | Separate M1 Decoder Lab worker and `learned-input.js` exist; connect to the color receiver after training |
| Authentication application logic | Extract from `apps/qr-neural-key/dist/js/auth/` into planned `shared/browser-auth/` where reusable; retain transport-specific adapters |
| Proof generation and verification | Existing `shared/browser-crypto/`, unchanged canonical engine |

The reusable authentication boundary is `auth/session.js` and `auth/envelope.js`;
`auth/link.js` is the current QR adapter. M3 extracts only what the color integration needs.
Model training is planned on the owner's GPU; M1 model/export checks run with CPU
PyTorch here. No GPU training has been performed. Training and capture data stay out of source
control. Record model provenance, export settings, runtime version, and metrics;
package only the selected runtime model/assets when implementing integration.

Use one current format per layer, update both endpoints together when necessary,
and do not add compatibility readers. A receiver-only model should not require
changing OBM5 or the proof format. Reuse the current authentication envelope where
possible; any transport-neutral rename must be coordinated with fixtures, not
maintained as a second parallel implementation.

## Measurement and device-test plan

At every device milestone save reports from **both endpoints**, including failed
attempts. Record:

- Phone/PC, browser, cameras, requested/actual resolution, screen arrangement,
  display size/brightness, and direction; start with 720p as well as the current
  working setting rather than assuming 1080p is better.
- Model hash/runtime/backend, input size, frame count, inference median/p95,
  queue drops, and actual retained-sample limits.
- Matrix dimensions, palette, phase count/hold, grid references, RS parity,
  chunk/window settings, ACK holds, retries, and decoder mode per device.
- Exact recovered application bytes, CRC/RS outcomes, full-message digest,
  elapsed transfer time, and **delivered application bytes per second** including
  framing, capture, phases, inference, control traffic, and retransmission.
- Pairing/challenge/proving/transfer/verification/result timings, total attempt
  time, proof repetitions/bytes, final states, and cancellation/timeout reasons.

Compare classical and learned decoders on the same recorded inputs and the same
observation budgets, then validate physical transfers. Measure both optical
directions separately. Better average symbol accuracy is insufficient if exact
frame recovery or completed-message goodput gets worse.

The earlier roughly **1 kB/s QR** observation is a useful stretch comparison,
not a prerequisite for authentication. A fair comparison requires the same
device arrangement and bidirectional completion criteria. First establish a
repeatable color POC, then improve its latency.

## Exploratory device observations retained as context

- Color matrices reached **128×128/16 colors** with the existing recovery
  mechanisms; 256×256 remained unreliable. Sustained proof-sized bidirectional
  transfer was not established by those single-matrix results.
- Favorable QR transfer tests reached roughly **1 kB/s** in their tested setup.
- A separate 32-repetition QR authentication carried a **17,474-byte proof** and
  completed in approximately **15 minutes 31 seconds**. Local verification took
  about **30.7 ms**; the verifier report showed a final incoming 64-byte chunk,
  one-chunk-per-ACK profile. Its proof-wait interval also included challenge
  delivery and proof generation, so it is not a raw channel-rate measurement.

These observations are not standardized benchmarks. See [VALIDATION.md](VALIDATION.md)
for automated checks and their limits. The hosted sites remain intentionally
blank; planning and repository maintenance do not reactivate them.

## Future research after the functional POC

The [research connections](RESEARCH_CONNECTIONS.md) motivate the following work,
which does **not** block M1–M5:

- A cryptanalytic comparison bench: differential distinguishers, direct secret
  recovery on small instances, alternative-witness searches, and SIMON/LowMC
  structural/cost baselines.
- Adversarial architecture search, independent attacker ensembles, and a separate
  tree parity machine synchronization experiment.
- Further proof compaction or alternative proof-friendly circuits, evaluated
  separately from the current end-to-end transport work.
- Larger matrices, multi-inlet networks, feature-level temporal fusion, Conv3D,
  transformers, and broader device generalization if simpler models leave a
  demonstrated gap.
- Persistent trust, mutual authentication in one session, temporary credentials,
  recency-based repetition policies, proof chaining, and hidden-inverse schemes.

Completing the POC demonstrates functionality and measured device behavior.
Cryptanalytic evaluation remains necessary before stronger security claims;
optical proximity, low surrogate accuracy, or a previous long proof do not
establish them.
