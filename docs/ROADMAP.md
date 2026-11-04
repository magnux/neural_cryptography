# Plan: Neural-ZKBoo authentication over color matrices

Planning baseline: **19 October 2026**. All work so far is the exploratory phase
of this research. The immediate goal is a repeatable phone/computer POC that
generates and verifies our existing Neural-ZKBoo proofs, with the complete
exchange carried by **color-matrix optical transport**. This is a functional
research demonstration, not a claim of production cryptographic security.

The original five development steps were guidelines. Reuse the current static
browser apps, local key generation, crypto worker, compact proof format, and
authentication flow. No Flask service, server-held keys, new cryptographic
primitive, or comprehensive attack bench is required to reach this POC.

## Planning documents

- [Neural corrector design and training](NEURAL_CORRECTOR.md): model contract,
  capture data, synthetic examples, training, export, and browser integration.
- [Neural-ZKBoo device demo](NEURAL_ZKBOO_DEMO.md): implemented turn-based
  authentication interface, model setup and the next physical-device test.
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
| Color multipart link | Existing OLM1 link plus a dedicated stop-and-wait adapter for authentication | Carry authentication messages with an ACK after START and every chunk; preserve final ACK until explicit handoff |
| QR authentication | Local identities, explicit pairing confirmation, fresh challenge, proof, result, and optical acknowledgments | Reuse the application flow and tests through a color-link adapter |
| Neural corrector | U-tube-2x-attn completed training; recorded validation recovered 100% at K=7/8 in both directions; live worker now connects scores to CRC recovery | Test the trained export in the Neural-ZKBoo interface on both physical devices |

Single-matrix reception does not establish reliable multipart communication.
The particularly difficult direction is often **phone display to PC webcam**,
including acknowledgments. The neural corrector and integration must be measured
on both directions from the start.

## Smallest useful end-to-end demonstration

1. Start the application on both devices and wait until their model and camera
   are ready; then disconnect the server/network while keeping both pages open.
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

The implemented demo uses **64×64/16-color DATA**, four permuted phases and the
G64 model; larger matrices require another trained contract. Keep small
**32×32/four-color ACK matrices** and configurable timing on each device.
Start with **95 ms phase display / 100 ms capture**; try longer display holds
when physical measurements show missed phases. The explicit turn control lets
the phone alternate between receiving DATA and displaying its ACK.
256/512 grids, fewer phases, and a particular throughput number are not gates
for the first completed authentication.

## Implementation order and completion criteria

Earlier G64 device reports established phone WebGPU and PC WASM fixture parity.
The **4 November 2026** training report now clears the recorded recovery target:
U-tube-2x-attn recovers every validation candidate at K=7/8 in both directions.
The **Neural-ZKBoo** page now connects corrected scores to live CRC recovery,
turn-based multipart transfer and the shared authentication session. Next,
export the trained checkpoint and run the exchange on both physical devices.
Recorded validation and clean-render integration tests do not yet establish live
transport goodput. See the
[serving and device guide](CORRECTOR_LAB.md) and [validation record](VALIDATION.md).

| Milestone | Work | Completion criterion |
| --- | --- | --- |
| **M1 — Interfaces and browser feasibility** | Define the rectified-raster/model/score contract. Export a tiny untrained 2D U-Net and run it in a worker on the real phone and PC. Identify the reusable authentication-session boundary. | Browser/Python outputs agree within recorded tolerance; model loads locally with WebGPU or WASM; latency, memory, and UI responsiveness recorded at 64/128 targets. Unsupported operators are replaced before training. |
| **M2 — Capture and trained neural corrector** | Capture both directions, generate coherent phase-grouped synthetic observations, and train the full-resolution U-tube family in plain PyTorch. The phase-grouped model has now passed the recorded recovery target; export its trained weights next. | Failed payloads retain independently verified labels. Session-disjoint evaluation improves exact CRC-valid recovery at equal observation budgets. Browser parity, latency and live recovery still need trained-device checks. |
| **M3 — Authentication integration** | Implemented: shared authentication session/envelopes, dedicated color stop-and-wait adapter, and `neural-zkboo.html`. The QR interface remains separate. | Automated identity/pairing/challenge/proof/result and matrix-recovery checks pass; physical ACK behavior still needs device validation. |
| **M4 — First color authentication** | Current next test: load the trained corrector and use the new turn-based interface with 32 repetitions. | Complete the exchange on the actual phone and PC, with every application/control message using color matrices. Repeat with roles swapped; export both reports, including failures. A model-assisted receive path is exercised and its use recorded. |
| **M5 — Repeatability and larger proof** | Tune phase holds and capture cadence from measured K4–K8 recovery, inference latency and ACK readability. | Three consecutive 32-repetition authentications in each role arrangement; one 219-repetition run in each arrangement. Obstruction recovers or reaches a bounded timeout, stale/invalid proofs fail, and an attempt succeeds after server disconnection. Publish device-specific timing and settings. |

The authentication demo deliberately requires the trained phase-grouped DATA
path; only ACKs use the classical receiver. The separate Optical Bench remains
available for classical diagnostics. Record which path produced every result;
do not silently replace a failed model with a different receiver and count it as
a model-assisted success. No new cryptographic design is needed to compensate
for optical failures.

### Source boundaries and remaining validation

| Work | Location / execution |
| --- | --- |
| Model, generator, training, evaluation, export | Implemented in `experiments/neural-corrector/`; phase-grouped training completed on the project owner's RTX 4090 |
| Capture and replay UI | `capture.html` collects bounded samples; canonical Node tools join labels and evaluate recorded inputs |
| Learned inference and score adapter | Corrector Lab worker and `learned-input.js`; the Neural-ZKBoo transport operations now perform progressive phase-grouped inference and CRC recovery |
| Authentication application logic | Shared browser authentication source, prepared into app-local `auth-core/`; separate QR and color transport adapters |
| Proof generation and verification | Existing `shared/browser-crypto/`, unchanged canonical engine |

The reusable authentication boundary is the shared session and envelope code;
the QR `auth/link.js` and color `neural-auth/` modules provide transport-specific behavior.
The owner has completed full phase-grouped training on the RTX 4090; local
implementation checks use CPU PyTorch and Node WASM. The recorded recovery
improvement is reported by the owner; trained live-device authentication remains
the next check. Training and capture data stay out of source control.
Record model provenance, export settings, runtime version, and metrics;
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

Compare classical and neural correctors on the same recorded inputs and the same
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

