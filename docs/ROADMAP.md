# Roadmap: from an optical proof of concept to reproducible research

**Working baseline.** An offline prover and verifier have completed a
Neural-ZKBoo exchange over Neural-QR. In the [two-phone G64 run](NEURAL_AUTH_DEMO.md#acknowledgments),
the 32-repetition proof was 17,474 bytes; with G16 receipts the verifier
finished in 129.3 seconds and the prover in 130.2. Those times include the
exchange, not just proof computation or inference. The
[earlier asymmetric run](../README.md#status-and-scope) used G128 from computer
to phone and G64 on the return path. Neither result establishes a security
level or a general speed advantage over standard QR.

## What to work on next

| Priority | First result worth sharing | Where to start |
| --- | --- | --- |
| Make the result repeatable | Repeat the G64 exchange on another device pair. Publish comparable times and failures in both directions, then run the standard-QR baseline with the same proof bytes. | [Mobile demo](NEURAL_AUTH_DEMO.md) and [measurement guide below](#measurement-and-device-tests) |
| Make the components reusable | Run headless Python and JavaScript examples that verify one another’s proofs through small public APIs. Keep one implementation per language and the existing wire format. | [Library research brief](../CONTRIBUTING.md#1-can-another-application-use-neural-zkboo-without-importing-a-demo) |
| Improve delivered bytes per second | Compare wider internal reference spacing and no internal references with grid-8. A higher-capacity layout advances if it delivers more verified bytes after retries on a real link. | [Optical experiment](../CONTRIBUTING.md#2-what-is-the-best-spacing-for-the-neural-qr-color-references) |
| Exercise larger proofs | At a fixed circuit and transport profile, record proof bytes, proving time, transfer time through final ACK, and verification time as repetitions increase from 32 to 219. Include timeouts and partial transfers. | [Proof reference](../experiments/python-zkboo/README.md) and [formats](FORMATS.md) |

Parallel to these engineering priorities, researchers can investigate
[discrete, learned, continuous and adversarial neural models](../CONTRIBUTING.md#open-research-questions).
They need not change the authentication protocol to test an idea. For precedents
and distinctions between these questions, see the [research background](RESEARCH_CONNECTIONS.md).

## What is already available

- The [browser proof core](../shared/browser-crypto/README.md) and its Python
  reference share public parity fixtures. Proof repetitions, circuit rounds and
  optical grid sizes are independent controls.
- [Neural-QR](NEURAL_QR.md) supports header-free acquisition at G16/G32/G64/G128.
  The [collection pipeline](W_TUBE_DATA.md) joins real captures to sender
  labels and generates paired synthetic bursts.
- [W-tube and VW](NEURAL_CORRECTOR.md) take four-phase camera bursts and predict
  palette symbols. Training, static ONNX export and local browser inference
  are implemented. Live use needs a trained export; a simulation is not a
  substitute for a camera result.
- The [standalone app](NEURAL_AUTH_DEMO.md) handles DATA, REQUESTs and final
  receipt over Neural-QR. The [bench](NEURAL_ZKBOO_DEMO.md) and
  [standard-QR app](../apps/qr-neural-key/README.md) remain available for comparison.

## Design constraints for experiments

- **Keep four anchors.** The asymmetric bottom-right finder retains its square
  localization core; size bits live in separate patches and do not change its
  orientation meaning.
- **Preserve optical independence.** The expected collection profile only
  cross-checks size. It never fills uncertain code bits. Phase cues are compared
  to measured color references from the same exposure.
- **Make packet grouping explicit.** Size and phase do not identify a packet.
  Begin with stop-and-wait and bounded per-phase pools, replace stale evidence
  across transitions, and advance only on valid full packet metadata.
- **Fit controls to real capacity.** G16 is the smallest supported profile;
  include references, complete identifiers and ECC when determining whether a
  control fits. The mobile demo’s G16 digest receipt uses one exposure; DATA
  uses four palette phases.
- **Bound long transfers.** Configure deadlines from message size and tested
  throughput with hard limits; duplicates must not extend them forever.

## Byte integrity and authentication gates

Neural-QR has no separately decoded header CRC. For DATA, require ECC
recovery, packet CRC, length/schema validation, full identifiers and sequence
matching, and a complete-message digest before application delivery. Never use
sender diagnostic truth to recover bytes. Neural confidence, CRC and physical
proximity are not authentication.

Keep pairing and trust policy visible in each experiment: automatic POC pairing
is not an independent identity check. Preserve the applicable checks for
fresh challenge binding, minimum repetitions, consumed/stale challenges and
tampered proofs. Keep final result delivery distinct from local proof verification.
Exercise these cases through the new transport alongside cancellation, dropped
ACKs, duplicate frames, interrupted capture and bounded timeout behavior. Passing
these tests validates implementation behavior; it does not replace cryptanalysis.

## Measurement and device tests

Save compact reports from **both endpoints**, including failures. Record:

- Device/browser, camera, actual resolution/frame rate, direction, physical
  placement, display size/brightness and relevant exposure/focus settings.
- Model hash, input contract, backend, inference median/p95, queue/buffer bounds
  and which processing path produced a recovered packet.
- Optical profile, phase dwell, capture interval, unique packet and ACK counts,
  ECC/CRC failures, acquisition time and packet-transition retries.
- Delivered application bytes per second including phases, control traffic and
  retries; separately record acquisition, collection, inference, ACK reading and
  time waiting for a physical turn where measurable.
- Complete authentication through final ACK, plus pairing/proving/transfer/
  verification timing, proof repetitions/bytes, outcomes and stop reasons.

Detection misses while a camera is pointed away and repeated reads of held ACKs
are not packet-loss estimates. Use completed-message goodput and repeated runs,
not only marginal cell accuracy or one favorable frame. Compare G64/G128 and
standard QR with equivalent payloads and completion criteria. Keep G64 as the baseline when evaluating a new layout or model.
G128 needs separate measurements for each camera and direction: twice as many
cells on each axis do not guarantee faster delivery.

## Further questions

Proof compaction, persistent enrollment and mutual proofs are useful follow-ups.
A result returned by the verifier is not a second proof of its identity.
Proposals for a different trust policy should state what is enrolled, what is
shown on each screen and what each party can conclude.

For a first experiment or a concise result-report template, see
[Contributing to Neural Cryptography](../CONTRIBUTING.md). The demo runs
locally; repository maintenance does not require a hosted site. See the
[license and attribution guide](LICENSING.md) before redistributing code or models.
