# Roadmap: fast optical authentication with Neural-QR

Planning baseline: **4 November 2026**. The long-term goal is feasible, fast and
safe optical authentication between devices. The immediate engineering goal is
a repeatable **Neural-ZKBoo** POC carried entirely by **Neural-QR**, our custom
optical format designed around the **neural-corrector**. The existing experiments
are its exploratory foundation. Functional success does not establish audited
cryptographic security.

Neural-QR is not standards-compatible QR. Its proposed acquisition path replaces
the dense, separately decoded monochrome header with asymmetric geometry and
large phase/reference cues; data and packet metadata pass through neural recovery.
The [Neural-QR design](NEURAL_QR.md) defines this direction. It is **planned**, not
the current runtime: the working [device demo](NEURAL_ZKBOO_DEMO.md) still uses
OBM5 color DATA and static standard QR ACKs.

## What is established

| Component | Evidence and role |
| --- | --- |
| Neural dynamics research | Recurrent/rotating convolution, normalization, Feistel and surrogate probes remain reference experiments |
| Canonical crypto and authentication | Browser/Python circuit and compact proof implementation; local key generation, pairing confirmation, fresh challenge, proof, result and acknowledgments |
| Trained neural-corrector | G64 U-tube-2x-attn with four phases and 4–8 captures; successful recorded evaluation and live device recovery |
| Hybrid optical authentication | The 4 November device reports confirm local proof acceptance and final result delivery for 32 repetitions |
| Optical engineering | Geometry, color references, temporal collection, ECC, CRC, multipart transfer and QR comparison tools are reusable |

The completed exchange took about **6 min 17 s**. A **17,474-byte** proof took
about **160 seconds** to send and acknowledge, approximately **109 B/s** including
alignment and turns. Earlier favorable QR-only transfers reportedly reached about
1 kB/s in different conditions. The completed authentication and useful neural
recovery are established; comparative throughput superiority needs matched tests.

The current PC report used single-threaded WASM after WebGPU initialization failed;
the phone used WebGPU. Inference and geometry costs matter, but the end-to-end
measurement also includes physical turns and acquisition. Improve the measured
path rather than treating model size or symbol accuracy alone as the objective.

## Reuse and scope

Keep computation local in browser workers and serve only static app/model assets.
Reuse the trained G64 checkpoint, U-tube family, capture/generation tools, shared
authentication session and compact proof engine. No Flask service or server-held
key distribution is necessary. No new cryptographic primitive is required for
the transport redesign.

The first QR-free run remains one-direction authentication with roles swapped in
a second attempt. The key holder's received result is a peer report, not a proof
of verifier identity. Persistent enrollment, simultaneous mutual proofs, session
keys, OTPs and recency policies remain later work.

Use the existing side-8, 16-circuit-round configuration and **32 proof repetitions**
for development. Then exercise **219 repetitions** (119,249 raw proof bytes).
These are configurable POC profiles, not certified security levels. Keep optical
dimensions, circuit rounds and proof repetitions independent. The verifier's
saved challenge, identity minimum and repetition policy continue to govern acceptance.

## Prioritized implementation plan

| Milestone | Work | Completion criterion |
| --- | --- | --- |
| **N1 — Neural-QR layout and acquisition** | Fixed G64/16-color/four-phase profile; four anchors with an oriented asymmetric corner; static high-contrast geometry, repeated phase-color cues and same-exposure references; remove the dense external header in the new codec | Correct rectification and phase selection under rotation, reflection, perspective, exposure/color changes and phase transitions; uncertain frames reject cleanly; no standard QR decoder in this acquisition path |
| **N2 — Neural payload and capture association** | Put packet metadata and integrity checks inside ECC-protected payloads; implement bounded phase pools and old/new packet-transition handling; retain the existing neural input/output contract initially | Exact payload recovery without the old header; mixed packets/phases do not advance transport; duplicates, missing phases and capture restarts recover or time out within bounds; test existing weights before deciding whether to fine-tune |
| **N3 — QR-free authentication** | Use Neural-QR for START, DATA and ACK; initially reuse the G64 neural envelope for controls; connect the unchanged authentication session | Complete identity/pairing/challenge/proof/result/final-ACK exchange with 32 repetitions on real devices in both role arrangements; no QR ACK or silent alternate reader; one exchange completes after server disconnection |
| **N4 — Faster complete exchanges** | From measured costs, improve control profile, camera/inference scheduling and cumulative ACK windows with selective retry; investigate PC GPU availability and capture stability | Three consecutive completions per direction with repeatable goodput and timing; lost/duplicated/corrupted captures recover or time out; compare standard QR under matched devices, placement, payloads and ACK criteria |
| **N5 — Larger payloads and proof** | Trial a separate G128 corrector, new-layout recordings, 512×512 inputs and 128×128 bottleneck/output; increase payload/chunk limits with the codec; validate 219-repetition operation | G128 improves acknowledged bytes/s or full authentication time over the working G64 profile in each tested direction; larger proof completes with suitable bounded deadlines; publish the settings and unsuccessful cases too |

N1–N3 are the shortest path to the new all-Neural-QR POC. The current successful
hybrid demo stays runnable while those pieces are developed. Activate the new
format on both endpoints together; maintain one current format per layer and
keep earlier implementations in Git history instead of adding compatibility readers.
The separate standard-QR experiment remains useful for comparison.

### Important implementation decisions

- **Keep four anchors.** Replacing a corner with an oriented asymmetric finder
  preserves perspective geometry; removing it requires additional constraints.
- **Preserve geometry contrast.** Phase colors belong in separate large patches;
  compare repeated cues with fixed references from that exposure.
- **Make packet grouping explicit.** Phase labels do not identify a packet. Begin
  with stop-and-wait and bounded latest-per-phase pools, resetting and replacing
  evidence across transitions. Only valid full packet metadata can advance the
  transfer. Add a coarse generation cue only if the measured transition behavior
  needs it; defer windows until grouping is reliable.
- **Do not assume a tiny ACK is free.** The current model expects G64/four phases.
  Reusing it removes QR dependencies but may increase control latency. A compact
  or single-frame neural control profile needs its own supported contract and tests.
- **Try existing weights first.** Keep G64 palette, phase masks, payload grid and
  normalization stable initially. Capture the new layout and fine-tune if needed;
  do not require another long architecture search before testing the format.
- **Bound long transfers.** The current ten-minute message limit would interrupt
  a 219-repetition proof at the observed rate. Configure deadlines from message
  size and tested throughput, with hard limits; duplicates must not extend them forever.

## Byte integrity and authentication gates

The new design removes the header's CRC gate, not payload validation. Require ECC
recovery, packet CRC, length/schema validation, full identifiers and sequence
matching, and a complete-message digest before application delivery. Never use
sender diagnostic truth to recover bytes. Neural confidence, CRC and physical
proximity are not authentication.

Preserve current checks for pinned identities, explicit pairing confirmation,
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
standard QR with equivalent payloads and completion criteria. The working G64
model remains the baseline if a larger profile is slower or less reliable.

## Later research

The [research connections](RESEARCH_CONNECTIONS.md) motivate later cryptanalytic
comparison, independent attacker ensembles, witness/key-recovery experiments and
alternative proof-friendly circuits. Further proof compaction, persistent trust,
mutual authentication, temporary credentials, proof chaining and broader model
families are separate research directions. They do not block N1–N3, and no claim
of production or post-quantum security follows from successful optical transport.

## Documentation and licensing

- [Neural-QR design](NEURAL_QR.md): next layout, phase cues, metadata, ACKs and tests.
- [Neural-corrector](NEURAL_CORRECTOR.md): architecture and model contract.
- [Current device demo](NEURAL_ZKBOO_DEMO.md): runnable hybrid setup and procedure.
- [Current format boundaries](FORMATS.md): distinguish implemented formats from plans.
- [Apache-2.0 license](../LICENSE), [NOTICE](../NOTICE), [licensing scope](LICENSING.md)
  and [citation metadata](../CITATION.cff): original project materials are credited
  to **Alejandro Hernandez Ruiz (magnux), Neural Cryptography**; third-party terms remain intact.

Hosted sites remain parked unless deployment is separately requested. Source,
documentation and current model tooling stay in the repository; private keys,
recordings, training checkpoints and individual experiment logs stay out of it.
