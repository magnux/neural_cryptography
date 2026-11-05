# Roadmap: fast optical authentication with Neural-QR

Planning baseline: **5 November 2026**. The long-term goal is feasible, fast and
safe optical authentication between devices. The immediate engineering goal is
a repeatable **Neural-ZKBoo** POC carried entirely by **Neural-QR**, our custom
optical format designed around the **neural-corrector**. The existing experiments
are its exploratory foundation. Functional success does not establish audited
cryptographic security.

Neural-QR is not standards-compatible QR. Its implemented collector acquisition replaces
the dense, separately decoded monochrome header with asymmetric geometry and
large phase/reference cues; data and packet metadata pass through neural recovery.
The [Neural-QR design](NEURAL_QR.md) specifies the implemented layout and remaining
packet integration. Header-free `capture.html` now collects its training data;
the working [device demo](NEURAL_ZKBOO_DEMO.md) still uses
OBM5 color DATA and static standard QR ACKs.

The implemented training model is **`multires`**, covering
G8/G16/G32/G64/G128/G256 with **2×2 pixels per cell**. Small controls and larger
data matrices share a convolution/attention trunk and select independent learned
low-frequency coefficient banks that reconstruct G×G anchors, enlarged exactly
2× at the inlet. The bottleneck has six alternating blocks C→A→C→A→C→A,
each with two convolution or attention layers;
the intermediate encoder/decoder level is removed. This replaces the separate-G128
model plan. Mixed-size training, paired synthetics, held-out matrix evaluation
and per-size ONNX export are implemented; training on the new recordings and
browser/packet integration remain next.

## What is established

| Component | Evidence and role |
| --- | --- |
| Neural dynamics research | Recurrent/rotating convolution, normalization, Feistel and surrogate probes remain reference experiments |
| Canonical crypto and authentication | Browser/Python circuit and compact proof implementation; local key generation, pairing confirmation, fresh challenge, proof, result and acknowledgments |
| Trained neural-corrector | G64 U-tube-2x-attn with four phases and 4–8 captures; successful recorded evaluation and live device recovery |
| Hybrid optical authentication | The 4 November device reports confirm local proof acceptance and final result delivery for 32 repetitions |
| Header-free acquisition | Neural-QR rendering, asymmetric orientation, repeated phase cues, 2×2-cell multires crops, balanced recording pools, uploads and offline joins; the user is collecting both directions across sizes |
| Multires training | Six independent low-frequency anchors; shared six-block conv/attention trunk; paired 50/50 real/synthetic sampling; G8…G256 model checks and selected-size ONNX parity |
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

Keep authentication and model computation local in browser workers and serve
their app/model assets statically. The data collector optionally uploads its
recordings to private server storage; it does not move inference to the server.
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
| **N1 — Shared model and Neural-QR format** | Use the implemented `multires` training/export pipeline for G8…G256 with independent anchors, 2×2 pixels/cell and six attention layers at G; train the shared trunk and size-specific anchors; use the implemented header-free collector; add protected in-band metadata and bounded live packet association alongside model work | Export/WASM parity for supported profiles; exact packet recovery; mixed phases/packets reject or refresh cleanly; real-device recovery assessed by size |
| **N2 — QR-free authentication** | Connect small Neural-QR controls and larger DATA profiles to the existing authentication session; choose controls that fit full framing/ECC/binding; use pre-agreed directional profiles initially | Complete identity/pairing/challenge/proof/result/final-ACK exchange with 32 repetitions in both role arrangements; no QR ACK or silent alternate reader; one exchange completes after server disconnection |
| **N3 — Fast complete exchanges and larger proofs** | Tune profile size, capture/inference scheduling and ACK windows from measured costs; update payload limits and bounded deadlines; validate 219-repetition operation | Repeatable delivered bytes/s and complete-authentication time; losses/duplicates/corruption recover or time out; larger profiles improve actual goodput where enabled; compare standard QR under matched conditions |

N1 and N2 deliver the new all-Neural-QR POC; they do not require a sequence of
separately trained networks. The current successful
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
- **Use `multires`.** Share a six-block convolution/attention bottleneck, with
  independent `[1,32,min(G,16),min(G,16)]` cosine coefficient banks and
  size-specific attention/loss masks. Reconstruct each anchor at G and enlarge
  exactly 2×; no bank serves another size. Static exports bake only the selected
  spatial anchor. The processing trunk has 231,120 parameters and all six
  coefficient banks add 43,008, giving 274,128 trainable parameters. Exported
  spatial anchors and geometry masks still affect bundle size. Load only needed
  sessions.
- **Fit controls to real capacity.** G8 has only 32 raw bytes before overhead.
  Select the smallest profile that fits the complete ACK and its binding/ECC;
  retain four phases initially rather than assuming single-frame recovery.
- **Train the current architecture.** Keep palette, phase masks, reference-grid
  conventions and normalization. Reuse the current 2×2-cell datasets in a new
  run; the low-frequency, six-block model does not automatically load the
  preceding dense-anchor multires checkpoint. The supported U-tube attention
  warm start can project its G64 anchor explicitly. Existing G64 recovery is
  not a claim about the new model.
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
standard QR with equivalent payloads and completion criteria. Keep the successful
G64 implementation as the measurement baseline while evaluating the new shared
model. Export parameter size, activation memory and attention work depend on
the chosen profile; G256 still requires device measurements.

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

