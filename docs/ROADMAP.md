# Roadmap: fast optical authentication with Neural-QR

Planning baseline: **7 November 2026**. The long-term goal is feasible, fast and
safe optical authentication between devices. The immediate engineering goal is
a repeatable **Neural-ZKBoo** POC carried entirely by **Neural-QR**, our custom
optical format designed around the **neural-corrector**. The existing experiments
are its exploratory foundation. Functional success does not establish audited
cryptographic security.

Neural-QR is not standards-compatible QR. Its implemented collector acquisition replaces
the dense, separately decoded monochrome header with asymmetric geometry and
large size/phase/reference cues; future data and packet metadata pass through
neural recovery.
The [Neural-QR design](NEURAL_QR.md) specifies the implemented layout and remaining
packet integration. Header-free `capture.html` now collects its training data;
the working [device demo](NEURAL_ZKBOO_DEMO.md) still uses
OBM5 color DATA and static standard QR ACKs.

The current model for the new corpus is **W-tube**, using **4×4 pixels per cell** at
G16/G32/G64/G128. Two complete U stages preserve full-resolution features and
reach the target G at each bottleneck. Each bottleneck has one two-convolution
block and one two-attention-layer block. Independent low-frequency anchors and
four-phase K4–K8 inputs are implemented. Python training, evaluation and strict
checkpoint loading are ready; new device accuracy and browser deployment still
require a trained model and measurement.

The current `multires` model remains the preceding 2×2-cell experiment. Its
completed run learned G16 and G32, but real G64+ recovery was insufficient;
constraining anchors did not resolve the real/synthetic gap. Collection version 2
now provides the richer crops required for the next experiment. Its files must
not be passed directly into the current Python multires training command.

## What is established

| Component | Evidence and role |
| --- | --- |
| Neural dynamics research | Recurrent/rotating convolution, normalization, Feistel and surrogate probes remain reference experiments |
| Canonical crypto and authentication | Browser/Python circuit and compact proof implementation; local key generation, pairing confirmation, fresh challenge, proof, result and acknowledgments |
| Trained neural-corrector | G64 U-tube-2x-attn with four phases and 4–8 captures; successful recorded evaluation and live device recovery |
| Hybrid optical authentication | The 4 November device reports confirm local proof acceptance and final result delivery for 32 repetitions |
| Header-free acquisition | Version-2 Neural-QR rendering, asymmetric orientation, repeated optical size/phase cues, direct 4×4-cell G16–G128 crops, bounded recording pools, uploads and offline joins; new device checks pending |
| Previous multires experiment | Six independent low-frequency anchors; shared six-block conv/attention trunk; 2×2-cell G8…G256 model checks and ONNX parity; insufficient real G64+ recovery motivates W-tube |
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
| **N1 — W-tube and Neural-QR format** | Collect native 4×4-cell G16/G32/G64/G128 with optical size codes; implement W-tube, tagged auxiliary imports and harder paired synthetics; train and export; add protected in-band metadata and bounded packet association | Native G64 recovery, then G128; export/WASM parity; exact packet recovery; mixed phases/packets reject or refresh cleanly |
| **N2 — QR-free authentication** | Connect small Neural-QR controls and larger DATA profiles to the existing authentication session; choose controls that fit full framing/ECC/binding; read and validate directional size profiles optically | Complete identity/pairing/challenge/proof/result/final-ACK exchange with 32 repetitions in both role arrangements; no QR ACK or silent alternate reader; one exchange completes after server disconnection |
| **N3 — Fast complete exchanges and larger proofs** | Tune profile size, capture/inference scheduling and ACK windows from measured costs; update payload limits and bounded deadlines; validate 219-repetition operation | Repeatable delivered bytes/s and complete-authentication time; losses/duplicates/corruption recover or time out; larger profiles improve actual goodput where enabled; compare standard QR under matched conditions |

N1 and N2 deliver the new all-Neural-QR POC; they do not require a sequence of
separately trained networks. The current successful
hybrid demo stays runnable while those pieces are developed. Activate the new
format on both endpoints together; maintain one current format per layer and
keep earlier implementations in Git history instead of adding compatibility readers.
The separate standard-QR experiment remains useful for comparison.

### Current iteration checklist

- [x] Freeze the collector profile at G16/G32/G64/G128, 16 colors, four phases
  and four rectified pixels per cell.
- [x] Add repeated optical size codes while retaining four finder cores, fixed
  asymmetric orientation, phase cues and color references. Read G independently,
  then cross-check the active assignment.
- [x] Update direct camera rectification, collection UI, byte budgets, version-2
  sender/receiver/session files, uploads and offline joining. Validate on devices
  before a long recording session; see [collection checks](MULTIRES_COLLECTION.md).
- [x] Import eligible older G64 payload crops without resizing and enlarge
  selected 2-pixel Neural-QR crops to four pixels/cell. Preserve original session
  splits and source provenance; keep fresh native validation primary. These are
  explicit offline dataset conversions, not live compatibility readers.
  See [the W-tube data pipeline](W_TUBE_DATA.md) for the three recording roots.
- [x] Generate harder paired synthetic bursts at a target **4:1 synthetic/real
  dataset volume**, retaining **50/50 training sampling**. Preserve coherent
  phase trajectories and vary camera resolution, blur, curved moiré, rolling
  shutter, glare, exposure and color effects across readable and hard examples.
- [x] Implement **W-tube**: two `4G → 2G → G → 2G → 4G` stages, 32-channel
  normalized feature paths, residual fusion, two convolution layers and two
  attention layers per bottleneck, independent low-frequency anchors, and a
  learned `4G → 2G → G` outlet. Retain presence/loss masks and phase/flip
  augmentation. [Start a new training run](NEURAL_CORRECTOR_TRAINING.md#train-w-tube).
- [ ] Balance sizes, directions and real origins. Keep native data the majority
  of the real quota once available and preserve an effective batch of 32.
  Native-first checkpoint selection and separate auxiliary validation summaries
  are already implemented; the real-origin sampling quota is still pending.
- [ ] Trim histories to static metadata plus canonical training and validation
  tables; derive aggregate views rather than nesting repeated copies.
- [ ] Export each supported geometry, check numerical parity and measure actual
  browser latency/memory. A small parameter count does not imply fast G128
  inference.
- [ ] Restore the authentication path with native G64 DATA, G16 controls where
  full framing fits, and G128 as a measured throughput extension.

### Integration decisions

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
  control fits. Retain four phases initially.
- **Bound long transfers.** Configure deadlines from message size and tested
  throughput with hard limits; duplicates must not extend them forever.

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
the chosen profile; G128 still requires device measurements.

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

