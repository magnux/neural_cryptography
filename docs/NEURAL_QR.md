# Neural-QR: optical transport designed for a neural corrector

**Status: header-free acquisition and dataset collection implemented, 4 November 2026.**
`capture.html` now renders and acquires Neural-QR without an OBM5 header. It uses
asymmetric orientation markers, repeated phase cues and an explicitly configured
matrix size. Neural-QR is the project's custom optical format, not a
standards-compatible QR code.

This milestone collects training evidence. It does not yet deliver Neural-QR
packets or perform QR-free authentication. The working
[Neural-ZKBoo demo](NEURAL_ZKBOO_DEMO.md) still uses its OBM5 color DATA and
standard QR ACKs while the new multires model and in-band packet codec are built.
The collector accepts only the new acquisition layout; it has no OBM5 fallback.

The goal is fast, reliable optical authentication, with Neural-QR carrying both
data and control messages, the **neural-corrector** recovering logical matrix
symbols, and **Neural-ZKBoo** checking the proof. Removing the separate dense
monochrome header should remove an acquisition bottleneck and free layout space.
It does not remove packet framing, error correction, or cryptographic verification.

## Visual layout and implemented acquisition

![Rendered Neural-QR G32 at phase 2: four anchors with asymmetric bottom-right satellites, repeated phase cues, fixed palettes and a header-free symbol matrix](images/neural-qr-acquisition.png)

The image is produced by the current renderer from a public random G32 training
matrix at phase 2. The executable layout is defined in
[`neural-qr/layout.js`](../apps/color-optical-bench/dist/js/neural-qr/layout.js).
The implemented bottom-right fiducial preserves the square localization core and
adds six asymmetric black/white satellite patches. The satellites distinguish
rotation and reflection. Geometry and reference colors stay fixed across phases;
only the phase patches and phase-permuted data colors change.

| Element | Implemented coordinates in the 256-unit square |
| --- | --- |
| Finder centers | (22,22), (234,22), (234,234), (22,234) |
| Payload | x44, y44, side168; G×G cells including grid references |
| Fixed palettes | x22/x234; 16 colors centered at y52+10i |
| Phase pairs | x104/x132 at y20, repeated at y236; 20-unit patches |
| Bottom-right satellites | x214 at y226/234/242 has bits100; y214 at x226/234/242 has bits110 |

The collector displays uniformly random logical symbols with fixed reference
cells, not an encoded application packet. It is suitable for learning the optical
correction task without requiring a trained model to label recordings.

The target is a **multi-resolution family**, G8/G16/G32/G64/G128/G256, read by
**`multires`**: a shared convolution/attention trunk with one learned G×G anchor
per size, enlarged exactly 2× at its inlet. At G the bottleneck alternates
**2 convolutions → 2 attention blocks → 2 convolutions → 2 attention blocks**.
G64 is the existing trained starting point;
small controls and larger data matrices will use different profiles after training.
The [model design](NEURAL_CORRECTOR.md#multires-architecture)
specifies crop sizes, anchors, masks, export and training changes.
Rectification uses **2×2 pixels per cell**: G32 → 64×64, G64 → 128×128 and
G256 → 512×512. Drawing cell dimensions are illustrative display units, not
camera sampling pixels.

## Design boundaries

| Layer | Responsibility |
| --- | --- |
| Neural-QR geometry and optical cues | Locate and orient the image; identify a clear palette phase; measure color references |
| neural-corrector | Convert distinct captured frames into logical symbol scores |
| Error correction and packet parsing | Recover bytes; reject damaged or malformed packets; check session, message and sequence context |
| Neural-ZKBoo | Verify the existing proof against the pinned public identity and fresh challenge |

Neural confidence is evidence for classification, not proof that bytes are exact.
CRC and Reed–Solomon recovery address channel errors, not a malicious sender.
The existing proof and authentication envelope remain the canonical crypto layer.

## Starting profile and shared multi-resolution model

The acquisition profile is **16 colors, four palette phases and grid-8
references**. Error correction belongs to the future packet profile. Both apps know
the directional grid sizes before scanning; no dense header is needed to discover
them. Preserve palette masks `(0, 15, 7, 8)`, logical-label output and phase pairing:
four to eight distinct exposures, at most two per phase. The planned model takes
`2G×2G` RGB crops. Compatible weights can be initialized from the trained G64
checkpoint, but the changed sampling and shared size family require training.
The current model/export pipeline is not a drop-in reader for these recordings.

Preserve the reference-grid convention and measured color normalization. Retain
the G64/RS32 data profile as a comparison while adding smaller controls and larger
payloads; overhead and error-correction settings must fit each profile.
Reclaiming header space may enlarge displayed cells, but a fixed G64 symbol grid
does not automatically carry more bytes. The final renderer determines the gain.

## Geometry and orientation

The detector keeps **four anchor locations**, using three ordinary high-contrast
finders and one **oriented asymmetric bottom-right fiducial**. Its six satellite
patches distinguish handedness as well as rotation while retaining the same
accurately localized square core. Acquisition tries all rotations and reflected
orders, then validates the asymmetric pattern.

Four point correspondences determine the planar perspective transform. Removing
a corner would require recovering additional constraints from marker edges or
the frame; three point correspondences alone determine an affine transform.
See the primary [OpenCV transform reference](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

Keep finder contours black/white and unchanged through the phase cycle. The
colored phase cues sit in separate patches with clear spacing. This keeps
geometry detection independent of phase color, exposure shifts and palette noise.
Known ordered border references also provide an orientation consistency check.
The acquisition implementation tries recent geometry first, then reacquires when
current cues disagree or the 400 ms geometry-age bound expires. Finder cores,
rings and asymmetric patches are checked again against the current exposure.
Reusing coordinates never supplies old phase data or extends the original
geometry acquisition time.

## Phase cues without the dense header

Acquisition measures large, repeated color patches against fixed palette
references in the **same exposure**. Each pair for phase `p` uses palette entries `(p, (p + 1) mod 4)`, using the four well-separated
entries at the start of the existing palette. Repeat that pair on the opposite
side of the frame. These cue colors are specified independently of the payload's
XOR phase permutation; the reference palette itself stays fixed.

The detector compares cues with the observed reference colors in RGB space,
requires separation between references and a margin between candidate phases,
and requires the top and bottom copies to agree. An uncertain
or inconsistent exposure is skipped. Top/bottom disagreement is useful evidence
of a display transition or rolling-shutter mixture. These are robust optical cues,
not a second byte-oriented header or an authentication mechanism.

The collector no longer requires the 512-bit header or its CRC to acquire a crop.
It does **not** imply that every frame is usable, or that one patch's
nearest color is sufficient evidence of its phase. Test cue colors and thresholds
under blur, clipping, glare and camera color processing before freezing the layout.

## Collection identity and sample admission

Without an optically decoded packet identifier, a phase number alone cannot
identify which random matrix is on screen. The collector therefore uses an
explicit **public collection code**, shown by the sender and entered on the
receiver together with the grid size. The eight hexadecimal characters are the
CRC32 of the logical training matrix. The sender manifest retains its full
SHA-256 and labels for offline validation; the receiver receives neither labels
nor clean pixels. The join checks code, profile, phase and manifest consistency
and rejects conflicting sender assignments.

The code is an out-of-band association, **not a value recovered from the camera**.
On the demo server, sender creation stages its manifest and publishes a public
collection-session descriptor. The receiver automatically reads the code, profile
and split; completed receiver upload finalizes the sender and clears the descriptor.
Manual code entry remains available on static hosts. No sender labels enter
acquisition. See [the collection workflow](MULTIRES_COLLECTION.md).
Typing a valid code while photographing another matrix can mislabel a recording;
phase agreement and image similarity cannot prove otherwise. Keep one static
logical matrix assigned for all four phases, save the batch before changing it,
and review suspect recordings offline. Full packet identity will move in-band
when the trained receiver and codec are integrated.

Admission requires clear geometric orientation and agreeing phase cues. The
collector then waits at least 500 ms and two observations of the same assigned
phase before retaining evidence. A corner displacement exceeding 4% of the
current diagonal restarts this settling guard. Payload CRC success is not an
admission gate:
these training matrices are random symbols, without an application packet or
packet CRC. Similarity selection uses only captured pixels and does not consult
sender truth. See [collection and retention](MULTIRES_COLLECTION.md) for the
centered sample distribution and exact workflow.

## Packet boundaries and metadata — next integration

Place packet type, full session/message identifiers, sequence, length and packet
integrity checks **inside the ECC-protected color payload**. Fixed profile
geometry defines symbol packing and the ECC block layout before those fields are
read. Decode the complete fixed-size body, then validate its encoded length and
packet schema. Pad unused cells as in the current dense training distribution.
Keep full-message digest checks and current transfer matching above this layer.

The collection code solves offline labeling for a manually held training matrix.
It is not the live protocol's packet-membership mechanism. Phase labels alone do
not identify a logical packet, so the live receiver still needs the following
in-band checks and bounded transition behavior.

For the first implementation, retain **strict stop-and-wait** and a bounded pool
of the latest one or two distinct captures per phase. Reset after an accepted
packet and on run/geometry discontinuities; reject stale observations by age.
Old/new packet mixtures can occur during a transition, so failed recovery must
allow fresh observations to replace them. A repeated accepted packet can reproduce
its ACK but must not be delivered to the application twice. Only an ECC/CRC-valid
packet with matching full identifiers and sequence may advance the transfer.

Evaluate this transition behavior before adding simultaneous packet windows. A
small, coarse generation cue is an optional grouping aid if recovery stalls; it
must not replace the full in-band identifiers or become another dense header.
This is a design decision to test, not an already implemented grouping guarantee.

```mermaid
flowchart TD
  A["Current camera exposure"] --> B["Geometry and repeated phase cues"]
  B -->|"clear cues"| C["Bounded phase pool"]
  B -->|"uncertain"| A
  C --> D["Neural symbol scores"]
  D --> E["ECC, packet CRC and context checks"]
  E -->|"retry with fresh evidence"| C
  E -->|"valid packet"| F["Transfer state and Neural-ZKBoo"]
```

## ACKs and control messages — next integration

The shared multi-resolution corrector makes a **smaller Neural-QR control profile**
the preferred path. Select the smallest grid that fits the complete ACK plus
reference cells, framing, ECC and integrity checks. G8 carries only 32 raw bytes
at 16 colors before any overhead, so do not assume it can hold the present ACK.
Keep ACK binding to transfer, message, acknowledged sequence and digest.

Use the same four-phase, four-to-eight-capture contract initially; smaller grids
reduce neural computation without claiming that a single exposure is sufficient.
One shared model can serve small controls and large data after mixed-size training
and export. Until that integration is available, the G64 envelope can exercise the
same codec on both paths. Measure control acquisition, processing and physical-turn
latency directly rather than assuming that removing standard QR makes it faster.

After grouping is reliable, cumulative acknowledgments and selective retransmission
can reduce physical turns. Do not advance through unrelated packets on a timer
without retaining the receiver's missing-packet state.

## Acceptance tests and measurement

- Geometry: perspective, all rotations, reflection, partial occlusion and confusing
  background shapes. A tentative location is distinct from an accepted packet.
- Phase cues: same-exposure color correction, clipped colors, motion blur, glare,
  rolling shutter and frames spanning a phase transition. Uncertainty causes retry.
- Burst membership: skipped phases, repeated exposures, old/new packet mixtures,
  duplicate packets, camera motion, restart and expired samples. Buffers stay bounded.
- Bytes and authentication: exact delivered bytes, no damaged packet acceptance in
  the test set, mandatory packet/message checks, lost ACK recovery, stale challenge
  rejection, altered proof rejection, and the final result acknowledgment.
- Devices: repeat phone→PC and PC→phone, record unique acknowledged bytes per second,
  acquisition/inference/ACK timings and complete authentication time. Keep image size,
  devices, repetitions and physical placement comparable when testing standard QR.

On **4 November 2026**, the existing hybrid demo completed a 32-repetition exchange
in about **6 min 17 s**. Its 17,474-byte proof took about 160 seconds to send and
acknowledge, approximately **109 B/s** including turns. Earlier favorable QR-only
transfers reportedly reached roughly 1 kB/s under different conditions. These
observations establish useful neural-assisted reception and end-to-end function;
they do not establish a controlled throughput advantage over standard QR or an
audited cryptographic security result.

## Implementation sequence

1. **Implemented:** header-free renderer, asymmetric orientation, repeated phase
   acquisition, 2×2-pixel crops, public collection assignment, bounded collection,
   uploads and offline joining. Validate the new marker/cue layout on both device
   directions before recording the full multires dataset.
2. **Implemented:** `multires` model, paired synthetic generation, size-balanced
   training, held-out matrix evaluation and static per-size ONNX export. Train
   on the collected sessions and measure exact logical matrices per size and
   direction before integrating the model into the receiver.
3. Add the in-band ECC/CRC packet codec and full identifier checks, bounded
   packet/phase association and small Neural-QR controls to authentication. This
   replaces the working OBM5/QR-ACK path when both endpoints are ready.
4. Complete QR-free authentication in both role arrangements and improve
   capture/inference scheduling and acknowledgment windows from measured costs.

The standalone standard-QR experiment remains a comparison tool. The runnable
Neural-ZKBoo baseline and header-free data collector are explicitly separate
entry points, not a receiver that guesses between formats. See the
[roadmap](ROADMAP.md) for end-to-end completion criteria.

## License and attribution

The project's original Neural-QR implementation and design documentation are
covered by [Apache-2.0](../LICENSE), with project/author attribution in
[NOTICE](../NOTICE): **Alejandro Hernandez Ruiz (magnux), Neural Cryptography**.
See [licensing scope](LICENSING.md) and [CITATION.cff](../CITATION.cff).
Third-party components retain their own terms and credit. Licensing these materials
does not claim ownership of standard QR, neural networks or earlier cryptographic methods.
