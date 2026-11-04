# Neural-QR: optical transport designed for a neural corrector

**Status: proposed next format, 4 November 2026.** Neural-QR is the project's
name for its custom optical matrices. It is not a standards-compatible QR code.
The design below is not yet the runnable wire format: the working
[Neural-ZKBoo demo](NEURAL_ZKBOO_DEMO.md) still uses OBM5 color DATA and standard
QR ACKs. Its successful device exchange is the baseline for the next implementation.

The goal is fast, reliable optical authentication, with Neural-QR carrying both
data and control messages, the **neural-corrector** recovering logical matrix
symbols, and **Neural-ZKBoo** checking the proof. Removing the separate dense
monochrome header should remove an acquisition bottleneck and free layout space.
It does not remove packet framing, error correction, or cryptographic verification.

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

## Initial profile

Start with one pre-agreed profile: **64×64 symbols, 16 colors, four palette
phases, grid-8 references and RS32**. Both apps know this profile before scanning;
no separate optical header is needed to discover it. Preserve the current palette,
phase masks `(0, 15, 7, 8)`, logical-label output and phase-paired input contract:
four to eight distinct exposures, at most two per phase, rectified to 256×256 RGB.

Preserve the payload grid and measured color normalization initially so the
trained G64 weights can be tested first. A changed outer frame does not by itself
prove retraining is required. New-layout captures must establish recovery; collect
additional real examples and fine-tune only if that evaluation shows a gap.
Reclaiming header space may enlarge displayed cells, but a fixed G64 symbol grid
does not automatically carry more bytes. The final renderer determines the gain.

## Geometry and orientation

Keep **four anchor locations**. Use three ordinary high-contrast finders and
replace the bottom-right finder with an **oriented asymmetric fiducial** whose
center can still be localized accurately. Its internal pattern must distinguish
handedness as well as rotation; merely making one corner a different square or
circle does not resolve every mirrored ordering.

Four point correspondences determine the planar perspective transform. Removing
a corner would require recovering additional constraints from marker edges or
the frame; three point correspondences alone determine an affine transform.
See the primary [OpenCV transform reference](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

Keep finder contours black/white and unchanged through the phase cycle. The
colored phase cues sit in separate patches with clear spacing. This keeps
geometry detection independent of phase color, exposure shifts and palette noise.
Known ordered border references also provide an orientation consistency check.
Try recently validated geometry first, then reacquire when current cues disagree
or the geometry-age bound expires. Cached coordinates never supply old phase data.

## Phase cues without the dense header

Use large, repeated color patches measured against fixed palette references in
the **same exposure**. A concrete initial candidate is a pair of patches for
phase `p`: palette entries `(p, (p + 1) mod 4)`, using the four well-separated
entries at the start of the existing palette. Repeat that pair on the opposite
side of the frame. These cue colors are specified independently of the payload's
XOR phase permutation; the reference palette itself stays fixed.

Classify the pair after local reference correction, require an adequate margin
between candidate phases, and require the separated copies to agree. An uncertain
or inconsistent exposure is skipped. Top/bottom disagreement is useful evidence
of a display transition or rolling-shutter mixture. These are robust optical cues,
not a second byte-oriented header or an authentication mechanism.

This removes the current 512-bit header and its CRC as a prerequisite to entering
the model. It does **not** imply that every frame is usable, or that one patch's
nearest color is sufficient evidence of its phase. Test cue colors and thresholds
under blur, clipping, glare and camera color processing before freezing the layout.

## Packet boundaries and metadata

Place packet type, full session/message identifiers, sequence, length and packet
integrity checks **inside the ECC-protected color payload**. Fixed profile
geometry defines symbol packing and the ECC block layout before those fields are
read. Decode the complete fixed-size body, then validate its encoded length and
packet schema. Pad unused cells as in the current dense training distribution.
Keep full-message digest checks and current transfer matching above this layer.

The current receiver groups captures using identity read from the monochrome header.
Removing that header therefore requires an explicit replacement for burst
membership; phase labels alone do not identify a logical packet.

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

## ACKs and control messages

The first QR-free prototype can carry the existing complete ACK packet through
the **same G64, four-phase Neural-QR envelope**. This exercises the same trained
corrector contract on both paths and avoids a second codec. It may cost more
acquisition/inference time than today's static QR ACK; measure that cost directly.

A later compact control profile should use fewer symbols and, if reliable, fewer
captures. The current G64 checkpoint cannot simply read an arbitrary tiny static
code. That optimization needs an explicitly supported model/input contract or a
separate small corrector, with its own recovery evaluation. Preserve ACK binding
to the transfer, message, acknowledged sequence and digest in either profile.

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

1. Freeze the G64 layout and cue rules using renderer/detector simulations and
   current model replay, including packet-transition tests.
2. Implement the single Neural-QR payload codec, capture association and score
   adapter; try the existing trained model and collect new-layout device examples.
3. Complete QR-free Neural-ZKBoo authentication, including Neural-QR ACKs, on both
   role arrangements. Update both endpoints together; no dual-format reader.
4. Improve compact controls, bounded capture/inference scheduling and acknowledged
   windows from measured bottlenecks.
5. Trial a separate G128 corrector: 512×512 input and 128×128 bottleneck/output,
   preserving local attention and compact parameters. Validate real camera detail
   and increase transport chunk capacity alongside the model; training alone does
   not remove the current application's chunk limit.

The working OBM5/QR-ACK demo remains runnable until the replacement is implemented.
Once Neural-QR becomes the active format, retain its predecessor through Git history,
not compatibility parsers. The standalone standard-QR experiment remains a comparison
tool. See the [roadmap](ROADMAP.md) for completion criteria.

## License and attribution

The project's original Neural-QR implementation and design documentation are
covered by [Apache-2.0](../LICENSE), with project/author attribution in
[NOTICE](../NOTICE): **Alejandro Hernandez Ruiz (magnux), Neural Cryptography**.
See [licensing scope](LICENSING.md) and [CITATION.cff](../CITATION.cff).
Third-party components retain their own terms and credit. Licensing these materials
does not claim ownership of standard QR, neural networks or earlier cryptographic methods.
