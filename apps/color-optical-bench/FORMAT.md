# OBM5 experimental transport

One format only. Every endpoint must run this release. There is no backwards
compatibility. All multibyte header integers are unsigned, big endian.

## Monochrome header

The 64-byte header is a 32×16 bit grid at virtual (48,8), with 5×3 unit cells.
The eight-byte content descriptor stays here, outside colored data cells.

| Byte(s) | Meaning |
|---|---|
| 0–3 | ASCII `OBM5` |
| 4 | Format version 5 |
| 5 | Bits per color symbol: 1, 2, 3, or 4 |
| 6–7 | Optical grid side, 8–512 |
| 8–15 | Random eight-byte sequence identifier |
| 16–17 | Logical content frame index, zero based |
| 18–19 | Logical frame count, 1–8 |
| 20–23 | Content chunk length, excluding parity and padding |
| 24–27 | Total content length |
| 28–31 | CRC32 of complete content |
| 32–35 | CRC32 of content chunk |
| 36–43 | `[O,C,1,type,length_be32]` content descriptor |
| 44 | Reference block size: 0, 4, 8, 16 |
| 45 | Phase index, zero based |
| 46 | Phase count: 1, 2, 4, 8, 16 |
| 47 | Palette permutation enabled: 0 or 1; zero for one phase |
| 48 | RS parity bytes per block: 0, 16, 32 |
| 49–50 | Intended phase hold in milliseconds: 250–10,000; default 4,000 |
| 51–52 | Reserved, must be zero |
| 53 | Diagnostic pattern: 0 ordinary, 1 random, 2 runs, 3 alternating |
| 54–55 | Reserved, must be zero |
| 56–59 | Diagnostic seed; zero when pattern is zero |
| 60–63 | CRC32 of header bytes 0–59 |

Content types: 0 raw, 1 UTF-8, 2 PNG, 3 JPEG, 4 PCM WAV. Frame lengths/count must
match computed net capacity. A diagnostic sequence must use raw content and fill
every chunk to capacity. Unknown versions, reserved bits, malformed profiles,
and invalid checksums are rejected. Phase hold is part of sequence identity;
changing playback rate regenerates the transmitted sequence.

## Reference geometry

The N×N optical matrix remains in virtual square (48,72), side 160. Four markers,
timing tracks, and both outside strips retain their geometry. The nested palette
has RGB levels 24,96,168,240; `PALETTE` in `format.js` defines its exact order.

For a block setting B>0, step S=B+1. Rows/columns divisible by S, plus the final
row and column, are reference cells. Other cells carry wire symbols in raster
order. Edge data blocks can be smaller than B. A horizontal reference has color
`(x + 5*floor(y/S)) mod K`; remaining vertical references have color
`(y + 7*floor(x/S)) mod K`. Intersections use the horizontal rule. Reference
colors do not change with payload or phase. No data truth is used as a reference.

Each capture fits an affine RGB correction from the 32 outside-strip samples.
Local prototypes use the nearest four same-color internal references within the
neighboring block region, weighted by inverse squared distance plus one. Their
current corrected RGB is shrunk 20% toward nominal colors. Missing local colors
use the same-exposure border model, and their count is reported. This is a simple
local estimator, not a claim that nearby cells share identical moiré.

## Reed–Solomon and interleaving

Let M be the count of nonreference cells, b=log2(K), and W=floor(M*b/8).
With P parity bytes, J=ceil(W/255) blocks and net capacity C=W-J*P. With P=0,
J=1 and C=W. Require C>0. Distribute C data bytes across J blocks as evenly as
possible, larger blocks first. Pad the content chunk with zero bytes to C.

RS uses GF(256), polynomial 0x11d, primitive element 2, generator roots alpha^0
through alpha^(P-1). Codewords are systematic and highest-degree byte first.
Each shortened block has at most 255 total bytes. The errors-and-erasures decoder
removes known erasure locations from the syndrome sequence, finds unknown error
locations using Berlekamp–Massey and Chien search, solves magnitudes against the
original syndromes, and rechecks all syndromes. Its correction bound is
`2e + s ≤ P`, for e errors outside s supplied erasure positions. Correct bytes
marked as erased still consume erasure capacity. Correction counts include actual
changed bytes, not every supplied erasure position.

Interleave codewords by taking column 0 from each block, then column 1, and so
on, skipping ended blocks. Pack the wire bytes MSB first into raster nonreference
cells; unused final cell bits are zero. Interleaving spreads adjacent wire-byte
errors across blocks. It does not guarantee correction of arbitrary image regions.

Reception follows this bounded procedure:

1. Undo interleaving and check chunk CRC plus zero padding. If valid, accept;
   parity-only damage does not need repair.
2. Otherwise, try ordinary RS on every block and check chunk CRC and padding again.
3. If enabled and still unsuccessful, rank each block's bytes by ascending
   confidence. Confidence comes from the top-two logical-symbol score margin;
   a byte receives the minimum confidence of the cells contributing its bits.
   This also covers three-bit symbols spanning byte boundaries.
4. Try erasure budgets 4, 8, 16, and 32, clipped to the selected parity and
   deduplicated. There are at most four whole-candidate trials. Reuse blocks that
   passed ordinary RS if another block failed. If every block passed its syndrome
   check but the content CRC failed, retry all blocks from the original bytes.
5. Accept only a candidate that passes the mandatory chunk CRC and zero padding.

Unused bits beyond the coded wire bytes are checked but have no RS protection.
There is no unbounded search, CRC-guided symbol editing, or use of diagnostic truth
in correction. A misleading confidence ranking can consume the erasure budget
without helping. Ordinary RS is always attempted first.

## Palette phases

Each phase sends the same coded chunk. Repetition mode uses identity mapping.
Permutation mode XORs every logical data symbol with a public mask. Take the first
Q entries for Q phases, cycling the list if Q exceeds the palette size K:

| K colors | Masks, in phase order |
|---|---|
| 2 | 0, 1 |
| 4 | 0, 3, 1, 2 |
| 8 | 0, 7, 3, 4, 1, 6, 2, 5 |
| 16 | 0, 15, 7, 8, 3, 12, 4, 11, 1, 14, 6, 9, 2, 13, 5, 10 |

For K=16, sixteen phases show each logical symbol in all sixteen physical colors.
Mappings repeat when Q>K. These masks provide optical diversity, not encryption.
Metadata, phase, timing, and content type remain in the monochrome header.

## Progressive fusion and sample selection

After independent per-exposure color fitting, squared color distances produce
softmax scores with denominator 3600, normalized and quantized to bytes. Scores
are indexed by logical labels after undoing the phase mapping. No raw RGB is
pooled across different mappings. Sequence, logical frame, and phase identify
separate pools; repeated captures of one phase cannot count as distinct phases.

The default robust estimator requests five minimum samples and two overhead
slots per phase. For each data cell independently:

1. Normalize every observation's vector across the K logical labels.
2. Choose the medoid: the observation with the smallest summed L1 distance to the
   other observations. Resolve ties deterministically by pool order.
3. Select the target number of observations closest to the medoid and average
   their normalized score vectors.

The first estimate can use five observations. With six or seven available, select
five from that larger pool. Pool replacement is FIFO, independent of whether a
sample was selected. Old winners are never permanently protected. Each cell may
choose different observations. The ordinary mean of all observations in the same
pool remains an independent candidate; mean-only and median-only modes remain
available as experiment controls.

A phase is eligible once it holds the effective minimum sample count. Fusion
starts when enough distinct phases are eligible: default four, clamped to the
transmitted phase count. Every eligible phase contributes equally, regardless of
its number of captures. Later eligible phases join subsequent attempts; new
observations keep refining the result after the pool fills. The default therefore
does not wait for all sixteen phases before trying to recover a sixteen-phase
transmission.

Border and local color models each produce fused candidates. In robust mode the
receiver tries robust local, robust border, ordinary-mean local, and ordinary-mean
border candidates, in that preference order. Any CRC-valid candidate can deliver
the chunk, but conflicting CRC-valid chunks are rejected. This is CRC arbitration,
not a vote that rewards a candidate for resembling expected diagnostic content.
Complete messages additionally pass full-content CRC and content validation.

## Retention and transition handling

The live receiver skips 300 ms after detecting a header transition. Invalid headers
contribute no evidence. Samples expire individually after the greater of 60 seconds
and two advertised complete sequence cycles:

`max(60,000 ms, 2 × holdMs × phases × logicalFrameCount)`.

Retained score arrays are limited to 64 MiB and eight logical frame groups. The
receiver first tries to retain every phase, reducing pool capacity and the minimum
sample target as needed. If even one capture per phase cannot fit, it uses a rolling
subset of phases sufficient for the requested first-attempt threshold and evicts
the least recently updated retained phase. A profile that cannot fit that threshold
is rejected. Older logical frame groups may also be evicted to enforce the total
budget. Effective target, pool capacity, phase retention limit, age limit, byte
count, and eviction count are reported. Temporary image and decoder allocations
are outside the retained-score cap.

New sequences require an explicit receiver reset. Changing receiver settings
resets evidence pools. A photo upload can decode one complete phase independently,
without claiming multi-phase fusion. CRC is not sender authentication.

## Diagnostics

Known diagnostic payloads are regenerated only after color classification. Raw
error rates count nonreference coded symbols, including parity and final bit
padding, before correction. Reports separate single exposures, ordinary fusion,
robust fusion, and accepted candidates for both color models. They record raw CRC
passes, ordinary RS passes (including raw-valid candidates), additional erasure
passes, first verified timing/capture, phase sample counts, and effective limits.
They retain confusion matrices and bounded histories without payloads or camera
images. Repeated rolling-window attempts are correlated; they are not independent
success trials. Lab elapsed time is processing time, not measured optical throughput.
