# Shared Python optical simulation

`neural_optical.py` provides the palettes, grid validation, indexed PNG writer,
HTML sequence viewer, NumPy capture files, aligned camera simulation, and optical
CLI arguments used by both Python experiments. It depends on NumPy and the
Python standard library. The scripts locate this directory relative to their
own files, so their execution does not depend on the current working directory.

This is a small offline simulator. The real-camera color bench uses its own
current **OBM5** wire format, detector, pilot grid, phases, and error correction.
The Python display matrices are **not interoperable with OBM5 or QR transport**.
The compact proof bytes carried inside a Python packet are nevertheless the same
`NPZK0002` proof format as the browser proof engine.

## Current Python data formats

| Format | Use |
| --- | --- |
| Raw symbol sequence | Neural-dynamics input/output; every cell is data, frame order supplied by the caller |
| NPM1 packet sequence | Python-only serialization and channel simulation of arbitrary bytes, including compact proofs |
| NPZ captures | Decoded symbol arrays for playback, simulation, and the ZKBoo `--captured` option |
| Indexed PNG and HTML | Lossless generated symbol images and an animated sequence viewer |

An NPZ contains `symbols` as uint8 `[frames, side, side]`, scalar integer
`colors`, and optionally `palette` as `[colors, 3]` RGB values. Capture loading
bounds archive size and array dimensions before loading; `allow_pickle=False`
is enforced. Symbols belong to a 2/4/8/16-color palette. Display generation uses
up to eight frames and sides from 8 to 512 cells. Capture decoding permits up to
64 frames so that repeated captures can be supplied.

## NPM1 frame layout

Each frame reserves a 32-byte, big-endian header (`>4sBBH8sHHIII`):

| Field | Bytes |
| --- | ---: |
| Magic `NPM1` | 4 |
| Version `1` | 1 |
| Bits per cell | 1 |
| Matrix side | 2 |
| Sequence identity | 8 |
| Frame index | 2 |
| Frame count | 2 |
| Chunk byte count | 4 |
| Complete payload byte count | 4 |
| CRC32 | 4 |

The header is followed by chunk data and canonical zero padding. Bytes and color
symbols use most-significant-bit-first packing. Usable bytes per matrix are
`floor(side * side * log2(colors) / 8) - 32`. The sequence identity and CRC32
provide transport integrity checks; they are not authentication.

The decoder accepts reordered frames and identical duplicates. It rejects
missing/conflicting frames, mixed sequences, invalid lengths, nonzero padding,
and CRC damage. It implements only the current Python simulation framing; it
contains no browser or historical-format compatibility reader.

## Simulator limits

The camera simulation assumes aligned cell means and separately observed palette
swatches, fits per-channel gain and offset, and adds Gaussian noise. It does not
model screen subpixels, moire, clipping, motion blur, perspective, rolling
shutter, or real camera processing. Passing this simulation is a software
round-trip check, not evidence of physical optical reliability.
