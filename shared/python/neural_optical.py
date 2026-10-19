"""Shared optical transport for the base neural and ZKBoo POCs. NumPy + stdlib.

Raw neural sequences retain all cells as data and rely on ordered frame delivery.
Framed byte payloads reserve 32 bytes per frame for indices, lengths and CRC.
Both formats use the same palettes, PNG writer, display and camera simulator.
"""
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import struct
import zipfile
import zlib
import numpy as np

FRAME_HEADER = struct.Struct(">4sBBH8sHHIII")


def add_optical_arguments(parser, *, size, colors, frames, frame_help):
    parser.add_argument("--size", "--matrix-size", type=int, default=size, help="matrix side, 8..512 cells")
    parser.add_argument("--colors", type=int, choices=(2, 4, 8, 16), default=colors)
    parser.add_argument("--frames", type=int, default=frames, help=frame_help)
    parser.add_argument("--fps", type=float, default=2., help="HTML sequence playback rate")
    parser.add_argument("--camera", action="store_true", help="simulate aligned, calibrated color capture")
    parser.add_argument("--camera-noise", type=float, default=.08, help="per-pixel RGB noise standard deviation")
    parser.add_argument("--cell-pixels", type=int, default=8, help="simulated sensor pixels per cell side")


def validate_optical_arguments(parser, args):
    if not 8 <= args.size <= 512 or not 1 <= args.frames <= 8:
        parser.error("size must be 8..512 and frames must be 1..8")
    if not math.isfinite(args.fps) or not .1 <= args.fps <= 60:
        parser.error("fps must be finite and in [0.1,60]")
    if not 2 <= args.cell_pixels <= 128 or not math.isfinite(args.camera_noise) or args.camera_noise < 0:
        parser.error("cell-pixels must be 2..128 and camera-noise must be finite/nonnegative")


def validate_grids(grids, colors, maximum_frames=64):
    palette(colors)
    grids = np.asarray(grids)
    if (grids.ndim != 3 or not 1 <= len(grids) <= maximum_frames or
            not 8 <= grids.shape[1] <= 512 or grids.shape[1] != grids.shape[2] or
            not np.issubdtype(grids.dtype, np.integer) or np.any(grids < 0) or np.any(grids >= colors)):
        raise ValueError("expected square color-symbol matrices within the configured limits")
    return grids.astype(np.uint8, copy=False)

def _sequence_identity(payload):
    # NPM1 is a standalone Python simulation format, separate from browser OBM5.
    fields = (b"NeuralOptical-NPM1", b"optical-sequence", payload)
    return hashlib.sha256(b"".join(struct.pack(">I", len(x)) + x for x in fields)).digest()[:8]

def palette(colors):
    """Nested, well-spaced RGB palettes from a fixed 4x4x4 candidate cube."""
    if colors not in (2, 4, 8, 16):
        raise ValueError("supported color counts are 2, 4, 8, 16")
    levels = (24, 96, 168, 240)
    candidates = np.array([(r, g, b) for r in levels for g in levels for b in levels], np.int32)
    chosen = [0, len(candidates) - 1]
    while len(chosen) < colors:
        distances = ((candidates[:, None] - candidates[chosen][None]) ** 2).sum(-1).min(-1)
        distances[chosen] = -1
        chosen.append(int(distances.argmax()))
    return candidates[chosen].astype(np.uint8)


def frame_capacity(size, colors):
    if type(size) is not int or not 8 <= size <= 512:
        raise ValueError("optical matrix size must be 8..512")
    palette(colors)
    return size * size * (colors.bit_length() - 1) // 8 - FRAME_HEADER.size


def frame_encode(payload, size, colors, max_frames):
    """Length-delimited binary packet -> indexed color grids with 32-byte headers.

    Every header includes sequence identity, frame index/count, total and chunk
    lengths, and CRC32. CRC detects channel damage; it is NOT authentication.
    """
    capacity = frame_capacity(size, colors)
    if not 1 <= max_frames <= 8:
        raise ValueError("this POC supports 1..8 frames per sequence")
    if capacity < 1:
        raise ValueError("matrix is too small for the 32-byte frame header; increase --size or --colors")
    needed = max(1, (len(payload) + capacity - 1) // capacity)
    if needed > max_frames:
        raise ValueError(f"Payload needs {needed} frames; --frames allows {max_frames}. "
                         f"Capacity is {capacity} payload bytes/frame. Increase --size/--colors/--frames.")
    bits_per_cell = colors.bit_length() - 1
    byte_capacity = size * size * bits_per_cell // 8
    identity = _sequence_identity(payload)
    grids = []
    for index in range(needed):
        chunk = payload[index * capacity:(index + 1) * capacity]
        head = FRAME_HEADER.pack(b"NPM1", 1, bits_per_cell, size, identity,
                                   index, needed, len(chunk), len(payload), 0)
        checksum = zlib.crc32(head[:-4] + chunk) & 0xffffffff
        packet = head[:-4] + struct.pack(">I", checksum) + chunk
        packet += bytes(byte_capacity - len(packet))
        bitstream = np.zeros(size * size * bits_per_cell, np.uint8)
        bitstream[:8 * len(packet)] = np.unpackbits(np.frombuffer(packet, dtype=np.uint8), bitorder="big")
        grouped = bitstream.reshape(-1, bits_per_cell)
        symbols = (grouped * (1 << np.arange(bits_per_cell - 1, -1, -1))).sum(axis=1).astype(np.uint8)
        grids.append(symbols.reshape(size, size))
    return np.stack(grids)


def frame_decode(grids, colors):
    """Accept out-of-order frames and identical repeats; reject missing/mixed data."""
    grids = np.asarray(grids)
    if grids.ndim != 3 or not 1 <= len(grids) <= 64 or grids.shape[1] != grids.shape[2]:
        raise ValueError("expected 1..64 square captures")
    size = grids.shape[1]
    capacity = frame_capacity(size, colors)
    if capacity < 1 or not np.issubdtype(grids.dtype, np.integer) or np.any(grids < 0) or np.any(grids >= colors):
        raise ValueError("invalid frame symbols")
    bits_per_cell = colors.bit_length() - 1
    chunks = {}
    sequence = None
    for grid in grids:
        bits = ((grid.ravel()[:, None] >> np.arange(bits_per_cell - 1, -1, -1)) & 1).astype(np.uint8).ravel()
        # A non-byte-aligned matrix has canonical zero pad bits at its end.
        usable = len(bits) // 8 * 8
        if np.any(bits[usable:]):
            raise ValueError("nonzero trailing symbol padding")
        packet = np.packbits(bits[:usable], bitorder="big").tobytes()
        magic, version, bpc, stored_size, identity, index, count, length, total, crc = FRAME_HEADER.unpack_from(packet)
        if magic != b"NPM1" or version != 1 or bpc != bits_per_cell or stored_size != size:
            raise ValueError("invalid frame header")
        if not 1 <= count <= 8 or not index < count or length > capacity or total > 8 * capacity:
            raise ValueError("invalid frame lengths")
        if count != max(1, (total + capacity - 1) // capacity):
            raise ValueError("noncanonical frame count")
        expected_length = min(capacity, max(0, total - index * capacity))
        if length != expected_length:
            raise ValueError("noncanonical chunk length")
        chunk = packet[FRAME_HEADER.size:FRAME_HEADER.size + length]
        if any(packet[FRAME_HEADER.size + length:]):
            raise ValueError("nonzero byte padding")
        if zlib.crc32(packet[:FRAME_HEADER.size - 4] + chunk) & 0xffffffff != crc:
            raise ValueError("frame CRC mismatch; capture again")
        current = (identity, count, total)
        if sequence is not None and current != sequence:
            raise ValueError("mixed frame sequences")
        sequence = current
        if index in chunks and chunks[index] != chunk:
            raise ValueError("conflicting repeated frame")
        chunks[index] = chunk
    identity, count, total = sequence
    if len(chunks) != count:
        raise ValueError(f"missing frames: received {len(chunks)}/{count}")
    payload = b"".join(chunks[i] for i in range(count))
    if len(payload) != total or _sequence_identity(payload) != identity:
        raise ValueError("sequence integrity check failed")
    return payload


def png(grid, palette_rgb):
    """Lossless indexed PNG, implemented with the standard library."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    height, width = grid.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in grid)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)) +
            chunk(b"PLTE", palette_rgb.tobytes()) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def save_transport(folder, name, payload, args):
    grids = frame_encode(payload, args.size, args.colors, args.frames)
    return save_sequence(folder, name, grids, args.colors, args.fps, title="Neural ZKBoo POC")


def save_sequence(folder, name, grids, colors, fps=2., title="Neural cryptography POC"):
    """Write raw symbol matrices, PNGs and an animated HTML viewer."""
    grids = validate_grids(grids, colors, maximum_frames=8)
    size = grids.shape[-1]
    if not math.isfinite(fps) or not .1 <= fps <= 60:
        raise ValueError("fps must be finite and in [0.1,60]")
    palette_rgb = palette(colors)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Remove only this named sequence's old PNGs when a later run emits fewer frames.
    for old in folder.glob(f"{name}_[0-9][0-9].png"):
        old.unlink()
    np.savez_compressed(folder / f"{name}_frames.npz", symbols=grids, colors=colors, palette=palette_rgb)
    urls = []
    for i, grid in enumerate(grids):
        encoded = png(grid, palette_rgb)
        (folder / f"{name}_{i + 1:02d}.png").write_bytes(encoded)
        urls.append("data:image/png;base64," + base64.b64encode(encoded).decode("ascii"))
    document = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DISPLAY_TITLE</title><style>
body{background:#12151c;color:#eee;font:16px system-ui;text-align:center;margin:16px}
img{width:min(84vw,80vh);height:auto;image-rendering:pixelated;border:12px solid white}
button{padding:10px;margin:8px}small{display:block;margin:10px}</style>
<h1>DISPLAY_TITLE</h1>
<p id="label"></p><div id="palette" style="display:flex;width:min(84vw,80vh);margin:8px auto"></div>
<img id="frame" alt="Encoded color matrix"><div>
<button id="toggle">Pause</button><button id="next">Next frame</button></div>
<small>Palette swatches above the matrix are calibration references. Hold the screen and camera aligned.
No perspective recovery or error correction.</small>
<script>const frames=FRAME_DATA;const palette=PALETTE_DATA;let i=0,playing=true;const label=document.getElementById('label');
for(const rgb of palette){const swatch=document.createElement('span');swatch.style.cssText='flex:1;height:24px;background:rgb('+rgb.join(',')+')';document.getElementById('palette').appendChild(swatch)}
function show(){document.getElementById('frame').src=frames[i];label.textContent=NAME+' — frame '+(i+1)+' / '+frames.length+' — SIZE × SIZE, COLORS colors';}
document.getElementById('toggle').onclick=()=>{playing=!playing;document.getElementById('toggle').textContent=playing?'Pause':'Play'};
document.getElementById('next').onclick=()=>{i=(i+1)%frames.length;show()};
setInterval(()=>{if(playing){i=(i+1)%frames.length;show()}},INTERVAL);show();</script>'''
    # Insert image data LAST: placeholder words can occur by chance in base64.
    document = (document.replace("DISPLAY_TITLE", html.escape(title)).replace("NAME", json.dumps(name)).replace("SIZE", str(size))
                .replace("COLORS", str(colors)).replace("INTERVAL", str(round(1000 / fps)))
                .replace("PALETTE_DATA", json.dumps(palette_rgb.tolist()))
                .replace("FRAME_DATA", json.dumps(urls)))
    (folder / f"{name}_display.html").write_text(document, encoding="utf-8")
    return grids


def camera_capture(grids, colors, seed, noise_std, cell_pixels):
    """Aligned central-patch RGB sampling; a scalable toy optical channel.

    Each RGB cell mean is perturbed by Gaussian pixel noise / sqrt(patch area).
    A frame-wide color gain/offset is calibrated using known palette swatches
    (assumed separately available). No perspective, demosaicing, clipping,
    motion blur, saturation, missing frames, or real camera acquisition model.
    """
    grids = validate_grids(grids, colors)
    if not math.isfinite(noise_std) or noise_std < 0 or not 2 <= cell_pixels <= 128:
        raise ValueError("invalid camera noise or cell-pixel setting")
    rng = np.random.default_rng(seed)
    palette_rgb = palette(colors).astype(np.float64) / 255
    decoded = []
    patch_side = max(1, cell_pixels // 2)
    mean_sigma = noise_std / patch_side
    for grid in grids:
        gain = rng.uniform(.85, 1.15, 3)
        offset = rng.uniform(-.025, .025, 3)
        observed_palette = palette_rgb * gain + offset + rng.normal(0, mean_sigma / 4, palette_rgb.shape)
        fitted_gain = np.empty(3)
        fitted_offset = np.empty(3)
        for c in range(3):
            fitted_gain[c], fitted_offset[c] = np.linalg.lstsq(
                np.column_stack((palette_rgb[:, c], np.ones(colors))), observed_palette[:, c], rcond=None)[0]
        rgb = palette_rgb[grid] * gain + offset + rng.normal(0, mean_sigma, (*grid.shape, 3))
        corrected = ((rgb - fitted_offset) / fitted_gain).reshape(-1, 3)
        # Chunked nearest palette avoids a size^2 * colors * 3 allocation.
        labels = np.empty(len(corrected), np.uint8)
        for start in range(0, len(corrected), 8192):
            patch = corrected[start:start + 8192]
            labels[start:start + len(patch)] = ((patch[:, None] - palette_rgb[None]) ** 2).sum(-1).argmin(-1)
        decoded.append(labels.reshape(grid.shape))
    return np.stack(decoded)


def load_captures(path):
    """Bound NPZ array dimensions BEFORE allocating or loading their contents."""
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("capture archive too large")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [entry.filename for entry in infos]
            if (len(names) != len(set(names)) or not {"symbols.npy", "colors.npy"} <= set(names)
                    or not set(names) <= {"symbols.npy", "colors.npy", "palette.npy"}
                    or sum(entry.file_size for entry in infos) > 20 * 1024 * 1024):
                raise ValueError("invalid capture archive entries/size")
            for name in ("symbols.npy", "colors.npy"):
                with archive.open(name) as handle:
                    version = np.lib.format.read_magic(handle)
                    readers = {(1, 0): np.lib.format.read_array_header_1_0,
                               (2, 0): np.lib.format.read_array_header_2_0}
                    if version not in readers:
                        raise ValueError("unsupported capture array format")
                    shape, fortran, dtype = readers[version](handle, max_header_size=4096)
                    if fortran or dtype.hasobject:
                        raise ValueError("invalid capture array encoding")
                    if name == "symbols.npy":
                        if (len(shape) != 3 or not 1 <= shape[0] <= 64 or
                                not 8 <= shape[1] <= 512 or shape[1] != shape[2] or dtype != np.dtype("uint8")):
                            raise ValueError("invalid captured symbol dimensions/type")
                    elif shape != () or dtype.kind not in "iu" or dtype.itemsize > 8:
                        raise ValueError("invalid captured color-count type")
                    if handle.tell() + math.prod(shape) * dtype.itemsize != archive.getinfo(name).file_size:
                        raise ValueError("capture array length mismatch")
        with np.load(path, allow_pickle=False) as capture:
            return capture["symbols"], int(capture["colors"])
    except (zipfile.BadZipFile, EOFError, zlib.error) as exc:
        raise ValueError("malformed capture archive") from exc
