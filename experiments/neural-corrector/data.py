"""Strict local optical datasets. Labels are joined offline by the canonical JS codec.

No wire-format, phase, CRC, or Reed–Solomon implementation lives in Python.
Whole capture sessions and payloads, rather than adjacent frames, define splits.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DATASET_FORMAT = "optical-decoder-dataset"
NEURAL_QR_DATASET_FORMAT = "neural-qr-dataset"
MULTIRES_SIZES = (8, 16, 32, 64, 128, 256)
W_TUBE_SIZES = (16, 32, 64, 128)
SPLITS = ("train", "validation", "test")
# Offline corpora can exceed 256 MiB of metadata. Bound a single metadata value,
# not the entire corpus, and parse samples individually instead of making an
# additional full-size JSON string alongside the resident sampling metadata.
MAX_MANIFEST_VALUE_CHARS = 16 * 1024 * 1024
SHA256 = re.compile(r"^[a-f0-9]{64}$")
_JSON_SPACE = re.compile(r"[ \t\r\n]*")
# Metadata only: do not keep decoded RGB crops resident as the corpus grows.
# Include file identity/timestamps and expected digest, so rewrites and changed
# manifests invalidate verification. Each newly encountered file is hashed from
# the same bytes returned to the caller, rather than read twice from disk.
_MAX_VERIFIED_FILES = 32_768
_VERIFIED_READS = OrderedDict()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest JSON key: {key}")
        result[key] = value
    return result


def _finite_json(value):
    raise ValueError(f"Non-finite manifest JSON value: {value}")


class _ManifestJSON:
    """Incremental JSON tokens for an object with a potentially huge samples array."""

    def __init__(self, stream, chunk_size):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder(object_pairs_hook=_unique_object,
                                        parse_constant=_finite_json)

    def fill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        block = self.stream.read(self.chunk_size)
        self.eof = not block
        self.buffer += block
        # One chunk of read-ahead can belong to the next sample; bound the
        # decoded token separately below before returning it.
        if len(self.buffer) > MAX_MANIFEST_VALUE_CHARS+self.chunk_size:
            raise ValueError("Individual manifest JSON value exceeds 16 million characters")

    def peek(self):
        while True:
            self.position = _JSON_SPACE.match(self.buffer, self.position).end()
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f"Malformed manifest JSON: expected {character!r}")
        self.position += 1

    def value(self):
        if not self.peek():
            raise ValueError("Truncated manifest JSON value")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError(f"Malformed manifest JSON: {error.msg}") from error
            else:
                if end-self.position > MAX_MANIFEST_VALUE_CHARS:
                    raise ValueError("Individual manifest JSON value exceeds 16 million characters")
                # A number may continue in the next chunk (including an exponent).
                # Only accept a complete token followed by a delimiter or EOF.
                if end < len(self.buffer) and self.buffer[end] in " \t\r\n,:]}":
                    self.position = end
                    return value
                if self.eof:
                    if end != len(self.buffer):
                        raise ValueError("Malformed manifest JSON value delimiter")
                    self.position = end
                    return value
            self.fill()


def load_manifest(path, *, chunk_size=64*1024):
    """Read the unchanged manifest format without an overall byte-count ceiling.

    Samples remain resident for indexing, but their serialized text is consumed
    one entry at a time. Top-level key order is unrestricted; both pretty-printed
    and compact JSON are accepted. Binary crops continue to be loaded lazily.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Manifest chunk size must be positive")
    with Path(path).open("r", encoding="utf8") as stream:
        reader = _ManifestJSON(stream, chunk_size)
        reader.expect("{")
        result = {}
        if reader.peek() != "}":
            while True:
                key = reader.value()
                if not isinstance(key, str) or key in result:
                    raise ValueError("Manifest JSON keys must be unique strings")
                reader.expect(":")
                if key == "samples" and reader.peek() == "[":
                    reader.expect("[")
                    values = []
                    if reader.peek() != "]":
                        while True:
                            values.append(reader.value())
                            if reader.peek() == "]":
                                break
                            reader.expect(",")
                    reader.expect("]")
                    result[key] = values
                else:
                    result[key] = reader.value()
                if reader.peek() == "}":
                    break
                reader.expect(",")
        reader.expect("}")
        if reader.peek():
            raise ValueError("Trailing content after manifest JSON")
        return result


def _file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Dataset file paths must be nonempty relative paths")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Dataset path escapes its directory")
    return path


@dataclass(frozen=True)
class Sample:
    root: Path
    manifest: Path
    metadata: dict
    dataset_format: str = DATASET_FORMAT
    dataset_version: int = 1

    @property
    def pixels_per_cell(self):
        return 2 if self.dataset_format == NEURAL_QR_DATASET_FORMAT and self.dataset_version == 1 else 4

    @property
    def id(self):
        return self.metadata["id"]

    @property
    def shape(self):
        return self.metadata["size"], self.metadata["colors"]

    def read(self, name: str, dtype, count: int | None = None) -> np.ndarray:
        path = local_path(self.root, self.metadata[name])
        before = path.stat()
        if before.st_size > 4*1024*1024:
            raise ValueError(f"Dataset binary exceeds 4 MiB: {name}")
        if count is not None and before.st_size != count * np.dtype(dtype).itemsize:
            raise ValueError(f"Wrong {name} byte count for {self.id}")
        expected = self.metadata.get("hashes", {}).get(name)
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError(f"Missing or mismatched {name} hash for {self.id}")
        value = np.fromfile(path, dtype=dtype)
        identity = _file_identity(before)
        if _file_identity(path.stat()) != identity or value.nbytes != before.st_size:
            raise ValueError(f"Dataset binary changed while reading: {name}")
        cache_key = str(path), identity, expected
        if cache_key not in _VERIFIED_READS:
            if hashlib.sha256(memoryview(value).cast("B")).hexdigest() != expected:
                raise ValueError(f"Missing or mismatched {name} hash for {self.id}")
            _VERIFIED_READS[cache_key] = True
            if len(_VERIFIED_READS) > _MAX_VERIFIED_FILES:
                _VERIFIED_READS.popitem(last=False)
        else:
            _VERIFIED_READS.move_to_end(cache_key)
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite {name} values for {self.id}")
        return value

    def tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g, colors = self.shape
        side = self.pixels_per_cell*g
        pixels = self.read("input", "<f4", 3 * side**2).reshape(3, side, side)
        labels = self.read("labels", "u1", g*g).reshape(g, g)
        mask = self.read("mask", "u1", g*g).reshape(g, g)
        if np.max(np.abs(pixels)) > 1.50001:
            raise ValueError(f"Input is outside the normalized tensor contract: {self.id}")
        if np.any(labels >= colors) or np.any(mask > 1) or not np.any(mask):
            raise ValueError(f"Invalid palette labels or empty/nonbinary loss mask: {self.id}")
        if self.dataset_format == NEURAL_QR_DATASET_FORMAT:
            from phases import canonical_labels
            expected = self.read("expected", "u1", g*g).reshape(g, g)
            if (self.metadata["hashes"]["expected"] != self.metadata["payloadId"]
                    or not np.array_equal(canonical_labels(labels, mask, self.metadata["info"]["phase"]), expected)):
                raise ValueError(f"Neural-QR phase labels disagree with their canonical target: {self.id}")
        return pixels, labels.astype(np.int64), mask.astype(bool)


class Dataset:
    """A set of manifests with globally checked IDs and split ownership."""

    def __init__(self, directories, split: str | None = None):
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.samples = []
        self.manifests = []
        self.dataset_format = None
        self.dataset_version = None
        owners = {"sessionId": {}, "payloadId": {}}
        ids = set()
        manifests = []
        for location in directories:
            path = Path(location).resolve()
            found = ([path/"manifest.json"] if (path/"manifest.json").is_file()
                     else sorted(path.rglob("manifest.json"))) if path.is_dir() else [path]
            if not found:
                raise ValueError(f"No dataset manifests beneath {path}")
            manifests.extend(found)
        if len(set(manifests)) != len(manifests):
            raise ValueError("Duplicate dataset manifest paths")
        for manifest in manifests:
            value = load_manifest(manifest)
            dataset_format = value.get("format")
            version = value.get("version")
            if (dataset_format not in (DATASET_FORMAT, NEURAL_QR_DATASET_FORMAT)
                    or type(version) is not int
                    or version not in ((1, 2) if dataset_format == NEURAL_QR_DATASET_FORMAT else (1,))):
                raise ValueError("Unsupported optical dataset format/version")
            if self.dataset_format is not None and (self.dataset_format, self.dataset_version) != (dataset_format, version):
                raise ValueError("Cannot mix dataset formats or raster versions in one training/evaluation set")
            self.dataset_format, self.dataset_version = dataset_format, version
            entries = value.get("samples")
            if not isinstance(entries, list) or not entries:
                raise ValueError("Dataset samples must be a nonempty list")
            self.manifests.append({"path": str(manifest), "sha256": digest(manifest)})
            for item in entries:
                if not isinstance(item, dict):
                    raise ValueError("Invalid sample metadata")
                for name in ("id", "sessionId", "payloadId"):
                    if not isinstance(item.get(name), str) or not 1 <= len(item[name]) <= 256:
                        raise ValueError(f"Missing/bad {name}")
                if not SHA256.fullmatch(item["payloadId"]):
                    raise ValueError("Payload identity must be its whole-payload SHA-256")
                if item["id"] in ids:
                    raise ValueError(f"Duplicate sample ID across manifests: {item['id']}")
                ids.add(item["id"])
                if item.get("split") not in SPLITS or item.get("source") not in ("synthetic", "real"):
                    raise ValueError("Explicit split and source are required")
                if dataset_format == NEURAL_QR_DATASET_FORMAT:
                    g = item.get("size")
                    ppc, sizes = (2, MULTIRES_SIZES) if version == 1 else (4, W_TUBE_SIZES)
                    if (type(g) is not int or g not in sizes or item.get("colors") != 16
                            or type(item.get("pixelsPerCell")) is not int or item["pixelsPerCell"] != ppc
                            or item.get("inputShape") != [3, ppc*g, ppc*g]):
                        raise ValueError(f"Neural-QR v{version} requires grids {sizes}, 16 colors and {ppc}x{ppc} pixels per cell")
                elif item.get("size") not in (64, 128) or item.get("colors") not in (4, 16):
                    raise ValueError("Unsupported neural-corrector dataset profile")
                for name in ("sessionId", "payloadId"):
                    identities = [item[name]]
                    if name == "sessionId" and item.get("senderSessionId") is not None:
                        sender = item["senderSessionId"]
                        if not isinstance(sender, str) or not 1 <= len(sender) <= 128:
                            raise ValueError("Invalid sender session identity")
                        identities.append(sender)
                    for identity in identities:
                        previous = owners[name].setdefault(identity, item["split"])
                        if previous != item["split"]:
                            raise ValueError(f"{name} crosses dataset splits: {identity}")
                for name in ("input", "labels", "mask", "baseline", "expected"):
                    local_path(manifest.parent, item.get(name))
                    if not SHA256.fullmatch(str(item.get("hashes", {}).get(name, ""))):
                        raise ValueError(f"Missing SHA-256 for {name}")
                if not isinstance(item.get("info"), dict):
                    raise ValueError("Missing canonical decoder frame metadata")
                if item["info"].get("size") != item["size"] or item["info"].get("colors") != item["colors"]:
                    raise ValueError("Canonical frame metadata disagrees with the tensor profile")
                if dataset_format == NEURAL_QR_DATASET_FORMAT:
                    from phases import phase_neutral_info
                    info = item["info"]
                    phase_neutral_info(info)
                    if (info.get("pixelsPerCell") != ppc or info.get("gridBlock") != 8
                            or info.get("parity") != 0):
                        raise ValueError("Unsupported current Neural-QR phase/reference geometry")
                    if item["hashes"]["expected"] != item["payloadId"]:
                        raise ValueError("Neural-QR target identity must hash the logical matrix")
                if split is None or item["split"] == split:
                    self.samples.append(Sample(manifest.parent, manifest, item, dataset_format, version))
        if not self.samples:
            raise ValueError(f"No samples for split {split!r}")
        self.samples.sort(key=lambda item: item.id)

    @property
    def pixels_per_cell(self):
        return 2 if self.dataset_format == NEURAL_QR_DATASET_FORMAT and self.dataset_version == 1 else 4

    @property
    def session_ids(self):
        return {identity for sample in self.samples for identity in
                (sample.metadata["sessionId"], sample.metadata.get("senderSessionId")) if identity is not None}

    @property
    def payload_ids(self):
        return {sample.metadata["payloadId"] for sample in self.samples}

    @property
    def manifest_hash(self):
        values = "\n".join(sorted(entry["sha256"] for entry in self.manifests))
        return hashlib.sha256(values.encode()).hexdigest()

    def groups(self):
        groups = {}
        for sample in self.samples:
            groups.setdefault(sample.shape, []).append(sample)
        return groups


def assert_disjoint(*datasets: Dataset) -> None:
    for index, first in enumerate(datasets):
        for other in datasets[index+1:]:
            if first.session_ids & other.session_ids:
                raise ValueError("Capture session overlap between dataset splits")
            if first.payload_ids & other.payload_ids:
                raise ValueError("Whole-payload overlap between dataset splits")


def check_checkpoint_overlap(checkpoint: dict, dataset: Dataset, split: str) -> None:
    """A test split must also be disjoint from checkpoint-selection sessions."""
    provenance = checkpoint.get("training", {})
    prefixes = ("train", "validation") if split == "test" else ("train",)
    for prefix in prefixes:
        if dataset.session_ids & set(provenance.get(prefix+"SessionIds", [])):
            raise ValueError(f"Evaluation sessions overlap checkpoint {prefix} data")
        if dataset.payload_ids & set(provenance.get(prefix+"PayloadIds", [])):
            raise ValueError(f"Evaluation payloads overlap checkpoint {prefix} data")

