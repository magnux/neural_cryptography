"""Strict local M2 datasets. Labels are joined offline by the canonical JS codec.

No wire-format, phase, CRC, or Reed–Solomon implementation lives in Python.
Whole capture sessions and payloads, rather than adjacent frames, define splits.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DATASET_FORMAT = "optical-decoder-dataset"
SPLITS = ("train", "validation", "test")
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SHA256 = re.compile(r"^[a-f0-9]{64}$")


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

    @property
    def id(self):
        return self.metadata["id"]

    @property
    def shape(self):
        return self.metadata["size"], self.metadata["colors"]

    def read(self, name: str, dtype, count: int | None = None) -> np.ndarray:
        path = local_path(self.root, self.metadata[name])
        if path.stat().st_size > 4*1024*1024:
            raise ValueError(f"Dataset binary exceeds 4 MiB: {name}")
        if count is not None and path.stat().st_size != count * np.dtype(dtype).itemsize:
            raise ValueError(f"Wrong {name} byte count for {self.id}")
        expected = self.metadata.get("hashes", {}).get(name)
        if not isinstance(expected, str) or not SHA256.fullmatch(expected) or digest(path) != expected:
            raise ValueError(f"Missing or mismatched {name} hash for {self.id}")
        value = np.fromfile(path, dtype=dtype)
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite {name} values for {self.id}")
        return value

    def tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g, colors = self.shape
        pixels = self.read("input", "<f4", 3 * (4*g)**2).reshape(3, 4*g, 4*g)
        labels = self.read("labels", "u1", g*g).reshape(g, g)
        mask = self.read("mask", "u1", g*g).reshape(g, g)
        if np.max(np.abs(pixels)) > 1.50001:
            raise ValueError(f"Input is outside the normalized tensor contract: {self.id}")
        if np.any(labels >= colors) or np.any(mask > 1) or not np.any(mask):
            raise ValueError(f"Invalid palette labels or empty/nonbinary loss mask: {self.id}")
        return pixels, labels.astype(np.int64), mask.astype(bool)


class Dataset:
    """A set of manifests with globally checked IDs and split ownership."""

    def __init__(self, directories, split: str | None = None):
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.samples = []
        self.manifests = []
        owners = {"sessionId": {}, "payloadId": {}}
        ids = set()
        for location in directories:
            manifest = Path(location).resolve()
            if manifest.is_dir():
                manifest /= "manifest.json"
            if manifest.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("Dataset manifest exceeds 16 MiB")
            value = json.loads(manifest.read_text(encoding="utf8"))
            if value.get("format") != DATASET_FORMAT or value.get("version") != 1:
                raise ValueError("Unsupported dataset format; regenerate the current M2 dataset")
            entries = value.get("samples")
            if not isinstance(entries, list) or not entries or len(entries) > 50_000:
                raise ValueError("Dataset must contain 1–50,000 samples")
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
                if item.get("size") not in (64, 128) or item.get("colors") not in (4, 16):
                    raise ValueError("Unsupported decoder dataset profile")
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
                if split is None or item["split"] == split:
                    self.samples.append(Sample(manifest.parent, manifest, item))
        if not self.samples:
            raise ValueError(f"No samples for split {split!r}")
        self.samples.sort(key=lambda item: item.id)

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
