"""Distinct exposures of one recorded matrix, without changing recording files.

Training samples random subsets within a physical/corruption burst. Evaluation
fixes a seeded random member order before nonoverlapping grouping; neither path
mixes palette phases, sources, sessions, or independent synthetic trajectories.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np

from contract import FRAME_COUNTS, MAX_FRAMES
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
# One fixed held-out draw, shared by training validation and standalone evaluation.
# SHA-256 ranking makes that draw portable without relying on random.shuffle's
# implementation or on the order in which other bursts consume a shared RNG.
EVALUATION_SEED = 7
EVALUATION_POLICY = "fixed-seeded-shuffle-within-burst"


def _burst_key(sample):
    row = sample.metadata
    burst = row.get("burstId")
    if not isinstance(burst, str) or not burst:
        raise ValueError("Multiframe input requires a recorded burstId; regenerate missing synthetic burst metadata")
    return (str(sample.manifest), row["source"], row["sessionId"],
            row.get("senderSessionId"), row["payloadId"], row["split"],
            json.dumps(row["info"], sort_keys=True, separators=(",", ":"), allow_nan=False), burst)


def _target_hashes(sample):
    hashes = tuple(sample.metadata.get("hashes", {}).get(name)
                   for name in ("labels", "mask", "expected"))
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes):
        raise ValueError("Burst samples require target label/mask/expected SHA-256 hashes")
    return sample.shape, hashes


def _time_order(sample):
    row = sample.metadata
    # Synthetic frame position is authoritative within its independent burst;
    # clean target timestamps are shared by its generated exposures.
    value = row.get("synthetic", {}).get("frame", row.get("timestamp", 0))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Burst frame timestamps/indices must be finite numbers")
    return value, sample.id


def _evaluation_identity(sample):
    """Portable recorded identity, excluding storage paths and target hashes."""
    row = sample.metadata
    identity = [row["source"], row["sessionId"], row.get("senderSessionId"),
                row["payloadId"], row["split"], row["info"], row.get("burstId")]
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _evaluation_order(values):
    identity = _evaluation_identity(values[0])

    def rank(sample):
        material = json.dumps([EVALUATION_POLICY, EVALUATION_SEED, identity, sample.id],
                              ensure_ascii=True, separators=(",", ":")).encode("utf8")
        return hashlib.sha256(material).digest(), sample.id

    return tuple(sorted(values, key=rank))


@dataclass(frozen=True)
class BurstSample:
    anchor: object
    members: tuple

    @property
    def id(self):
        return self.anchor.id

    @property
    def metadata(self):
        return self.anchor.metadata

    @property
    def shape(self):
        return self.anchor.shape

    @property
    def root(self):
        return self.anchor.root

    @property
    def manifest(self):
        return self.anchor.manifest

    def read(self, *args, **kwargs):
        return self.anchor.read(*args, **kwargs)

    def tensors(self):
        values = [member.tensors() for member in self.members]
        labels, mask = values[0][1:]
        if any(not np.array_equal(row[1], labels) or not np.array_equal(row[2], mask)
               for row in values[1:]):
            raise ValueError("Burst exposures disagree on labels or data-cell mask")
        pixels = values[0][0] if len(values) == 1 else np.concatenate([row[0] for row in values], axis=0)
        return pixels, labels, mask


class BurstIndex:
    """Index once; choose fresh members without corpus scans or frame padding."""

    def __init__(self, samples, frames=1):
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Input frame count must be 1 to {MAX_FRAMES}")
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("Burst input requires recorded exposures")
        self.frames = frames
        self.by_id = {}
        groups = {}
        targets = {}
        for sample in self.samples:
            if sample.id in self.by_id:
                raise ValueError("Duplicate capture ID in burst input")
            key = _burst_key(sample) if frames > 1 else (sample.id,)
            if frames > 1:
                target = _target_hashes(sample)
                previous = targets.setdefault(key, target)
                if target != previous:
                    raise ValueError("A burst contains inconsistent target hashes or profiles")
            groups.setdefault(key, []).append(sample)
            self.by_id[sample.id] = key
        self.groups = {key: tuple(sorted(values, key=_time_order)) for key, values in groups.items()}

    def validate_training(self):
        short = [values for values in self.groups.values() if len(values) < self.frames]
        if short:
            raise ValueError(f"{len(short)} training burst(s) have fewer than {self.frames} distinct exposures; "
                             "collect more frames or regenerate synthetic bursts with sufficient --frames")

    def train_sample(self, anchor, rng):
        if anchor.id not in self.by_id:
            raise ValueError("Training anchor is not in this burst index")
        values = self.groups[self.by_id[anchor.id]]
        if len(values) < self.frames:
            raise ValueError(f"Burst has fewer than {self.frames} distinct exposures")
        if self.frames == 1:
            return BurstSample(anchor, (anchor,))
        others = [sample for sample in values if sample.id != anchor.id]
        members = [anchor, *rng.sample(others, self.frames-1)]
        rng.shuffle(members)
        return BurstSample(anchor, tuple(members))

    def eval_samples(self):
        windows = []
        discarded, insufficient = 0, 0
        # Fix one random permutation per burst before forming disjoint groups.
        # Ranking never uses pixel values, target hashes, decoder predictions,
        # validation loss, timestamps or absolute manifest paths. In particular,
        # adding another burst cannot consume RNG state and change this draw.
        groups = sorted(self.groups.values(), key=lambda values:
                        (_evaluation_identity(values[0]), tuple(sorted(s.id for s in values))))
        for original in groups:
            values = _evaluation_order(original)
            count = len(values)//self.frames
            insufficient += int(count == 0)
            discarded += len(values) % self.frames
            for index in range(count):
                members = values[index*self.frames:(index+1)*self.frames]
                windows.append(BurstSample(members[0], members))
        return windows, {"method": EVALUATION_POLICY, "seed": EVALUATION_SEED,
                         "framesPerInput": self.frames, "inputCaptures": len(self.samples),
                         "usedCaptures": len(windows)*self.frames,
                         "discardedCaptures": discarded, "insufficientBursts": insufficient,
                         "completeBursts": len(windows)}
