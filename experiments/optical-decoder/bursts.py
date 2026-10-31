"""Distinct exposures of one recorded matrix, without changing recording files.

Training samples random subsets within a physical/corruption burst. Evaluation
uses deterministic nonoverlapping windows; neither path mixes palette phases,
sources, sessions, or independent synthetic corruption trajectories.
"""

from dataclasses import dataclass
import json
import math
import re

import numpy as np

from contract import FRAME_COUNTS
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _burst_key(sample):
    row = sample.metadata
    burst = row.get("burstId")
    if not isinstance(burst, str) or not burst:
        raise ValueError("Four-frame input requires a recorded burstId; regenerate missing synthetic burst metadata")
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
            raise ValueError("Input frame count must be 1 or 4")
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
        # Group/member ordering uses only recorded identity and capture order,
        # never confidence, payload correctness or validation loss.
        for key in sorted(self.groups, key=repr):
            values = self.groups[key]
            count = len(values)//self.frames
            insufficient += int(count == 0)
            discarded += len(values) % self.frames
            for index in range(count):
                members = values[index*self.frames:(index+1)*self.frames]
                windows.append(BurstSample(members[0], members))
        return windows, {"framesPerInput": self.frames, "inputCaptures": len(self.samples),
                         "usedCaptures": len(windows)*self.frames,
                         "discardedCaptures": discarded, "insufficientBursts": insufficient,
                         "completeBursts": len(windows)}
