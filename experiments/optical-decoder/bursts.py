"""Distinct exposures of one recorded matrix, without changing recording files.

Training samples random subsets within a physical/corruption burst. Evaluation
fixes a seeded random member order before nonoverlapping grouping, with nested
presets for variable observations. Fixed inputs remain within one phase;
phase-paired inputs combine four palette phases of one logical matrix, never
different sessions, payloads, or independent synthetic trajectories.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np

from contract import FRAME_COUNTS, MAX_FRAMES
from phases import canonical_labels, phase_neutral_info
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
# One fixed held-out draw, shared by training validation and standalone evaluation.
# SHA-256 ranking makes that draw portable without relying on random.shuffle's
# implementation or on the order in which other bursts consume a shared RNG.
EVALUATION_SEED = 7
EVALUATION_POLICY = "fixed-seeded-shuffle-within-burst"
VARIABLE_EVALUATION_POLICY = "fixed-seeded-shuffle-within-phase/nested-phase-pairs"


def _burst_key(sample):
    row = sample.metadata
    burst = row.get("burstId")
    if not isinstance(burst, str) or not burst:
        raise ValueError("Multiframe input requires a recorded burstId; regenerate missing synthetic burst metadata")
    return (str(sample.manifest), row["source"], row["sessionId"],
            row.get("senderSessionId"), row["payloadId"], row["split"],
            json.dumps(row["info"], sort_keys=True, separators=(",", ":"), allow_nan=False), burst)


def _phase_group_key(sample):
    row = sample.metadata
    _burst_key(sample)  # Require current recording burst metadata, even across phases.
    info = phase_neutral_info(row["info"])
    trajectory = None
    if row["source"] == "synthetic":
        trajectory = row.get("synthetic", {}).get("phaseGroupId")
        if not isinstance(trajectory, str) or not 1 <= len(trajectory) <= 256:
            raise ValueError("Phase-paired synthetic inputs require synthetic.phaseGroupId; regenerate coherent phase groups")
    return (str(sample.manifest), row["source"], row["sessionId"],
            row.get("senderSessionId"), row["payloadId"], row["split"],
            json.dumps(info, sort_keys=True, separators=(",", ":"), allow_nan=False), trajectory)


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
    slots: tuple | None = None
    capacity: int | None = None
    phase_grouped: bool = False

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

    def raw_tensors(self):
        """Load actual RGB observations before augmentation or padding."""
        values = [member.tensors() for member in self.members]
        targets = [(canonical_labels(row[1], row[2], member.metadata["info"]["phase"])
                    if self.phase_grouped else row[1])
                   for member, row in zip(self.members, values)]
        labels, mask = targets[0], values[0][2]
        if any(not np.array_equal(target, labels) or not np.array_equal(row[2], mask)
               for target, row in zip(targets[1:], values[1:])):
            raise ValueError("Burst exposures disagree on labels or data-cell mask")
        pixels = values[0][0] if len(values) == 1 else np.concatenate([row[0] for row in values], axis=0)
        return pixels, labels, mask

    def pack_pixels(self, pixels):
        """Pad only after augmentation; presence planes never receive color noise."""
        if self.slots is None:
            return pixels
        if (self.capacity != 8 or len(self.slots) != len(self.members)
                or len(set(self.slots)) != len(self.slots)
                or any(type(slot) is not int or not 0 <= slot < self.capacity for slot in self.slots)
                or pixels.dtype != np.float32 or pixels.ndim != 3
                or pixels.shape[0] != 3*len(self.members)):
            raise ValueError("Invalid variable-frame RGB observations or slot assignment")
        if self.phase_grouped and any(slot//2 != member.metadata["info"]["phase"]
                                      for member, slot in zip(self.members, self.slots)):
            raise ValueError("Phase-paired input slots disagree with capture phase")
        packed = np.zeros((4*self.capacity, *pixels.shape[1:]), dtype=np.float32)
        for index, slot in enumerate(self.slots):
            packed[3*slot:3*slot+3] = pixels[3*index:3*index+3]
            packed[3*self.capacity+slot] = 1.
        return packed

    def tensors(self):
        pixels, labels, mask = self.raw_tensors()
        return self.pack_pixels(pixels), labels, mask


class BurstIndex:
    """Index once; choose actual members separately from optional empty slots."""

    def __init__(self, samples, frames=1, *, min_frames=None, presence=False):
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"Input frame count must be 1 to {MAX_FRAMES}")
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("Burst input requires recorded exposures")
        self.frames = frames
        self.min_frames = frames if min_frames is None else min_frames
        if (type(presence) is not bool or type(self.min_frames) is not int
                or (presence and (frames != 8 or self.min_frames != 4))
                or (not presence and self.min_frames != frames)):
            raise ValueError("Presence inputs require capacity 8 and counts 4–8; fixed inputs require one count")
        self.presence = presence
        self.by_id = {}
        groups = {}
        targets = {}
        for sample in self.samples:
            if sample.id in self.by_id:
                raise ValueError("Duplicate capture ID in burst input")
            key = _phase_group_key(sample) if presence else _burst_key(sample) if frames > 1 else (sample.id,)
            if frames > 1:
                target = _target_hashes(sample)
                target_key = (key, sample.metadata["info"]["phase"]) if presence else key
                previous = targets.setdefault(target_key, target)
                if target != previous:
                    raise ValueError("A burst contains inconsistent target hashes or profiles")
            groups.setdefault(key, []).append(sample)
            self.by_id[sample.id] = key
        self.groups = {key: tuple(sorted(values, key=_time_order)) for key, values in groups.items()}
        self.phase_groups = {}
        if presence:
            for key, values in self.groups.items():
                phases = tuple(tuple(row for row in values if row.metadata["info"]["phase"] == phase)
                               for phase in range(4))
                # Physical label hashes differ by phase. The mask and expected
                # logical payload must agree before any member is sampled.
                common = {(row.shape, row.metadata["hashes"]["mask"], row.metadata["hashes"]["expected"])
                          for row in values}
                if len(common) != 1:
                    raise ValueError("Phase group has inconsistent masks, payloads or profiles")
                self.phase_groups[key] = phases

    def validate_training(self):
        if self.presence:
            short = sum(any(len(phase) < 2 for phase in phases) for phases in self.phase_groups.values())
            if short:
                raise ValueError(f"{short} training phase group(s) need at least two distinct captures per phase (four phases)")
            return
        short = [values for values in self.groups.values() if len(values) < self.frames]
        if short:
            raise ValueError(f"{len(short)} training burst(s) have fewer than {self.frames} distinct exposures; "
                             "collect more frames or regenerate synthetic bursts with sufficient --frames")

    def train_sample(self, anchor, rng, actual_frames=None):
        if anchor.id not in self.by_id:
            raise ValueError("Training anchor is not in this burst index")
        if self.presence:
            count = rng.randint(self.min_frames, self.frames) if actual_frames is None else actual_frames
            if type(count) is not int or not self.min_frames <= count <= self.frames:
                raise ValueError("Actual frame count is outside this model's observation range")
            phases = self.phase_groups[self.by_id[anchor.id]]
            if any(len(values) < 2 for values in phases):
                raise ValueError("Training phase groups need at least two distinct captures per phase")
            doubled = set(rng.sample(range(4), count-4))
            members, slots = [], []
            for phase, values in enumerate(phases):
                n = 2 if phase in doubled else 1
                if anchor in values:
                    selected = [anchor, *rng.sample([value for value in values if value.id != anchor.id], n-1)]
                else:
                    selected = rng.sample(values, n)
                rng.shuffle(selected)
                members.extend(selected)
                slots.extend(rng.sample(range(2*phase, 2*phase+2), n))
            return BurstSample(anchor, tuple(members), tuple(slots), self.frames, True)
        values = self.groups[self.by_id[anchor.id]]
        if len(values) < self.frames:
            raise ValueError(f"Burst has fewer than {self.frames} distinct exposures")
        count = self.frames if actual_frames is None else actual_frames
        if type(count) is not int or not self.min_frames <= count <= self.frames:
            raise ValueError("Actual frame count is outside this model's observation range")
        if self.frames == 1:
            return BurstSample(anchor, (anchor,))
        others = [sample for sample in values if sample.id != anchor.id]
        members = [anchor, *rng.sample(others, count-1)]
        rng.shuffle(members)
        return BurstSample(anchor, tuple(members))

    def eval_samples(self, actual_frames=None):
        actual_frames = self.frames if actual_frames is None else actual_frames
        if type(actual_frames) is not int or not self.min_frames <= actual_frames <= self.frames:
            raise ValueError("Actual frame count is outside this model's observation range")
        if self.presence:
            return self._phase_eval_samples(actual_frames)
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
                block = values[index*self.frames:(index+1)*self.frames]
                members = block[:actual_frames]
                windows.append(BurstSample(members[0], members))
        stats = {"method": EVALUATION_POLICY,
                         "seed": EVALUATION_SEED,
                         "framesPerInput": actual_frames, "inputCaptures": len(self.samples),
                         "usedCaptures": len(windows)*actual_frames,
                         "discardedCaptures": discarded, "insufficientBursts": insufficient,
                         "completeBursts": len(windows)}
        return windows, stats

    def _phase_eval_samples(self, actual_frames):
        """Fixed shuffled pairs per phase; every K sees a nested observation set.

        Full eight-capture blocks fix the same candidate denominator for all
        presets. Live inference only needs one capture per phase to attempt K4.
        """
        windows, discarded, insufficient = [], 0, 0
        groups = sorted(self.phase_groups.values(), key=lambda phases:
                        tuple(_evaluation_identity(values[0]) if values else "" for values in phases))
        for original in groups:
            phases = tuple(_evaluation_order(values) if values else () for values in original)
            count = min(len(values)//2 for values in phases)
            insufficient += int(count == 0)
            discarded += sum(map(len, phases))-8*count
            for index in range(count):
                pairs = [values[2*index:2*index+2] for values in phases]
                anchor = pairs[0][0]
                identity = _evaluation_identity(anchor)
                def rank(tag, value):
                    material = json.dumps([VARIABLE_EVALUATION_POLICY, EVALUATION_SEED,
                                           identity, anchor.id, tag, value], separators=(",", ":")).encode()
                    return hashlib.sha256(material).digest()
                extra_order = sorted(range(4), key=lambda phase: rank("extra", phase))
                first_slots = [2*phase + (rank("slot", phase)[0] & 1) for phase in range(4)]
                members = [pair[0] for pair in pairs]
                slots = list(first_slots)
                for phase in extra_order[:actual_frames-4]:
                    members.append(pairs[phase][1])
                    slots.append(first_slots[phase] ^ 1)
                windows.append(BurstSample(anchor, tuple(members), tuple(slots), self.frames, True))
        stats = {"method": VARIABLE_EVALUATION_POLICY, "seed": EVALUATION_SEED,
                 "framesPerInput": actual_frames, "frameCapacity": self.frames,
                 "phaseCounts": 4, "capturesPerPhase": [1, 2],
                 "inputCaptures": len(self.samples), "usedCaptures": len(windows)*actual_frames,
                 "discardedCaptures": len(self.samples)-len(windows)*actual_frames,
                 "remainderCaptures": discarded,
                 "unusedCapacityCaptures": len(windows)*(self.frames-actual_frames),
                 "insufficientBursts": insufficient, "completeBursts": len(windows)}
        return windows, stats
