"""Offline source/direction-balanced draws without changing capture formats.

Groups describe acquisition domains (for example pc_to_phone), never classes or
payload truth. Index once; draw a complete optimizer update so gradient
accumulation does not accidentally change source or direction quotas.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re


GROUP_FORMAT = "optical-decoder-session-groups"
MAX_GROUP_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _target_key(sample):
    """Exact target identity, excluding capture time and corruption parameters."""
    row = sample.metadata
    return (row["sessionId"], row["payloadId"],
            json.dumps(row["info"], sort_keys=True, separators=(",", ":"), allow_nan=False))


def _target_content(sample):
    row = sample.metadata
    hashes = tuple(row.get("hashes", {}).get(field) for field in ("labels", "mask", "expected"))
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes):
        raise ValueError("Paired targets require SHA-256 hashes for labels, mask and expected bytes")
    if row.get("split") not in ("train", "validation", "test"):
        raise ValueError("Paired targets require an explicit dataset split")
    return sample.shape, row.get("senderSessionId"), row["split"], hashes


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate session-group JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite session-group JSON value: {value}")


def _validate_mapping(mapping, *, nonempty=False):
    if not isinstance(mapping, dict) or (nonempty and not mapping):
        raise ValueError("Session groups must be a nonempty object" if nonempty else
                         "Session groups must be an object")
    for session, group in mapping.items():
        if (not isinstance(session, str) or not 1 <= len(session) <= 256
                or session.strip() != session or any(ord(c) < 32 for c in session)):
            raise ValueError("Invalid receiver session ID in session groups")
        if (not isinstance(group, str) or not 1 <= len(group) <= 64
                or group.strip() != group or any(ord(c) < 32 for c in group)
                or group.startswith("source:")):
            raise ValueError("Group names must be 1–64 characters and cannot start with source:")


def load_session_groups(path) -> dict:
    """Load the single current sidecar format; reject ambiguous duplicate keys."""
    path = Path(path)
    if path.stat().st_size > MAX_GROUP_BYTES:
        raise ValueError("Session-group file exceeds 1 MiB")
    value = json.loads(path.read_text(encoding="utf8"), object_pairs_hook=_object,
                       parse_constant=_invalid_constant)
    if (not isinstance(value, dict) or set(value) != {"format", "version", "sessions"}
            or value.get("format") != GROUP_FORMAT or type(value.get("version")) is not int
            or value["version"] != 1):
        raise ValueError("Expected optical-decoder-session-groups version 1")
    _validate_mapping(value["sessions"], nonempty=True)
    return value["sessions"]


def validate_session_groups(samples, mapping) -> set:
    """Require an explicit group for every real receiver session; ignore synthetic.

    Extra entries are allowed so the same sidecar can cover train/validation/test.
    The return value lists the real groups represented in this sample collection.
    """
    _validate_mapping(mapping)
    missing, groups = set(), set()
    for sample in samples:
        row = sample.metadata
        if row["source"] == "real":
            session = row["sessionId"]
            if session not in mapping:
                missing.add(session)
            else:
                groups.add(mapping[session])
    if missing:
        raise ValueError("Missing session groups for real receiver sessions: " + ", ".join(sorted(missing)))
    return groups


def infer_session_groups(samples) -> dict:
    """Use recorded direction hints, falling back to an explicit device-pair ID.

    Browser device classes are heuristics, not verified hardware identities.
    Unknown devices are never silently labeled phone/PC. A caller can replace
    this mapping with the existing explicit --session-groups sidecar.
    """
    result = {}
    for sample in samples:
        row = sample.metadata
        if row["source"] != "real":
            continue
        provenance = row.get("provenance", {})
        sender = provenance.get("senderEnvironment") or {}
        receiver = provenance.get("receiverEnvironment") or {}
        classes = (sender.get("deviceClass"), receiver.get("deviceClass"))
        if classes == ("mobile", "desktop"):
            group = "phone-to-pc"
        elif classes == ("desktop", "mobile"):
            group = "pc-to-phone"
        else:
            pair = row.get("devicePair") or {}
            identities = (pair.get("sender") or sender.get("deviceId"),
                          pair.get("receiver") or receiver.get("deviceId"))
            group = ("device-pair-" + hashlib.sha256(json.dumps(identities).encode()).hexdigest()[:12]
                     if all(isinstance(value, str) and value for value in identities)
                     else "unclassified")
        session = row["sessionId"]
        previous = result.setdefault(session, group)
        if previous != group:
            raise ValueError(f"Receiver session has conflicting device/direction metadata: {session}")
    _validate_mapping(result)
    return result


@dataclass(frozen=True)
class _Burst:
    index: int
    samples: tuple


class _Pool:
    """Immutable session → payload → phase → burst hierarchy."""

    def __init__(self, samples):
        tree = {}
        for sample in sorted(samples, key=lambda sample: sample.id):
            row = sample.metadata
            info = row.get("info", {})
            phase = info.get("phase", 0)
            burst = row.get("burstId") or f"{info.get('index', 0)}/{phase}"
            node = tree.setdefault(row["sessionId"], {}).setdefault(row["payloadId"], {})
            node.setdefault(phase, {}).setdefault(burst, []).append(sample)
        self.bursts = []

        def freeze(node, depth=0):
            if depth == 4:
                burst = _Burst(len(self.bursts), tuple(node))
                self.bursts.append(burst)
                return burst
            return tuple(freeze(node[key], depth+1) for key in sorted(node))

        self.tree = freeze(tree)
        self.size = len(samples)

    def hierarchical_burst(self, rng):
        node = self.tree
        while not isinstance(node, _Burst):
            node = rng.choice(node)
        return node

    def draw(self, rng, usage):
        # Usage contains only this update's draws, never a corpus-sized copy.
        used_bursts, used_samples = usage
        fresh_burst = len(used_bursts) < len(self.bursts)
        fresh_sample = len(used_samples) < self.size

        def eligible(burst):
            if fresh_burst:
                return burst.index not in used_bursts
            return not fresh_sample or len(used_bursts[burst.index]) < len(burst.samples)

        for _ in range(16):
            burst = self.hierarchical_burst(rng)
            if eligible(burst):
                break
        else:
            # A bounded fallback for very uneven trees: at most the number of
            # previously drawn bursts + 1 can be ineligible. No corpus filter.
            start = rng.randrange(len(self.bursts))
            for offset in range(len(self.bursts)):
                burst = self.bursts[(start+offset) % len(self.bursts)]
                if eligible(burst):
                    break
        previous = used_bursts.setdefault(burst.index, set())
        remaining = len(burst.samples) - len(previous)
        if previous and remaining:
            # Pick a uniform rank among unused captures. Starting at a random
            # array position and scanning forward would favor a capture directly
            # after already-used entries. Scan only this burst, without copying
            # its members or searching the full source pool.
            rank = rng.randrange(remaining)
            for sample in burst.samples:
                if sample.id not in previous:
                    if rank == 0:
                        break
                    rank -= 1
        else:
            sample = burst.samples[rng.randrange(len(burst.samples))]
        repeated_burst = bool(previous)
        repeated_sample = sample.id in used_samples
        previous.add(sample.id)
        used_samples.add(sample.id)
        return sample, repeated_burst, repeated_sample


class BalancedSampler:
    """Choose one profile per update, then exact rounded source/group quotas.

    Hierarchical draws prevent a session with many stored frames from dominating.
    Repeated bursts are avoided until a group's distinct bursts are exhausted;
    exact samples are reused only after that group's samples are exhausted.
    Paired mode draws one synthetic corruption of each chosen real target, with
    the same header/phase and verified target hashes, per optimizer update.
    """

    def __init__(self, samples, session_groups, seed=7, real_fraction=.5, *, paired_targets=False, require_all_groups=True):
        if (type(seed) is not int or not 0 <= seed <= 0xffffffff
                or isinstance(real_fraction, bool) or not math.isfinite(real_fraction)
                or not 0 <= real_fraction <= 1):
            raise ValueError("Sampler requires a uint32 seed and real fraction in [0, 1]")
        if type(paired_targets) is not bool or (paired_targets and real_fraction != .5):
            raise ValueError("Paired targets require real-fraction .5")
        samples = list(samples)
        if not samples:
            raise ValueError("Sampler needs at least one sample")
        validate_session_groups(samples, session_groups)
        buckets, ids = {}, set()
        for sample in samples:
            if sample.id in ids:
                raise ValueError(f"Duplicate sampler sample ID: {sample.id}")
            ids.add(sample.id)
            row = sample.metadata
            shape = sample.shape
            neural_qr = getattr(sample, "dataset_format", None) == "neural-qr-dataset"
            if ((neural_qr and (shape[0] not in (8, 16, 32, 64, 128, 256) or shape[1] != 16))
                    or (not neural_qr and (shape[0] not in (64, 128) or shape[1] not in (4, 16)))):
                raise ValueError("Unsupported sampler grid/palette profile")
            source = row["source"]
            if source not in ("real", "synthetic"):
                raise ValueError("Unsupported sampler source")
            group = session_groups[row["sessionId"]] if source == "real" else "source:synthetic"
            buckets.setdefault(shape, {}).setdefault(source, {}).setdefault(group, []).append(sample)
        required_groups = {group for shape in buckets.values() for group in shape.get("real", {})}
        for shape, sources in buckets.items():
            real_groups = set(sources.get("real", {}))
            if require_all_groups and real_groups and real_groups != required_groups:
                raise ValueError(f"Real profile {shape} is missing groups: {sorted(required_groups-real_groups)}")
        self.pools = {shape: {source: {group: _Pool(rows) for group, rows in sorted(groups.items())}
                              for source, groups in sources.items()} for shape, sources in buckets.items()}
        self.shapes = sorted(self.pools)
        self.rng = random.Random(seed)
        self.real_fraction = float(real_fraction)
        self.paired_targets = paired_targets
        self.paired_pools = {}
        if paired_targets:
            targets, synthetic = {}, {}
            for sample in samples:
                key, content = _target_key(sample), _target_content(sample)
                previous = targets.setdefault(key, content)
                if content != previous:
                    raise ValueError("Paired target metadata or labels/mask/expected hashes disagree")
                if sample.metadata["source"] == "synthetic":
                    synthetic.setdefault(key, []).append(sample)
            real_keys = {_target_key(sample) for sample in samples if sample.metadata["source"] == "real"}
            if not real_keys:
                raise ValueError("Paired targets require real captures and their synthetic counterparts")
            if real_keys - synthetic.keys():
                raise ValueError("Missing synthetic counterparts for real targets; regenerate targets and corruptions")
            if synthetic.keys() - real_keys:
                raise ValueError("Synthetic targets have no matching real captures; paired mode accepts only matched targets")
            self.paired_pools = {key: _Pool(rows) for key, rows in synthetic.items()}
        self.last_counts = {}
        # Parallel to the returned microbatches; duplicate anchors may appear
        # in one update, so sample IDs alone cannot identify a particular pair.
        self.last_pair_ids = []

    def draw_update(self, batch_size, accumulate):
        if (type(batch_size) is not int or not 1 <= batch_size <= 128
                or type(accumulate) is not int or not 1 <= accumulate <= 128):
            raise ValueError("Batch size and accumulation must be integers in [1, 128]")
        total = batch_size*accumulate
        if self.paired_targets and (total < 2 or total % 2):
            raise ValueError("Paired targets require an even effective update size of at least two")
        shape = self.rng.choice(self.shapes)
        pools = self.pools[shape]
        if self.paired_targets:
            real_count = total//2
        elif "real" not in pools:
            real_count = 0
        elif "synthetic" not in pools:
            real_count = total
        else:
            desired = total*self.real_fraction
            real_count = math.floor(desired) + int(self.rng.random() < desired-math.floor(desired))
        slots = []
        if real_count:
            groups = list(pools["real"])
            self.rng.shuffle(groups)
            base, remainder = divmod(real_count, len(groups))
            for index, group in enumerate(groups):
                slots.extend([("real", group)]*(base+int(index < remainder)))
        if not self.paired_targets:
            slots.extend([("synthetic", "source:synthetic")]*(total-real_count))
        self.rng.shuffle(slots)
        usage = {}
        counts = {"real": 0, "synthetic": 0, "groups": {}, "repeatedBursts": 0,
                  "repeatedSamples": 0, "pairedTargets": 0, "shape": list(shape)}
        chosen = []
        for source, group in slots:
            sample, repeated_burst, repeated_sample = pools[source][group].draw(
                self.rng, usage.setdefault((source, group), ({}, set())))
            pair_id = counts["pairedTargets"] if self.paired_targets else None
            chosen.append((sample, pair_id))
            counts[source] += 1
            if source == "real":
                counts["groups"][group] = counts["groups"].get(group, 0)+1
            counts["repeatedBursts"] += int(repeated_burst)
            counts["repeatedSamples"] += int(repeated_sample)
            if self.paired_targets:
                key = _target_key(sample)
                counterpart, repeated_burst, repeated_sample = self.paired_pools[key].draw(
                    self.rng, usage.setdefault(("paired", key), ({}, set())))
                chosen.append((counterpart, pair_id))
                counts["synthetic"] += 1
                counts["pairedTargets"] += 1
                counts["repeatedBursts"] += int(repeated_burst)
                counts["repeatedSamples"] += int(repeated_sample)
        if self.paired_targets:
            self.rng.shuffle(chosen)
        self.last_counts = counts
        self.last_pair_ids = [[pair_id for _, pair_id in chosen[start:start+batch_size]]
                              for start in range(0, total, batch_size)]
        return [[sample for sample, _ in chosen[start:start+batch_size]]
                for start in range(0, total, batch_size)]
