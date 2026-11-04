"""Direction/source quotas, target pairing and burst diversity for training."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
from sampling import (BalancedSampler, GROUP_FORMAT, MAX_GROUP_BYTES, _Pool,
                      load_session_groups, validate_session_groups)


def examples(source, session, *, payloads=1, phases=4, frames=3, shape=(64, 16)):
    rows = []
    for payload in range(payloads):
        for phase in range(phases):
            for frame in range(frames):
                rows.append(SimpleNamespace(id=f"{shape}/{session}/{payload}/{phase}/{frame}", shape=shape,
                    metadata={"source": source, "sessionId": session, "payloadId": f"{session}-{payload}",
                              "info": {"phase": phase}, "burstId": f"{session}/{payload}/{phase}"}))
    return rows


def flat(batches):
    return [sample for batch in batches for sample in batch]


def paired_examples(session, *, variants=3, **kwargs):
    real = examples("real", session, **kwargs)
    synthetic, seen = [], set()
    for sample in real:
        row = sample.metadata
        row.update(split="train", senderSessionId=f"sender-{session}")
        row["info"].update(size=sample.shape[0], colors=sample.shape[1], index=0)
        target = (row["sessionId"], row["payloadId"], json.dumps(row["info"], sort_keys=True))
        row["hashes"] = {field: hashlib.sha256(f"{target}/{field}".encode()).hexdigest()
                         for field in ("labels", "mask", "expected")}
        if target in seen:
            continue
        seen.add(target)
        for variant in range(variants):
            metadata = copy.deepcopy(row)
            metadata.update(source="synthetic", timestamp=10000+variant)
            synthetic.append(SimpleNamespace(id=f"synthetic/{sample.id}/{variant}",
                shape=sample.shape, metadata=metadata))
    return real+synthetic


def target_counts(rows, source):
    return Counter((r.metadata["sessionId"], r.metadata["payloadId"],
                    json.dumps(r.metadata["info"], sort_keys=True))
                   for r in rows if r.metadata["source"] == source)


class SamplingTests(unittest.TestCase):
    def test_sidecar_validation_rejects_ambiguous_or_incomplete_groups(self):
        base = ROOT/"validation/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            path = Path(temporary)/"groups.json"
            value = {"format": GROUP_FORMAT, "version": 1, "sessions": {"a": "pc_to_phone", "b": "phone_to_pc"}}
            path.write_text(json.dumps(value))
            self.assertEqual(load_session_groups(path), value["sessions"])
            for raw in ('{"format":"optical-decoder-session-groups","version":1,"sessions":{"a":"x","a":"y"}}',
                        json.dumps({**value, "version": True}),
                        json.dumps({**value, "sessions": {"a": "source:synthetic"}}),
                        json.dumps({**value, "sessions": {}}),
                        json.dumps({**value, "sessions": {"a": float("nan")}})):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    load_session_groups(path)
            path.write_text(" "*(MAX_GROUP_BYTES+1))
            with self.assertRaisesRegex(ValueError, "1 MiB"):
                load_session_groups(path)
        with self.assertRaisesRegex(ValueError, "Missing session groups"):
            validate_session_groups(examples("real", "a"), {})
        self.assertEqual(validate_session_groups(examples("synthetic", "s"), {}), set())

    def test_exact_update_quotas_and_distinct_bursts_despite_unequal_frame_counts(self):
        samples = (examples("real", "a", frames=20) + examples("real", "b", frames=1)
                   + examples("real", "c", frames=1) + examples("synthetic", "s", payloads=3))
        mapping = {"a": "phone_to_pc", "b": "pc_to_phone", "c": "pc_to_phone"}
        sampler = BalancedSampler(samples, mapping, seed=8)
        batches = sampler.draw_update(4, 4)
        self.assertEqual([len(batch) for batch in batches], [4]*4)
        self.assertEqual(sampler.last_counts["real"], 8)
        self.assertEqual(sampler.last_counts["synthetic"], 8)
        self.assertEqual(sampler.last_counts["groups"], {"phone_to_pc": 4, "pc_to_phone": 4})
        self.assertEqual(sampler.last_counts["repeatedBursts"], 0)
        self.assertEqual(len({row.metadata["burstId"] for row in flat(batches)}), 16)

    def test_draws_are_reproducible_and_independent_of_input_order(self):
        samples = examples("real", "a", payloads=3)+examples("real", "b", payloads=3)+examples("synthetic", "s", payloads=4)
        mapping = {"a": "one", "b": "two"}
        a = BalancedSampler(samples, mapping, seed=18)
        b = BalancedSampler(list(reversed(samples)), mapping, seed=18)
        for _ in range(10):
            self.assertEqual([r.id for r in flat(a.draw_update(4, 4))], [r.id for r in flat(b.draw_update(4, 4))])
            self.assertEqual(a.last_counts, b.last_counts)

    def test_hierarchy_does_not_weight_sessions_or_phases_by_frame_count(self):
        samples = (examples("real", "large", payloads=1, phases=1, frames=100)
                   + examples("real", "small", payloads=1, phases=4, frames=1))
        sampler = BalancedSampler(samples, {"large": "same", "small": "same"}, seed=24, real_fraction=1)
        sessions = {"large": 0, "small": 0}
        phases = dict.fromkeys(range(4), 0)
        for _ in range(2000):
            row = sampler.draw_update(1, 1)[0][0]
            sessions[row.metadata["sessionId"]] += 1
            if row.metadata["sessionId"] == "small":
                phases[row.metadata["info"]["phase"]] += 1
        self.assertTrue(850 < sessions["large"] < 1150, sessions)
        self.assertTrue(all(170 < count < 330 for count in phases.values()), phases)

    def test_small_pools_reuse_bursts_before_reusing_samples(self):
        sampler = BalancedSampler(examples("real", "one", phases=1, frames=3), {"one": "camera"}, seed=3)
        rows = flat(sampler.draw_update(2, 2))
        self.assertEqual(len({row.id for row in rows[:3]}), 3)
        self.assertEqual(sampler.last_counts["real"], 4)
        self.assertEqual(sampler.last_counts["repeatedBursts"], 3)
        self.assertEqual(sampler.last_counts["repeatedSamples"], 1)

    def test_reused_burst_draws_uniformly_among_remaining_captures(self):
        # Enumerate the random ranks instead of relying on statistical tolerance.
        # With A already used, the previous random-start/forward-scan algorithm
        # selected B for two of four starts, but C and D for only one each.
        rows = examples("real", "one", phases=1, frames=4)
        pool = _Pool(rows)
        ordered = pool.bursts[0].samples

        class RankedRandom:
            def __init__(self, rank):
                self.rank = rank
                self.bounds = []

            def choice(self, values):
                return values[0]

            def randrange(self, stop):
                self.bounds.append(stop)
                return self.rank

        for used_indices in ((0,), (1,), (0, 2), (0, 1, 2)):
            previous = {ordered[i].id for i in used_indices}
            remaining = [row.id for row in ordered if row.id not in previous]
            observed = []
            for rank in range(len(remaining)):
                rng = RankedRandom(rank)
                usage = ({0: set(previous)}, set(previous))
                sample, repeated_burst, repeated_sample = pool.draw(rng, usage)
                self.assertEqual(rng.bounds, [len(remaining)])
                self.assertTrue(repeated_burst)
                self.assertFalse(repeated_sample)
                observed.append(sample.id)
            self.assertEqual(observed, remaining)

    def test_profiles_stay_separate_and_missing_direction_profiles_fail(self):
        rows = examples("real", "a", shape=(64, 16))+examples("real", "b", shape=(128, 16))
        with self.assertRaisesRegex(ValueError, "missing groups"):
            BalancedSampler(rows, {"a": "left", "b": "right"})
        rows += examples("real", "a", shape=(128, 16))+examples("real", "b", shape=(64, 16))
        rows += examples("synthetic", "s", shape=(64, 4))
        sampler = BalancedSampler(rows, {"a": "left", "b": "right"}, seed=22)
        observed = set()
        for _ in range(20):
            batch = flat(sampler.draw_update(4, 2))
            shapes = {row.shape for row in batch}
            self.assertEqual(len(shapes), 1)
            observed.update(shapes)
        self.assertEqual(observed, {(64, 16), (128, 16), (64, 4)})

    def test_source_fraction_boundaries_and_argument_checks(self):
        rows = examples("real", "a")+examples("synthetic", "s")
        for fraction, source in ((0, "synthetic"), (1, "real")):
            sampler = BalancedSampler(rows, {"a": "one"}, real_fraction=fraction)
            self.assertTrue(all(row.metadata["source"] == source for row in flat(sampler.draw_update(2, 2))))
        for fraction in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                BalancedSampler(rows, {"a": "one"}, real_fraction=fraction)
        sampler = BalancedSampler(rows, {"a": "one"})
        for batch, accumulation in ((0, 1), (1, 0), (True, 1), (1, 129)):
            with self.assertRaises(ValueError):
                sampler.draw_update(batch, accumulation)

    def test_paired_updates_match_every_real_target_and_balance_directions(self):
        rows = paired_examples("a", payloads=2, frames=20)+paired_examples("b", payloads=2, frames=1)
        before = copy.deepcopy([row.metadata for row in rows])
        sampler = BalancedSampler(rows, {"a": "phone_to_pc", "b": "pc_to_phone"},
                                  paired_targets=True, seed=17)
        for _ in range(10):
            batches = sampler.draw_update(4, 4)
            self.assertEqual([len(batch) for batch in batches], [4]*4)
            chosen = flat(batches)
            self.assertEqual(target_counts(chosen, "real"), target_counts(chosen, "synthetic"))
            self.assertEqual(sampler.last_counts["real"], 8)
            self.assertEqual(sampler.last_counts["synthetic"], 8)
            self.assertEqual(sampler.last_counts["pairedTargets"], 8)
            self.assertEqual(sampler.last_counts["groups"], {"phone_to_pc": 4, "pc_to_phone": 4})
        self.assertEqual(before, [row.metadata for row in rows])

    def test_paired_draws_reproduce_and_keep_profiles_separate(self):
        rows = paired_examples("a", shape=(64, 16))+paired_examples("a", shape=(128, 4))
        a = BalancedSampler(rows, {"a": "camera"}, paired_targets=True, seed=28)
        b = BalancedSampler(list(reversed(rows)), {"a": "camera"}, paired_targets=True, seed=28)
        profiles = set()
        for _ in range(20):
            chosen = flat(a.draw_update(3, 2))
            self.assertEqual([r.id for r in chosen], [r.id for r in flat(b.draw_update(3, 2))])
            self.assertEqual(len({r.shape for r in chosen}), 1)
            self.assertEqual(target_counts(chosen, "real"), target_counts(chosen, "synthetic"))
            profiles.add(chosen[0].shape)
        self.assertEqual(profiles, {(64, 16), (128, 4)})

    def test_paired_corruption_variants_are_used_before_repetition(self):
        rows = paired_examples("a", phases=1, frames=3, variants=3)
        sampler = BalancedSampler(rows, {"a": "camera"}, paired_targets=True, seed=8)
        chosen = flat(sampler.draw_update(6, 1))
        self.assertEqual(len({r.id for r in chosen}), 6)
        self.assertEqual(sampler.last_counts["repeatedSamples"], 0)

    def test_pairing_rejects_incomplete_or_unrelated_targets(self):
        rows = paired_examples("a")
        real = [r for r in rows if r.metadata["source"] == "real"]
        synthetic = [r for r in rows if r.metadata["source"] == "synthetic"]
        for invalid in (real, synthetic, real+synthetic[:-3],
                        rows+[r for r in paired_examples("extra") if r.metadata["source"] == "synthetic"]):
            with self.assertRaisesRegex(ValueError, "counterparts|matching real"):
                BalancedSampler(invalid, {"a": "camera"}, paired_targets=True)
        changed = copy.deepcopy(rows)
        next(r for r in changed if r.metadata["source"] == "synthetic").metadata["info"]["phase"] = 15
        with self.assertRaisesRegex(ValueError, "matching real"):
            BalancedSampler(changed, {"a": "camera"}, paired_targets=True)

    def test_pairing_rejects_content_metadata_or_duplicate_identity_mismatch(self):
        rows = paired_examples("a")
        for field in ("labels", "mask", "expected"):
            changed = copy.deepcopy(rows)
            next(r for r in changed if r.metadata["source"] == "synthetic").metadata["hashes"][field] = "f"*64
            with self.assertRaisesRegex(ValueError, "hashes disagree"):
                BalancedSampler(changed, {"a": "camera"}, paired_targets=True)
        for field, value in (("senderSessionId", "different-sender"), ("split", "validation")):
            changed = copy.deepcopy(rows)
            next(r for r in changed if r.metadata["source"] == "synthetic").metadata[field] = value
            with self.assertRaisesRegex(ValueError, "metadata"):
                BalancedSampler(changed, {"a": "camera"}, paired_targets=True)
        with self.assertRaisesRegex(ValueError, "Duplicate sampler"):
            BalancedSampler(rows+[rows[0]], {"a": "camera"}, paired_targets=True)
        rows[-1].metadata["hashes"].pop("mask")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            BalancedSampler(rows, {"a": "camera"}, paired_targets=True)

    def test_pairing_requires_even_updates_and_equal_source_weight(self):
        rows = paired_examples("a")
        for fraction in (0, .25, .75, 1):
            with self.assertRaisesRegex(ValueError, "real-fraction .5"):
                BalancedSampler(rows, {"a": "camera"}, real_fraction=fraction, paired_targets=True)
        sampler = BalancedSampler(rows, {"a": "camera"}, paired_targets=True)
        for batch, accumulation in ((1, 1), (3, 1), (3, 3)):
            with self.assertRaisesRegex(ValueError, "even effective update"):
                sampler.draw_update(batch, accumulation)


if __name__ == "__main__":
    unittest.main()
