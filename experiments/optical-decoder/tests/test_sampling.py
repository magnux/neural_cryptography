"""Direction/source quotas and burst diversity for offline adaptation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
from sampling import (BalancedSampler, GROUP_FORMAT, MAX_GROUP_BYTES,
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


if __name__ == "__main__":
    unittest.main()
