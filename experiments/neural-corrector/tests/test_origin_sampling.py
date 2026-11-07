"""Origin quotas apply to optimizer updates, not the number of stored captures."""

from collections import Counter
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling import BalancedSampler, REAL_ORIGINS
from test_sampling import paired_examples, flat, target_counts


def corpus(sizes=(16,), availability=None, *, frames=3):
    availability = availability or {"pc-to-phone": REAL_ORIGINS, "phone-to-pc": REAL_ORIGINS}
    rows, mapping = [], {}
    for size in sizes:
        for direction, origins in availability.items():
            for origin in origins:
                session = f"{size}/{direction}/{origin}"
                mapping[session] = direction
                examples = paired_examples(session, payloads=2, shape=(size, 16), frames=frames)
                for row in examples:
                    row.dataset_format = "neural-qr-dataset"
                    row.dataset_version = 2
                    row.metadata["origin"] = origin
                rows.extend(examples)
    return rows, mapping


def sampler(rows, groups, **options):
    return BalancedSampler(rows, groups, native_real_fraction=.75, paired_targets=True,
                           require_all_groups=False, **options)


class OriginSamplingTests(unittest.TestCase):
    def test_complete_coverage_exact_update_quotas_and_same_origin_pairing(self):
        rows, groups = corpus()
        before = copy.deepcopy([row.metadata for row in rows])
        draw = sampler(rows, groups, seed=45)
        for _ in range(15):
            batches = draw.draw_update(8, 4)
            chosen = flat(batches)
            self.assertEqual([len(batch) for batch in batches], [8]*4)
            self.assertEqual(draw.last_counts["real"], 16)
            self.assertEqual(draw.last_counts["synthetic"], 16)
            self.assertEqual(draw.last_counts["origins"], {"native": 12, "resampled": 2, "header-g64": 2})
            self.assertEqual(draw.last_counts["groups"], {"pc-to-phone": 8, "phone-to-pc": 8})
            self.assertEqual(target_counts(chosen, "real"), target_counts(chosen, "synthetic"))
            pairs = {}
            for sample, pair_id in zip(chosen, sum(draw.last_pair_ids, [])):
                pairs.setdefault(pair_id, []).append(sample)
            for pair in pairs.values():
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[0].metadata["origin"], pair[1].metadata["origin"])
            for count in draw.last_counts["originsByGroup"].values():
                self.assertEqual(count, {"native": 6, "resampled": 1, "header-g64": 1})
        self.assertEqual(before, [row.metadata for row in rows])

    def test_accumulation_input_order_and_seed_do_not_change_an_update(self):
        rows, groups = corpus((16, 32, 64, 128))
        a = sampler(rows, groups, seed=56)
        b = sampler(list(reversed(rows)), groups, seed=56)
        sizes = Counter()
        for _ in range(200):
            one, two = flat(a.draw_update(8, 4)), flat(b.draw_update(4, 8))
            self.assertEqual([s.id for s in one], [s.id for s in two])
            self.assertEqual(a.last_counts, b.last_counts)
            self.assertEqual(len({s.shape for s in one}), 1)
            sizes[one[0].shape[0]] += 1
        self.assertEqual(set(sizes), {16, 32, 64, 128})
        self.assertTrue(all(30 < n < 70 for n in sizes.values()), sizes)

    def test_missing_origins_redistribute_within_direction_without_faking_native(self):
        rows, groups = corpus(availability={"pc-to-phone": ("native",),
                                            "phone-to-pc": ("resampled", "header-g64")})
        draw = sampler(rows, groups, seed=4)
        draw.draw_update(8, 4)
        self.assertEqual(draw.last_counts["origins"], {"native": 8, "resampled": 4, "header-g64": 4})
        self.assertEqual(draw.last_counts["groups"], {"pc-to-phone": 8, "phone-to-pc": 8})
        policy = draw.origin_policy()
        self.assertEqual(policy["nativeRealFraction"], .75)
        self.assertEqual({row["group"]: row["expectedNativeRealFraction"] for row in policy["coverage"]},
                         {"pc-to-phone": 1., "phone-to-pc": 0.})
        # A grid with only one auxiliary origin receives all its real quota.
        rows, groups = corpus(availability={"pc-to-phone": ("header-g64",)})
        draw = sampler(rows, groups)
        draw.draw_update(8, 4)
        self.assertEqual(draw.last_counts["origins"], {"native": 0, "resampled": 0, "header-g64": 16})

    def test_tiny_updates_round_without_systematic_native_or_auxiliary_bias(self):
        rows, groups = corpus(availability={"camera": REAL_ORIGINS})
        draw = sampler(rows, groups, seed=13)
        totals = Counter()
        for _ in range(4000):
            draw.draw_update(1, 2)  # One real capture, paired with one synthetic.
            totals.update(draw.last_counts["origins"])
        self.assertTrue(2900 < totals["native"] < 3100, totals)
        self.assertTrue(400 < totals["resampled"] < 600, totals)
        self.assertTrue(400 < totals["header-g64"] < 600, totals)

    def test_origin_validation_pair_mismatch_and_fraction_limits(self):
        rows, groups = corpus()
        for value in (-.1, 1.1, float("nan"), float("inf"), True, "0.75"):
            with self.assertRaisesRegex(ValueError, "Native real fraction"):
                BalancedSampler(rows, groups, native_real_fraction=value)
        for origin in (None, "synthetic", "other"):
            changed = copy.deepcopy(rows)
            changed[0].metadata["origin"] = origin
            with self.assertRaisesRegex(ValueError, "requires native"):
                sampler(changed, groups)
        changed = copy.deepcopy(rows)
        next(row for row in changed if row.metadata["source"] == "synthetic").metadata["origin"] = "resampled"
        with self.assertRaisesRegex(ValueError, "origin"):
            sampler(changed, groups)
        for value, native in ((0, 0), (1, 16)):
            draw = BalancedSampler(rows, groups, native_real_fraction=value, paired_targets=True)
            draw.draw_update(8, 4)
            self.assertEqual(draw.last_counts["origins"]["native"], native)

    def test_disabled_policy_preserves_existing_sampler_results(self):
        rows, groups = corpus()
        a = BalancedSampler(rows, groups, paired_targets=True, seed=19)
        b = BalancedSampler(rows, groups, paired_targets=True, seed=19, native_real_fraction=None)
        self.assertIsNone(a.origin_policy())
        for _ in range(10):
            self.assertEqual([r.id for r in flat(a.draw_update(8, 4))],
                             [r.id for r in flat(b.draw_update(8, 4))])
            self.assertNotIn("origins", a.last_counts)


if __name__ == "__main__":
    unittest.main()
