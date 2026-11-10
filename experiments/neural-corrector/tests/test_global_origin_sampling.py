"""Global origin proportions remain global when an origin only supports G64."""

from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling import BalancedSampler, DEFAULT_ORIGIN_RATIOS, parse_origin_ratios
from test_origin_sampling import corpus
from test_sampling import flat, target_counts


def mixed_corpus():
    rows, groups = [], {}
    for size in (16, 32, 64, 128):
        origins = ("native", "resampled", "header-g64") if size == 64 else ("native", "resampled")
        examples, mapping = corpus((size,), availability={
            "pc-to-phone": origins, "phone-to-pc": origins})
        rows.extend(examples)
        groups.update(mapping)
    return rows, groups


class GlobalOriginSamplingTests(unittest.TestCase):
    def test_named_ratios_validation(self):
        self.assertEqual(parse_origin_ratios("header-g64=.15, native=.5,resampled=.35"),
                         DEFAULT_ORIGIN_RATIOS)
        for value in ("native=.5,resampled=.5", "native=.5,native=.35,header-g64=.15",
                      "native=x,resampled=.5,header-g64=0", {"native": 1},
                      dict(native=True, resampled=0, **{"header-g64": 0}),
                      dict(native=float("nan"), resampled=.5, **{"header-g64": .5}),
                      dict(native=.5, resampled=.5, **{"header-g64": .5}),
                      dict(native=1.1, resampled=-.1, **{"header-g64": 0})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_origin_ratios(value)

    def test_global_shares_and_induced_grid_mix_with_paired_updates(self):
        rows, groups = mixed_corpus()
        draw = BalancedSampler(rows, groups, seed=19, paired_targets=True,
                               origin_ratios=DEFAULT_ORIGIN_RATIOS)
        origins, grids = Counter(), Counter()
        for _ in range(3000):
            chosen = flat(draw.draw_update(8, 4))
            self.assertEqual(len({s.shape for s in chosen}), 1)
            self.assertEqual(len({s.metadata["origin"] for s in chosen}), 1)
            self.assertEqual(draw.last_counts["real"], 16)
            self.assertEqual(draw.last_counts["synthetic"], 16)
            self.assertEqual(draw.last_counts["groups"], {"pc-to-phone": 8, "phone-to-pc": 8})
            self.assertEqual(target_counts(chosen, "real"), target_counts(chosen, "synthetic"))
            origins.update(draw.last_counts["origins"])
            grids[chosen[0].shape[0]] += 1
            if chosen[0].metadata["origin"] == "header-g64":
                self.assertEqual(chosen[0].shape[0], 64)
        for origin, expected in DEFAULT_ORIGIN_RATIOS.items():
            self.assertAlmostEqual(origins[origin]/sum(origins.values()), expected, delta=.025)
        for size, expected in ((16, .2125), (32, .2125), (64, .3625), (128, .2125)):
            self.assertAlmostEqual(grids[size]/sum(grids.values()), expected, delta=.025)
        policy = draw.origin_policy()
        self.assertEqual(policy["allocation"], "global-real-origin/eligible-grid/available-direction")
        expected = {row["size"]: row["share"] for row in policy["expectedGridShares"]}
        self.assertAlmostEqual(expected[64], .3625)
        self.assertAlmostEqual(policy["expectedDirectionShares"]["phone-to-pc"], .5)

    def test_seed_order_and_accumulation_do_not_change_draws(self):
        rows, groups = mixed_corpus()
        a = BalancedSampler(rows, groups, seed=123, paired_targets=True,
                            origin_ratios=DEFAULT_ORIGIN_RATIOS)
        b = BalancedSampler(list(reversed(rows)), groups, seed=123, paired_targets=True,
                            origin_ratios=dict(reversed(list(DEFAULT_ORIGIN_RATIOS.items()))))
        for _ in range(50):
            one, two = flat(a.draw_update(8, 4)), flat(b.draw_update(4, 8))
            self.assertEqual([s.id for s in one], [s.id for s in two])
            self.assertEqual(a.last_counts, b.last_counts)

    def test_unavailable_origin_rejected_zero_share_allowed_and_alternative_exclusive(self):
        rows, groups = corpus((16,), availability={"camera": ("native",)})
        with self.assertRaisesRegex(ValueError, "resampled has no real"):
            BalancedSampler(rows, groups, origin_ratios=DEFAULT_ORIGIN_RATIOS)
        ratios = {"native": 1, "resampled": 0, "header-g64": 0}
        draw = BalancedSampler(rows, groups, paired_targets=True, origin_ratios=ratios)
        draw.draw_update(4, 2)
        self.assertEqual(draw.last_counts["origins"], {"native": 4, "resampled": 0, "header-g64": 0})
        with self.assertRaisesRegex(ValueError, "not both"):
            BalancedSampler(rows, groups, native_real_fraction=.5, origin_ratios=ratios)
        with self.assertRaisesRegex(ValueError, "positive real fraction"):
            BalancedSampler(rows, groups, real_fraction=0, origin_ratios=ratios)

    def test_missing_direction_disclosed_and_never_substitutes_another_origin(self):
        rows, groups = corpus((64,), availability={"pc-to-phone": ("native", "resampled"),
                                                  "phone-to-pc": ("header-g64",)})
        draw = BalancedSampler(rows, groups, seed=67, paired_targets=True,
                               origin_ratios=DEFAULT_ORIGIN_RATIOS)
        policy = draw.origin_policy()
        self.assertAlmostEqual(policy["expectedDirectionShares"]["pc-to-phone"], .85)
        self.assertAlmostEqual(policy["expectedDirectionShares"]["phone-to-pc"], .15)
        for _ in range(30):
            chosen = flat(draw.draw_update(4, 2))
            origin = chosen[0].metadata["origin"]
            group = "phone-to-pc" if origin == "header-g64" else "pc-to-phone"
            self.assertEqual(draw.last_counts["groups"], {group: 4})

    def test_unpaired_sources_follow_chosen_origin_and_incomplete_synthetics_fail(self):
        rows, groups = mixed_corpus()
        draw = BalancedSampler(rows, groups, seed=3, origin_ratios=DEFAULT_ORIGIN_RATIOS)
        for _ in range(30):
            chosen = flat(draw.draw_update(4, 2))
            self.assertEqual(draw.last_counts["real"], 4)
            self.assertEqual(draw.last_counts["synthetic"], 4)
            self.assertEqual(len({s.metadata["origin"] for s in chosen}), 1)
        incomplete = [s for s in rows if not (s.metadata["origin"] == "resampled"
                                               and s.metadata["source"] == "synthetic")]
        with self.assertRaisesRegex(ValueError, "no synthetic captures"):
            BalancedSampler(incomplete, groups, origin_ratios=DEFAULT_ORIGIN_RATIOS)
        real_only = [s for s in rows if s.metadata["source"] == "real"]
        real_draw = BalancedSampler(real_only, groups, origin_ratios=DEFAULT_ORIGIN_RATIOS)
        real_draw.draw_update(4, 2)
        self.assertEqual(real_draw.last_counts["real"], 8)


if __name__ == "__main__":
    unittest.main()
