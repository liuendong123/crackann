from __future__ import annotations

import unittest

from crackfann.designer.promotion import PromotionPolicy
from crackfann.predicate.cell import MaterializationLevel, PredicateCell


class PromotionPolicyTest(unittest.TestCase):
    def test_accepts_positive_break_even(self) -> None:
        cell = PredicateCell(0, None, None, None, 0.0, 1.0, data_count=1000, level=MaterializationLevel.L1_BITMAP)
        cell.query_count_total = 100
        cell.current_latency_ema = 1000.0
        policy = PromotionPolicy(
            {
                "min_observations": 10,
                "min_cell_size": 10,
                "build_cost_per_work": 1.0,
                "future_query_multiplier": 2.0,
                "promotion_margin": 0.0,
            }
        )
        decision = policy.decide(cell, estimated_l3_query_cost=100.0)
        self.assertTrue(decision.accept)
        self.assertGreater(decision.predicted_gain, 0.0)

    def test_rejects_cold_cell(self) -> None:
        cell = PredicateCell(0, None, None, None, 0.0, 1.0, data_count=1000, level=MaterializationLevel.L1_BITMAP)
        policy = PromotionPolicy({"min_observations": 10, "min_cell_size": 10})
        decision = policy.decide(cell, estimated_l3_query_cost=100.0)
        self.assertFalse(decision.accept)
        self.assertEqual(decision.reason, "insufficient_observations")


if __name__ == "__main__":
    unittest.main()

