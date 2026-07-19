from __future__ import annotations

import unittest

from crackfann.core.types import RangePredicate
from crackfann.predicate.interval import Interval


class IntervalTest(unittest.TestCase):
    def test_range_predicate_contains_and_overlaps(self) -> None:
        predicate = RangePredicate(attr_id=0, low=0.2, high=0.4)
        self.assertTrue(predicate.contains(0.3))
        self.assertFalse(predicate.contains(0.5))
        self.assertTrue(predicate.overlaps(0.35, 0.8))
        self.assertFalse(predicate.overlaps(0.5, 0.8))

    def test_interval_width(self) -> None:
        interval = Interval(0.1, 0.9)
        self.assertAlmostEqual(interval.width, 0.8)
        self.assertTrue(interval.contains(Interval(0.2, 0.3)))


if __name__ == "__main__":
    unittest.main()

