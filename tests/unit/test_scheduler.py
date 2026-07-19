from __future__ import annotations

import unittest

from crackfann.core.dataset import Dataset
from crackfann.core.types import FilteredQuery, RangePredicate
from crackfann.metrics.recall import recall_at_k
from crackfann.system import CrackFANNSystem


class SchedulerTest(unittest.TestCase):
    def test_prefilter_exact_reaches_ground_truth(self) -> None:
        dataset = Dataset.synthetic(n=1000, d=16, seed=7)
        system = CrackFANNSystem({"materialization": {"exact_scan_threshold": 1000}})
        system.load_dataset(dataset)
        system.build_base_index()
        system.initialize_fixed_cells(num_cells=8)
        query = FilteredQuery(
            query_id=1,
            vector=dataset.vectors[0],
            predicates=(RangePredicate(0, 0.2, 0.5),),
            k=10,
            timestamp=1,
            phase="test",
        )
        result = system.search(query)
        truth = dataset.exact_search(query)
        self.assertEqual(result.plan_id, "PREFILTER_EXACT")
        self.assertEqual(recall_at_k(truth.ids, result.ids, query.k), 1.0)


if __name__ == "__main__":
    unittest.main()

