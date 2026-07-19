from __future__ import annotations

import unittest

import numpy as np

from crackfann.core.dataset import Dataset
from crackfann.core.types import FilteredQuery, RangePredicate
from crackfann.system import CrackFANNSystem


class SplitPolicyIntegrationTest(unittest.TestCase):
    def test_repeated_boundaries_trigger_split(self) -> None:
        dataset = Dataset.synthetic(n=1200, d=8, seed=9, attr_mode="uniform")
        cfg = {
            "seed": 9,
            "local_ann": {"backend": "exact"},
            "materialization": {"exact_scan_threshold": 2000},
            "designer": {"tick_interval": 10},
            "promotion": {"min_observations": 9999},
            "split": {
                "enabled": True,
                "max_leaf_cells": 8,
                "min_observations": 5,
                "min_boundary_hits": 5,
                "min_cell_size": 100,
                "min_child_size": 20,
                "cooldown_queries": 0,
                "max_splits_per_tick": 1,
            },
        }
        system = CrackFANNSystem(cfg)
        system.load_dataset(dataset)
        system.build_base_index()
        system.initialize_fixed_cells(num_cells=2)

        vector = dataset.vectors[0]
        for query_id in range(11):
            query = FilteredQuery(
                query_id=query_id,
                vector=vector,
                predicates=(RangePredicate(0, 0.10, 0.25),),
                k=5,
                timestamp=query_id,
                phase="split_test",
            )
            system.search(query)

        self.assertGreater(len(system._tree().leaf_ids), 2)
        self.assertTrue(any(record.action == "SPLIT" for record in system.action_log))


if __name__ == "__main__":
    unittest.main()

