from __future__ import annotations

import unittest

import numpy as np

from crackfann.predicate.tree import PredicateTree


class TreeCoverTest(unittest.TestCase):
    def test_quantile_tree_cover(self) -> None:
        values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        tree = PredicateTree.from_quantiles(values, num_cells=4)
        parts = tree.cover(0.2, 0.8)
        self.assertGreaterEqual(len(parts), 2)
        self.assertLessEqual(parts[0].low, 0.2 + 1e-6)
        self.assertGreaterEqual(parts[-1].high, 0.8 - 1e-6)
        tree.validate()

    def test_cell_masks_cover_all_values_once(self) -> None:
        values = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        tree = PredicateTree.from_quantiles(values, num_cells=5)
        counts = np.zeros(values.size, dtype=np.int64)
        for cell in tree.cells.values():
            counts += tree.mask_for_cell(values, cell).astype(np.int64)
        self.assertTrue(np.all(counts == 1))

    def test_split_leaf_preserves_ordered_cover(self) -> None:
        values = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        tree = PredicateTree.from_quantiles(values, num_cells=2)
        original = tree.leaf_ids[0]
        parent = tree.cells[original]
        cut = (parent.low + parent.high) / 2.0
        left, right = tree.split_leaf(original, cut)
        self.assertNotIn(original, tree.cells)
        self.assertEqual(left.high, right.low)
        self.assertEqual(tree.leaf_ids[:2], [left.cell_id, right.cell_id])
        tree.validate()


if __name__ == "__main__":
    unittest.main()
