from __future__ import annotations

import unittest

import numpy as np

from crackfann.materialization.bitmap_store import BitmapStore


class BitmapStoreTest(unittest.TestCase):
    def test_build_and_union(self) -> None:
        store = BitmapStore()
        a = store.build(1, np.array([1, 2, 3]))
        b = store.build(2, np.array([3, 4]))
        self.assertEqual(store.cardinality(a), 3)
        self.assertEqual(store.union([a, b]).tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()

