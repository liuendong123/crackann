from __future__ import annotations

import unittest

import numpy as np

from crackfann.materialization.local_ann_store import (
    ExactLocalANNStore,
    RandomProjectionANNStore,
    available_backends,
    create_local_ann_store,
)


class LocalANNStoreTest(unittest.TestCase):
    def test_exact_backend_returns_nearest_neighbor(self) -> None:
        store = ExactLocalANNStore()
        vectors = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
        ids = np.array([10, 11, 12], dtype=np.int64)
        report = store.build(0, ids, vectors)
        result = store.search(report.handle, np.array([0.9, 0.9], dtype=np.float32), k=1)
        self.assertEqual(result.ids.tolist(), [11])
        self.assertEqual(result.source, "local_exact_l3")

    def test_factory_creates_rpann(self) -> None:
        store = create_local_ann_store({"backend": "rpann", "candidate_fraction": 1.0}, seed=3)
        self.assertIsInstance(store, RandomProjectionANNStore)

    def test_factory_creates_exact(self) -> None:
        store = create_local_ann_store({"backend": "exact"}, seed=3)
        self.assertIsInstance(store, ExactLocalANNStore)

    def test_backend_availability_reports_optional_modules(self) -> None:
        availability = available_backends()
        self.assertTrue(availability["exact"])
        self.assertTrue(availability["rpann"])
        self.assertIn("faiss", availability)
        self.assertIn("hnswlib", availability)


if __name__ == "__main__":
    unittest.main()

