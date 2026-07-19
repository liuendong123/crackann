from __future__ import annotations

import numpy as np

from crackfann.core.dataset import Dataset
from crackfann.core.distance import topk_from_ids
from crackfann.core.types import CandidateBatch, RangePredicate


class ExactBaseANN:
    """Exact base path used as correctness oracle and dependency-free fallback."""

    def __init__(self, dataset: Dataset | None = None) -> None:
        self.dataset = dataset
        self.vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray, params: dict | None = None) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def attach_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.build(dataset.vectors, {})

    def _require_vectors(self) -> np.ndarray:
        if self.vectors is None:
            raise RuntimeError("Base index has not been built")
        return self.vectors

    def search_unfiltered(self, q: np.ndarray, k: int, ef: int = 0) -> CandidateBatch:
        vectors = self._require_vectors()
        ids = np.arange(vectors.shape[0], dtype=np.int64)
        return topk_from_ids(vectors, ids, q, k, source="base_exact")

    def search_postfilter(
        self,
        q: np.ndarray,
        predicates: tuple[RangePredicate, ...],
        k: int,
        ef: int = 0,
    ) -> CandidateBatch:
        if self.dataset is None:
            raise RuntimeError("Postfilter search requires an attached dataset")
        ids = self.dataset.ids_for_predicates(predicates)
        return topk_from_ids(
            self._require_vectors(),
            ids,
            q,
            k,
            source="base_postfilter_exact",
            predicate_checks=self.dataset.n,
        )

    def search_with_allowed_ids(
        self,
        q: np.ndarray,
        allowed_ids: np.ndarray,
        k: int,
        budget: object | None = None,
    ) -> CandidateBatch:
        return topk_from_ids(self._require_vectors(), allowed_ids, q, k, source="base_allowed_exact")

