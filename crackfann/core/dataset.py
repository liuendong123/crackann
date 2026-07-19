from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackfann.core.distance import topk_from_ids
from crackfann.core.types import CandidateBatch, FilteredQuery, RangePredicate


@dataclass
class Dataset:
    vectors: np.ndarray
    attributes: np.ndarray
    ids: np.ndarray

    @classmethod
    def from_arrays(cls, vectors: np.ndarray, attributes: np.ndarray) -> "Dataset":
        vectors = np.asarray(vectors, dtype=np.float32)
        attributes = np.asarray(attributes, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if attributes.ndim == 1:
            attributes = attributes.reshape(-1, 1)
        if attributes.shape[0] != vectors.shape[0]:
            raise ValueError("attributes and vectors must have the same row count")
        return cls(vectors=vectors, attributes=attributes, ids=np.arange(vectors.shape[0], dtype=np.int64))

    @classmethod
    def synthetic(
        cls,
        n: int = 10000,
        d: int = 32,
        seed: int = 42,
        attr_mode: str = "correlated",
        noise: float = 0.05,
    ) -> "Dataset":
        rng = np.random.default_rng(seed)
        vectors = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
        if attr_mode == "uniform":
            attr = rng.random(n, dtype=np.float32)
        elif attr_mode == "clustered":
            centers = rng.choice(np.linspace(0.05, 0.95, 12), size=n)
            attr = centers + rng.normal(0.0, noise, size=n)
            attr = np.clip(attr, 0.0, 1.0).astype(np.float32)
        elif attr_mode == "correlated":
            raw = vectors[:, 0]
            scaled = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-12)
            attr = np.clip(scaled + rng.normal(0.0, noise, size=n), 0.0, 1.0).astype(np.float32)
        else:
            raise ValueError(f"Unknown attr_mode: {attr_mode}")
        return cls.from_arrays(vectors, attr)

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def d(self) -> int:
        return int(self.vectors.shape[1])

    def attr_values(self, attr_id: int = 0) -> np.ndarray:
        return self.attributes[:, attr_id]

    def mask_for_predicates(self, predicates: tuple[RangePredicate, ...]) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        for predicate in predicates:
            mask &= predicate.contains_values(self.attr_values(predicate.attr_id))
        return mask

    def ids_for_predicates(self, predicates: tuple[RangePredicate, ...]) -> np.ndarray:
        return self.ids[self.mask_for_predicates(predicates)]

    def count_for_predicates(self, predicates: tuple[RangePredicate, ...]) -> int:
        return int(self.mask_for_predicates(predicates).sum())

    def exact_search(self, query: FilteredQuery, source: str = "ground_truth") -> CandidateBatch:
        ids = self.ids_for_predicates(query.predicates)
        return topk_from_ids(self.vectors, ids, query.vector, query.k, source=source, predicate_checks=self.n)

