from __future__ import annotations

from typing import Protocol

import numpy as np

from crackfann.core.types import CandidateBatch, RangePredicate


class BaseANN(Protocol):
    def build(self, vectors: np.ndarray, params: dict) -> None:
        ...

    def search_unfiltered(self, q: np.ndarray, k: int, ef: int) -> CandidateBatch:
        ...

    def search_postfilter(self, q: np.ndarray, predicates: tuple[RangePredicate, ...], k: int, ef: int) -> CandidateBatch:
        ...

    def search_with_allowed_ids(self, q: np.ndarray, allowed_ids: np.ndarray, k: int, budget: object | None) -> CandidateBatch:
        ...

