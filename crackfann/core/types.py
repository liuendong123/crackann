from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RangePredicate:
    attr_id: int
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Invalid range predicate: [{self.low}, {self.high}]")

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def contains_values(self, values: np.ndarray) -> np.ndarray:
        return (values >= self.low) & (values <= self.high)

    def overlaps(self, low: float, high: float) -> bool:
        return self.low <= high and low <= self.high


@dataclass(frozen=True)
class FilteredQuery:
    query_id: int
    vector: np.ndarray
    predicates: tuple[RangePredicate, ...]
    k: int
    timestamp: int
    phase: str = "default"


@dataclass
class CandidateBatch:
    ids: np.ndarray
    distances: np.ndarray
    distance_computations: int
    predicate_checks: int = 0
    source: str = "unknown"


@dataclass
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    latency_ms: float
    distance_computations: int
    predicate_checks: int
    plan_id: str
    visited_cells: list[int]
    recall: float | None = None
    quality_risk: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildReport:
    handle: str
    build_ms: float
    memory_bytes: int
    build_work: float
    details: dict[str, Any] = field(default_factory=dict)

