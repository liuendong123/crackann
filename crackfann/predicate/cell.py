from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MaterializationLevel(IntEnum):
    L0_STATS = 0
    L1_BITMAP = 1
    L2_SUMMARY = 2
    L3_LOCAL_ANN = 3


@dataclass
class PredicateCell:
    cell_id: int
    parent_id: int | None
    left_child: int | None
    right_child: int | None
    low: float
    high: float
    data_count: int
    level: MaterializationLevel = MaterializationLevel.L0_STATS
    state: str = "ACTIVE"
    generation: int = 0
    query_count_total: int = 0
    query_count_ema: float = 0.0
    covered_query_count: int = 0
    last_access_ts: int = -1
    estimated_lifetime: float = 0.0
    base_latency_ema: float = 0.0
    current_latency_ema: float = 0.0
    build_cost_ms: float = 0.0
    memory_bytes: int = 0
    realized_saving_ms: float = 0.0
    probe_debt_ms: float = 0.0
    bitmap_handle: str | None = None
    summary_handle: str | None = None
    ann_handle: str | None = None

    @property
    def width(self) -> float:
        return self.high - self.low

