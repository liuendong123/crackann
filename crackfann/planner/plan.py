from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PlanType(str, Enum):
    PREFILTER_EXACT = "PREFILTER_EXACT"
    BASE_POSTFILTER = "BASE_POSTFILTER"
    CELL_L3 = "CELL_L3"
    HYBRID_CELLS = "HYBRID_CELLS"
    BASE_FALLBACK = "BASE_FALLBACK"


@dataclass
class SearchBudget:
    max_candidates: int | None = None
    probe_buckets: int = 8
    ef: int = 0


@dataclass
class QueryPlan:
    plan_type: PlanType
    cell_ids: list[int]
    budget: SearchBudget = field(default_factory=SearchBudget)
    reason: str = ""

