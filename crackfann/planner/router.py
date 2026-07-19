from __future__ import annotations

from crackfann.planner.plan import PlanType, QueryPlan, SearchBudget
from crackfann.predicate.cell import MaterializationLevel, PredicateCell
from crackfann.predicate.tree import CoverPart


class QueryPlanner:
    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.exact_scan_threshold = int(config.get("exact_scan_threshold", 5000))
        self.broad_selectivity = float(config.get("broad_selectivity", 0.20))
        self.broad_cell_threshold = int(config.get("broad_cell_threshold", 12))
        self.l3_probe_buckets = int(config.get("l3_probe_buckets", 64))
        self.l3_candidate_fraction = float(config.get("l3_candidate_fraction", 0.75))

    def choose(
        self,
        cover: list[CoverPart],
        candidate_count: int,
        dataset_size: int,
        cells: dict[int, PredicateCell],
        k: int,
    ) -> QueryPlan:
        covered_ids = [part.cell_id for part in cover]
        selectivity = candidate_count / max(dataset_size, 1)
        full_l3 = [
            part.cell_id
            for part in cover
            if part.full and cells[part.cell_id].level >= MaterializationLevel.L3_LOCAL_ANN
        ]
        if full_l3 and candidate_count > self.exact_scan_threshold:
            max_candidates = max(
                4 * k,
                min(candidate_count, int(candidate_count * self.l3_candidate_fraction)),
            )
            return QueryPlan(
                plan_type=PlanType.HYBRID_CELLS,
                cell_ids=covered_ids,
                budget=SearchBudget(max_candidates=max_candidates, probe_buckets=self.l3_probe_buckets),
                reason="full_l3_available",
            )
        if candidate_count <= self.exact_scan_threshold:
            return QueryPlan(
                plan_type=PlanType.PREFILTER_EXACT,
                cell_ids=covered_ids,
                reason="small_candidate_set",
            )
        if selectivity >= self.broad_selectivity or len(cover) >= self.broad_cell_threshold:
            return QueryPlan(
                plan_type=PlanType.BASE_POSTFILTER,
                cell_ids=covered_ids,
                reason="broad_or_many_cells",
            )
        return QueryPlan(
            plan_type=PlanType.PREFILTER_EXACT,
            cell_ids=covered_ids,
            reason="medium_candidate_set",
        )
