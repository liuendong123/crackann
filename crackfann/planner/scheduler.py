from __future__ import annotations

import time

import numpy as np

from crackfann.base_index.exact_backend import ExactBaseANN
from crackfann.core.dataset import Dataset
from crackfann.core.distance import merge_topk, topk_from_ids
from crackfann.core.types import CandidateBatch, FilteredQuery, SearchResult
from crackfann.materialization.bitmap_store import BitmapStore
from crackfann.materialization.local_ann_store import LocalANNBackend
from crackfann.planner.plan import PlanType, QueryPlan
from crackfann.predicate.cell import MaterializationLevel, PredicateCell
from crackfann.predicate.tree import CoverPart, PredicateTree


class CrossCellScheduler:
    def __init__(
        self,
        dataset: Dataset,
        base_index: ExactBaseANN,
        tree: PredicateTree,
        cells: dict[int, PredicateCell],
        bitmap_store: BitmapStore,
        local_ann_store: LocalANNBackend,
    ) -> None:
        self.dataset = dataset
        self.base_index = base_index
        self.tree = tree
        self.cells = cells
        self.bitmap_store = bitmap_store
        self.local_ann_store = local_ann_store

    def execute(self, plan: QueryPlan, query: FilteredQuery, cover: list[CoverPart]) -> SearchResult:
        start = time.perf_counter()
        if plan.plan_type == PlanType.BASE_POSTFILTER:
            batch = self.base_index.search_postfilter(query.vector, query.predicates, query.k)
        elif plan.plan_type == PlanType.PREFILTER_EXACT:
            ids = self.dataset.ids_for_predicates(query.predicates)
            batch = topk_from_ids(
                self.dataset.vectors,
                ids,
                query.vector,
                query.k,
                source="prefilter_exact",
                predicate_checks=self.dataset.n,
            )
        elif plan.plan_type in {PlanType.CELL_L3, PlanType.HYBRID_CELLS}:
            batch = self._execute_hybrid(plan, query, cover)
        else:
            batch = self.base_index.search_postfilter(query.vector, query.predicates, query.k)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(
            ids=batch.ids,
            distances=batch.distances,
            latency_ms=elapsed_ms,
            distance_computations=batch.distance_computations,
            predicate_checks=batch.predicate_checks,
            plan_id=plan.plan_type.value,
            visited_cells=plan.cell_ids,
            quality_risk=batch.ids.size < query.k,
            extra={"source": batch.source, "reason": plan.reason},
        )

    def _execute_hybrid(self, plan: QueryPlan, query: FilteredQuery, cover: list[CoverPart]) -> CandidateBatch:
        batches: list[CandidateBatch] = []
        for part in cover:
            cell = self.cells[part.cell_id]
            if part.full and cell.level >= MaterializationLevel.L3_LOCAL_ANN and cell.ann_handle:
                batches.append(self.local_ann_store.search(cell.ann_handle, query.vector, query.k, plan.budget))
                continue
            ids = self._ids_for_cover_part(part, query)
            if ids.size:
                batches.append(
                    topk_from_ids(
                        self.dataset.vectors,
                        ids,
                        query.vector,
                        query.k,
                        source="hybrid_exact_residual",
                        predicate_checks=int(ids.size),
                    )
                )
        merged = merge_topk(batches, query.k)
        valid_mask = self.dataset.mask_for_predicates(query.predicates)
        if merged.ids.size:
            keep = valid_mask[merged.ids]
            merged.ids = merged.ids[keep]
            merged.distances = merged.distances[keep]
        return merged

    def _ids_for_cover_part(self, part: CoverPart, query: FilteredQuery) -> np.ndarray:
        cell = self.cells[part.cell_id]
        if cell.bitmap_handle:
            ids = self.bitmap_store.get_ids(cell.bitmap_handle)
        else:
            ids = self.dataset.ids_for_predicates(query.predicates)
        if part.full:
            return ids
        if ids.size == 0:
            return ids
        mask = np.ones(ids.size, dtype=bool)
        for predicate in query.predicates:
            values = self.dataset.attributes[ids, predicate.attr_id]
            mask &= predicate.contains_values(values)
        return ids[mask]
