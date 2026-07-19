from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from crackfann.base_index.exact_backend import ExactBaseANN
from crackfann.core.dataset import Dataset
from crackfann.core.types import FilteredQuery
from crackfann.designer.actions import ActionRecord
from crackfann.designer.monitor import ema
from crackfann.designer.promotion import PromotionPolicy
from crackfann.materialization.bitmap_store import BitmapStore
from crackfann.materialization.catalog import MaterializationCatalog
from crackfann.materialization.local_ann_store import create_local_ann_store
from crackfann.planner.router import QueryPlanner
from crackfann.planner.scheduler import CrossCellScheduler
from crackfann.predicate.cell import MaterializationLevel
from crackfann.predicate.tree import PredicateTree


class CrackFANNSystem:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.dataset: Dataset | None = None
        self.base_index = ExactBaseANN()
        self.tree: PredicateTree | None = None
        self.catalog: MaterializationCatalog | None = None
        self.bitmap_store = BitmapStore()
        ann_cfg = self.config.get("local_ann", {})
        self.local_ann_store = create_local_ann_store(ann_cfg, seed=int(self.config.get("seed", 42)))
        planner_cfg = dict(self.config.get("planner", {}))
        planner_cfg.setdefault("exact_scan_threshold", self.config.get("materialization", {}).get("exact_scan_threshold", 5000))
        self.planner = QueryPlanner(planner_cfg)
        self.promotion_policy = PromotionPolicy(self.config.get("promotion", {}))
        self.scheduler: CrossCellScheduler | None = None
        self.action_log: list[ActionRecord] = []
        self.last_plan_reason = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CrackFANNSystem":
        return cls(config=config)

    def load_dataset(self, dataset: Dataset | np.ndarray, attributes: np.ndarray | None = None) -> None:
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            if attributes is None:
                raise ValueError("attributes are required when loading raw vectors")
            self.dataset = Dataset.from_arrays(dataset, attributes)
        self.base_index.attach_dataset(self.dataset)

    def build_base_index(self) -> None:
        dataset = self._dataset()
        self.base_index.attach_dataset(dataset)

    def initialize_fixed_cells(self, num_cells: int = 16, attr_id: int = 0) -> None:
        dataset = self._dataset()
        self.tree = PredicateTree.from_quantiles(dataset.attr_values(attr_id), num_cells=num_cells, attr_id=attr_id)
        values = dataset.attr_values(attr_id)
        for cell in self.tree.cells.values():
            mask = self.tree.mask_for_cell(values, cell)
            ids = dataset.ids[mask]
            cell.data_count = int(ids.size)
            cell.bitmap_handle = self.bitmap_store.build(cell.cell_id, ids)
            cell.level = MaterializationLevel.L1_BITMAP
            cell.memory_bytes = self.bitmap_store.memory_bytes(cell.bitmap_handle)
        self.catalog = MaterializationCatalog(self.tree.cells)
        self.scheduler = CrossCellScheduler(
            dataset=dataset,
            base_index=self.base_index,
            tree=self.tree,
            cells=self.tree.cells,
            bitmap_store=self.bitmap_store,
            local_ann_store=self.local_ann_store,
        )

    def search(self, query: FilteredQuery):
        dataset = self._dataset()
        tree = self._tree()
        scheduler = self._scheduler()
        if not query.predicates:
            raise ValueError("CrackFANN v0 requires at least one range predicate")
        predicate = query.predicates[0]
        cover = tree.cover(predicate.low, predicate.high)
        candidate_count = dataset.count_for_predicates(query.predicates)
        plan = self.planner.choose(cover, candidate_count, dataset.n, tree.cells, query.k)
        self.last_plan_reason = plan.reason
        result = scheduler.execute(plan, query, cover)
        self._update_cell_stats(query.timestamp, cover, result.distance_computations, candidate_count)
        tick_interval = int(self.config.get("designer", {}).get("tick_interval", 100))
        if tick_interval > 0 and query.query_id > 0 and query.query_id % tick_interval == 0:
            self.designer_tick(query.timestamp)
        return result

    def designer_tick(self, timestamp: int) -> None:
        dataset = self._dataset()
        tree = self._tree()
        for cell in tree.cells.values():
            estimated_l3_cost = self.local_ann_store.estimate_query_cost(cell.data_count)
            decision = self.promotion_policy.decide(cell, estimated_l3_cost)
            if not decision.accept:
                continue
            ids = self.bitmap_store.get_ids(cell.bitmap_handle) if cell.bitmap_handle else np.empty(0, dtype=np.int64)
            if ids.size == 0:
                continue
            report = self.local_ann_store.build(cell.cell_id, ids, dataset.vectors[ids], self.config.get("local_ann", {}))
            previous_level = int(cell.level)
            cell.ann_handle = report.handle
            cell.level = MaterializationLevel.L3_LOCAL_ANN
            cell.generation += 1
            cell.build_cost_ms = report.build_ms
            cell.memory_bytes += report.memory_bytes
            self.action_log.append(
                ActionRecord(
                    ts=timestamp,
                    action="PROMOTE",
                    cell_id=cell.cell_id,
                    from_level=previous_level,
                    to_level=int(cell.level),
                    predicted_gain=decision.predicted_gain,
                    realized_gain=0.0,
                    build_ms=report.build_ms,
                    reason=decision.reason,
                )
            )

    def snapshot_cells(self, timestamp: int) -> list[dict[str, Any]]:
        tree = self._tree()
        rows = []
        for cell in tree.cells.values():
            rows.append(
                {
                    "ts": timestamp,
                    "cell_id": cell.cell_id,
                    "low": cell.low,
                    "high": cell.high,
                    "level": int(cell.level),
                    "data_count": cell.data_count,
                    "query_count_total": cell.query_count_total,
                    "query_count_ema": cell.query_count_ema,
                    "memory_bytes": cell.memory_bytes,
                    "state": cell.state,
                }
            )
        return rows

    def save_run(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "config_resolved.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, sort_keys=True)

    def _update_cell_stats(self, timestamp: int, cover, actual_cost: int, candidate_count: int) -> None:
        tree = self._tree()
        alpha = float(self.config.get("monitor", {}).get("alpha", 0.05))
        per_cell_cost = actual_cost / max(len(cover), 1)
        for part in cover:
            cell = tree.cells[part.cell_id]
            cell.query_count_total += 1
            cell.covered_query_count += 1
            cell.query_count_ema = ema(cell.query_count_ema, 1.0, alpha)
            cell.current_latency_ema = ema(cell.current_latency_ema, per_cell_cost, alpha)
            cell.base_latency_ema = ema(cell.base_latency_ema, max(candidate_count, 1), alpha)
            cell.last_access_ts = timestamp

    def _dataset(self) -> Dataset:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been loaded")
        return self.dataset

    def _tree(self) -> PredicateTree:
        if self.tree is None:
            raise RuntimeError("Predicate cells have not been initialized")
        return self.tree

    def _scheduler(self) -> CrossCellScheduler:
        if self.scheduler is None:
            raise RuntimeError("Scheduler has not been initialized")
        return self.scheduler
