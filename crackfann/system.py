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
from crackfann.designer.split_merge import SplitPolicy
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
        self.split_policy = SplitPolicy(self.config.get("split", {}))
        self.scheduler: CrossCellScheduler | None = None
        self.action_log: list[ActionRecord] = []
        self.boundary_observations: dict[int, list[float]] = {}
        self.range_observations: dict[int, list[tuple[float, float]]] = {}
        self.last_split_ts: dict[int, int] = {}
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
            self._materialize_l1_cell(cell.cell_id, values)
        self.catalog = MaterializationCatalog(self.tree.cells)
        self._install_scheduler()

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
        self._record_boundary_observations(query, cover)
        self._update_cell_stats(query.timestamp, cover, result.distance_computations, candidate_count)
        tick_interval = int(self.config.get("designer", {}).get("tick_interval", 100))
        if tick_interval > 0 and query.query_id > 0 and query.query_id % tick_interval == 0:
            self.designer_tick(query.timestamp)
        return result

    def designer_tick(self, timestamp: int) -> None:
        dataset = self._dataset()
        tree = self._tree()
        self._run_split_tick(timestamp)
        for cell in list(tree.cells.values()):
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

    def _run_split_tick(self, timestamp: int) -> None:
        if not self.split_policy.enabled:
            return
        tree = self._tree()
        values = self._dataset().attr_values(tree.attr_id)
        split_count = 0
        candidates = sorted(
            list(tree.cells.values()),
            key=lambda cell: len(self.boundary_observations.get(cell.cell_id, [])),
            reverse=True,
        )
        for cell in candidates:
            if cell.level >= MaterializationLevel.L3_LOCAL_ANN:
                continue
            if split_count >= self.split_policy.max_splits_per_tick:
                break
            proposal = self.split_policy.propose_cut(
                cell=cell,
                boundary_samples=self.boundary_observations.get(cell.cell_id, []),
                timestamp=timestamp,
                last_split_ts=self.last_split_ts.get(cell.cell_id),
                leaf_count=len(tree.leaf_ids),
            )
            if not proposal.accept or proposal.cut is None:
                continue
            left_count, right_count = self._estimate_split_counts(cell, proposal.cut, values)
            if left_count < self.split_policy.min_child_size or right_count < self.split_policy.min_child_size:
                continue
            decision = self.split_policy.evaluate_candidate(
                cell=cell,
                proposal=proposal,
                query_ranges=self.range_observations.get(cell.cell_id, []),
                left_count=left_count,
                right_count=right_count,
            )
            if not decision.accept or decision.cut is None:
                continue
            left, right = tree.split_leaf(cell.cell_id, decision.cut)
            self._materialize_l1_cell(left.cell_id, values)
            self._materialize_l1_cell(right.cell_id, values)
            self.boundary_observations[left.cell_id] = [
                value for value in self.boundary_observations.get(cell.cell_id, []) if left.low < value < left.high
            ]
            self.boundary_observations[right.cell_id] = [
                value for value in self.boundary_observations.get(cell.cell_id, []) if right.low < value < right.high
            ]
            parent_ranges = self.range_observations.get(cell.cell_id, [])
            self.range_observations[left.cell_id] = [
                (max(low, left.low), min(high, left.high))
                for low, high in parent_ranges
                if low <= left.high and left.low <= high
            ]
            self.range_observations[right.cell_id] = [
                (max(low, right.low), min(high, right.high))
                for low, high in parent_ranges
                if low <= right.high and right.low <= high
            ]
            self.boundary_observations.pop(cell.cell_id, None)
            self.range_observations.pop(cell.cell_id, None)
            self.last_split_ts[left.cell_id] = timestamp
            self.last_split_ts[right.cell_id] = timestamp
            self.action_log.append(
                ActionRecord(
                    ts=timestamp,
                    action="SPLIT",
                    cell_id=cell.cell_id,
                    from_level=int(cell.level),
                    to_level=int(MaterializationLevel.L1_BITMAP),
                    predicted_gain=decision.score,
                    realized_gain=0.0,
                    build_ms=0.0,
                    reason=(
                        f"{decision.reason}:cut={decision.cut:.6g}->children={left.cell_id},{right.cell_id}"
                        f";scan_saving={decision.scan_saving:.3f}"
                        f";cover_penalty={decision.cover_penalty:.3f}"
                        f";cover_growth={decision.cover_growth:.3f}"
                    ),
                )
            )
            split_count += 1
        if split_count:
            self._install_scheduler()

    def _estimate_split_counts(self, cell, cut: float, values: np.ndarray) -> tuple[int, int]:
        left = (values >= cell.low) & (values < cut)
        right = (values >= cut) & (values <= cell.high)
        return int(left.sum()), int(right.sum())

    def _record_boundary_observations(self, query: FilteredQuery, cover) -> None:
        if not self.split_policy.enabled or not query.predicates:
            return
        tree = self._tree()
        predicate = query.predicates[0]
        for part in cover:
            cell = tree.cells[part.cell_id]
            samples = self.boundary_observations.setdefault(cell.cell_id, [])
            if cell.low < predicate.low < cell.high:
                samples.append(float(predicate.low))
            if cell.low < predicate.high < cell.high:
                samples.append(float(predicate.high))
            if len(samples) > 512:
                del samples[: len(samples) - 512]
            ranges = self.range_observations.setdefault(cell.cell_id, [])
            ranges.append((float(part.low), float(part.high)))
            if len(ranges) > 512:
                del ranges[: len(ranges) - 512]

    def _materialize_l1_cell(self, cell_id: int, values: np.ndarray | None = None) -> None:
        dataset = self._dataset()
        tree = self._tree()
        values = dataset.attr_values(tree.attr_id) if values is None else values
        cell = tree.cells[cell_id]
        mask = tree.mask_for_cell(values, cell)
        ids = dataset.ids[mask]
        cell.data_count = int(ids.size)
        cell.bitmap_handle = self.bitmap_store.build(cell.cell_id, ids)
        cell.summary_handle = None
        cell.ann_handle = None
        cell.level = MaterializationLevel.L1_BITMAP
        cell.memory_bytes = self.bitmap_store.memory_bytes(cell.bitmap_handle)
        cell.state = "ACTIVE"

    def _install_scheduler(self) -> None:
        dataset = self._dataset()
        tree = self._tree()
        self.scheduler = CrossCellScheduler(
            dataset=dataset,
            base_index=self.base_index,
            tree=tree,
            cells=tree.cells,
            bitmap_store=self.bitmap_store,
            local_ann_store=self.local_ann_store,
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
                    "width": cell.width,
                    "parent_id": cell.parent_id if cell.parent_id is not None else "",
                    "query_count_total": cell.query_count_total,
                    "query_count_ema": cell.query_count_ema,
                    "boundary_samples": len(self.boundary_observations.get(cell.cell_id, [])),
                    "range_samples": len(self.range_observations.get(cell.cell_id, [])),
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
