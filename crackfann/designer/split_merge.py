from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackfann.predicate.cell import PredicateCell


@dataclass
class SplitDecision:
    accept: bool
    cut: float | None
    score: float
    reason: str


class SplitPolicy:
    """Conservative query-boundary split policy for v0.2."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.max_leaf_cells = int(config.get("max_leaf_cells", 64))
        self.min_observations = int(config.get("min_observations", 40))
        self.min_boundary_hits = int(config.get("min_boundary_hits", 20))
        self.min_cell_size = int(config.get("min_cell_size", 1000))
        self.min_child_size = int(config.get("min_child_size", 500))
        self.cooldown_queries = int(config.get("cooldown_queries", 200))
        self.max_splits_per_tick = int(config.get("max_splits_per_tick", 2))
        self.min_width_fraction = float(config.get("min_width_fraction", 0.05))

    def decide(
        self,
        cell: PredicateCell,
        boundary_samples: list[float],
        timestamp: int,
        last_split_ts: int | None,
        leaf_count: int,
    ) -> SplitDecision:
        if not self.enabled:
            return SplitDecision(False, None, 0.0, "split_disabled")
        if leaf_count >= self.max_leaf_cells:
            return SplitDecision(False, None, 0.0, "max_leaf_cells")
        if last_split_ts is not None and timestamp - last_split_ts < self.cooldown_queries:
            return SplitDecision(False, None, 0.0, "cooldown")
        if cell.query_count_total < self.min_observations:
            return SplitDecision(False, None, 0.0, "insufficient_observations")
        if cell.data_count < self.min_cell_size:
            return SplitDecision(False, None, 0.0, "cell_too_small")
        usable = [value for value in boundary_samples if cell.low < value < cell.high]
        if len(usable) < self.min_boundary_hits:
            return SplitDecision(False, None, 0.0, "insufficient_boundary_hits")

        cut = float(np.median(np.asarray(usable, dtype=np.float64)))
        min_gap = max((cell.high - cell.low) * self.min_width_fraction, 1e-9)
        if cut <= cell.low + min_gap or cut >= cell.high - min_gap:
            return SplitDecision(False, cut, float(len(usable)), "cut_too_close_to_edge")
        return SplitDecision(True, cut, float(len(usable)), "boundary_hotspot")

