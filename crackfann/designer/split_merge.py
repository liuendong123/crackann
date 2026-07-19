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
    scan_saving: float = 0.0
    cover_penalty: float = 0.0
    cover_growth: float = 0.0


class SplitPolicy:
    """Conservative query-boundary split policy for v0.3."""

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
        self.min_net_gain = float(config.get("min_net_gain", 0.0))
        self.cover_penalty_weight = float(config.get("cover_penalty_weight", 500.0))
        self.max_cover_growth = float(config.get("max_cover_growth", 0.35))

    def decide(
        self,
        cell: PredicateCell,
        boundary_samples: list[float],
        timestamp: int,
        last_split_ts: int | None,
        leaf_count: int,
        query_ranges: list[tuple[float, float]] | None = None,
        left_count: int | None = None,
        right_count: int | None = None,
    ) -> SplitDecision:
        proposal = self.propose_cut(cell, boundary_samples, timestamp, last_split_ts, leaf_count)
        if not proposal.accept or proposal.cut is None:
            return proposal
        if query_ranges is None or left_count is None or right_count is None:
            return proposal
        return self.evaluate_candidate(cell, proposal, query_ranges, left_count, right_count)

    def propose_cut(
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

    def evaluate_candidate(
        self,
        cell: PredicateCell,
        proposal: SplitDecision,
        query_ranges: list[tuple[float, float]],
        left_count: int,
        right_count: int,
    ) -> SplitDecision:
        if proposal.cut is None:
            return proposal
        usable_ranges = [
            (max(low, cell.low), min(high, cell.high))
            for low, high in query_ranges
            if low <= cell.high and cell.low <= high
        ]
        if not usable_ranges:
            return SplitDecision(False, proposal.cut, 0.0, "no_range_samples")

        cut = proposal.cut
        scan_saving = 0.0
        cover_growth_total = 0.0
        for low, high in usable_ranges:
            before = float(cell.data_count)
            if high <= cut:
                after = float(left_count)
                after_cover = 1.0
            elif low >= cut:
                after = float(right_count)
                after_cover = 1.0
            else:
                after = float(left_count + right_count)
                after_cover = 2.0
            scan_saving += max(0.0, before - after)
            cover_growth_total += after_cover - 1.0

        scan_saving /= len(usable_ranges)
        cover_growth = cover_growth_total / len(usable_ranges)
        cover_penalty = cover_growth * self.cover_penalty_weight
        net_gain = scan_saving - cover_penalty
        if cover_growth > self.max_cover_growth:
            return SplitDecision(
                False,
                cut,
                net_gain,
                "cover_growth_too_high",
                scan_saving=scan_saving,
                cover_penalty=cover_penalty,
                cover_growth=cover_growth,
            )
        if net_gain < self.min_net_gain:
            return SplitDecision(
                False,
                cut,
                net_gain,
                "below_split_net_gain",
                scan_saving=scan_saving,
                cover_penalty=cover_penalty,
                cover_growth=cover_growth,
            )
        return SplitDecision(
            True,
            cut,
            net_gain,
            "boundary_hotspot_with_penalty",
            scan_saving=scan_saving,
            cover_penalty=cover_penalty,
            cover_growth=cover_growth,
        )
