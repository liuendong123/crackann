from __future__ import annotations

from dataclasses import dataclass

from crackfann.predicate.cell import MaterializationLevel, PredicateCell


@dataclass
class PromotionDecision:
    accept: bool
    predicted_gain: float
    saving_per_query: float
    break_even_queries: float
    reason: str


class PromotionPolicy:
    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.min_observations = int(config.get("min_observations", 20))
        self.min_cell_size = int(config.get("min_cell_size", 1000))
        self.build_cost_per_work = float(config.get("build_cost_per_work", 1.0))
        self.future_query_multiplier = float(config.get("future_query_multiplier", 1.0))
        self.margin = float(config.get("promotion_margin", 0.0))

    def decide(self, cell: PredicateCell, estimated_l3_query_cost: float) -> PromotionDecision:
        if cell.level >= MaterializationLevel.L3_LOCAL_ANN:
            return PromotionDecision(False, 0.0, 0.0, float("inf"), "already_l3")
        if cell.data_count < self.min_cell_size:
            return PromotionDecision(False, 0.0, 0.0, float("inf"), "cell_too_small")
        if cell.query_count_total < self.min_observations:
            return PromotionDecision(False, 0.0, 0.0, float("inf"), "insufficient_observations")
        exact_cost = max(cell.current_latency_ema, float(cell.data_count))
        saving = max(0.0, exact_cost - estimated_l3_query_cost)
        if saving <= 0.0:
            return PromotionDecision(False, 0.0, saving, float("inf"), "no_query_saving")
        build_cost = cell.data_count * self.build_cost_per_work
        break_even = build_cost / max(saving, 1e-9)
        future_queries = cell.query_count_total * self.future_query_multiplier
        predicted_gain = saving * future_queries - build_cost
        if predicted_gain <= self.margin:
            return PromotionDecision(False, predicted_gain, saving, break_even, "below_margin")
        return PromotionDecision(True, predicted_gain, saving, break_even, "break_even_positive")

