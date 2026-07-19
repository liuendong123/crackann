from __future__ import annotations


def cumulative_distance_cost(query_rows: list[dict]) -> float:
    return float(sum(float(row.get("distance_count", 0.0)) for row in query_rows))


def cumulative_build_cost(action_rows: list[dict], build_weight: float = 1.0) -> float:
    return float(sum(float(row.get("build_ms", 0.0)) * build_weight for row in action_rows))


def total_cost(query_rows: list[dict], action_rows: list[dict], build_weight: float = 1.0) -> float:
    return cumulative_distance_cost(query_rows) + cumulative_build_cost(action_rows, build_weight)

