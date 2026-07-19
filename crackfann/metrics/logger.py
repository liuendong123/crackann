from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from crackfann.core.types import FilteredQuery, SearchResult


class RunLogger:
    def __init__(self) -> None:
        self.query_rows: list[dict[str, Any]] = []
        self.action_rows: list[dict[str, Any]] = []
        self.cell_rows: list[dict[str, Any]] = []

    def record_query(self, query: FilteredQuery, result: SearchResult, candidate_count: int, recall: float) -> None:
        predicate = query.predicates[0]
        self.query_rows.append(
            {
                "query_id": query.query_id,
                "timestamp": query.timestamp,
                "phase": query.phase,
                "low": predicate.low,
                "high": predicate.high,
                "selectivity": candidate_count,
                "plan_type": result.plan_id,
                "cell_ids": ";".join(str(cell_id) for cell_id in result.visited_cells),
                "latency_ms": result.latency_ms,
                "distance_count": result.distance_computations,
                "predicate_checks": result.predicate_checks,
                "recall": recall,
                "quality_risk": int(result.quality_risk),
                "fallback_reason": result.extra.get("reason", ""),
                "source": result.extra.get("source", ""),
                "covered_cell_count": result.extra.get("covered_cell_count", 0),
                "full_cell_count": result.extra.get("full_cell_count", 0),
                "partial_cell_count": result.extra.get("partial_cell_count", 0),
                "l3_cover_cell_count": result.extra.get("l3_cover_cell_count", 0),
                "scheduler_steps": result.extra.get("scheduler_steps", 0),
                "l3_cells": result.extra.get("l3_cells", 0),
                "exact_residual_cells": result.extra.get("exact_residual_cells", 0),
                "l3_distance_count": result.extra.get("l3_distance_count", 0),
                "exact_residual_distance_count": result.extra.get("exact_residual_distance_count", 0),
            }
        )

    def extend_actions(self, rows: list[dict[str, Any]]) -> None:
        self.action_rows.extend(rows)

    def extend_cells(self, rows: list[dict[str, Any]]) -> None:
        self.cell_rows.extend(rows)

    def write(self, output_dir: str | Path, manifest: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._write_csv(output / "query_log.csv", self.query_rows)
        self._write_csv(output / "action_log.csv", self.action_rows)
        self._write_csv(output / "cell_snapshots.csv", self.cell_rows)
        self._write_csv(output / "summary_by_phase.csv", summarize_by_phase(self.query_rows))
        with (output / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest or {}, f, indent=2, sort_keys=True)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def summarize_by_phase(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    phases = sorted({row["phase"] for row in rows})
    summary = []
    for phase in phases:
        phase_rows = [row for row in rows if row["phase"] == phase]
        recalls = np.array([float(row["recall"]) for row in phase_rows], dtype=np.float64)
        latencies = np.array([float(row["latency_ms"]) for row in phase_rows], dtype=np.float64)
        distances = np.array([float(row["distance_count"]) for row in phase_rows], dtype=np.float64)
        covered_cells = np.array([float(row.get("covered_cell_count", 0.0)) for row in phase_rows], dtype=np.float64)
        scheduler_steps = np.array([float(row.get("scheduler_steps", 0.0)) for row in phase_rows], dtype=np.float64)
        exact_residual = np.array(
            [float(row.get("exact_residual_distance_count", 0.0)) for row in phase_rows],
            dtype=np.float64,
        )
        summary.append(
            {
                "phase": phase,
                "queries": len(phase_rows),
                "recall_mean": float(recalls.mean()),
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "distance_count_mean": float(distances.mean()),
                "cumulative_distance_count": float(distances.sum()),
                "covered_cell_count_mean": float(covered_cells.mean()),
                "scheduler_steps_mean": float(scheduler_steps.mean()),
                "exact_residual_distance_mean": float(exact_residual.mean()),
                "quality_risk_count": int(sum(int(row["quality_risk"]) for row in phase_rows)),
            }
        )
    return summary
