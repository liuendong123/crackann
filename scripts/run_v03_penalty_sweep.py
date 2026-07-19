from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crackfann.cli.run_workload import run_config


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path, params: dict, baseline: dict | None = None) -> dict:
    summary_rows = parse_csv(run_dir / "summary_by_phase.csv")
    action_rows = parse_csv(run_dir / "action_log.csv")
    cell_rows = parse_csv(run_dir / "cell_snapshots.csv")
    if not summary_rows:
        raise RuntimeError(f"Missing summary_by_phase.csv in {run_dir}")
    summary = summary_rows[0]
    action_counts: dict[str, int] = {}
    for row in action_rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1

    result = {
        **params,
        "run_dir": str(run_dir),
        "queries": int(summary["queries"]),
        "recall_mean": float(summary["recall_mean"]),
        "p95_latency_ms": float(summary["p95_latency_ms"]),
        "distance_count_mean": float(summary["distance_count_mean"]),
        "cumulative_distance_count": float(summary["cumulative_distance_count"]),
        "covered_cell_count_mean": float(summary.get("covered_cell_count_mean", 0.0)),
        "scheduler_steps_mean": float(summary.get("scheduler_steps_mean", 0.0)),
        "exact_residual_distance_mean": float(summary.get("exact_residual_distance_mean", 0.0)),
        "quality_risk_count": int(summary["quality_risk_count"]),
        "actions": len(action_rows),
        "splits": action_counts.get("SPLIT", 0),
        "promotes": action_counts.get("PROMOTE", 0),
        "final_leaf_cells": len(cell_rows),
    }
    if baseline is not None:
        base_distance = float(baseline["cumulative_distance_count"])
        base_p95 = float(baseline["p95_latency_ms"])
        result["distance_reduction_vs_baseline"] = (
            (base_distance - result["cumulative_distance_count"]) / base_distance if base_distance else 0.0
        )
        result["p95_delta_vs_baseline"] = result["p95_latency_ms"] - base_p95
    return result


def run_one(cfg: dict, run_id: str, out_root: Path, dry_run: bool) -> Path:
    run_dir = out_root / run_id
    if dry_run:
        print(f"DRY RUN {run_id}")
        return run_dir
    run_config(cfg, run_id=run_id, output_dir=run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/paper/synthetic_adaptive_hnswlib.json")
    parser.add_argument("--fixed_config", default="configs/paper/synthetic_fixed4_template_hnswlib.json")
    parser.add_argument("--out_root", default="outputs/v03_penalty_sweep_hnswlib")
    parser.add_argument("--run_prefix", default="penalty")
    parser.add_argument("--min_net_gain", default="25,50,100")
    parser.add_argument("--cover_penalty_weight", default="200,400,800")
    parser.add_argument("--max_cover_growth", default="0.35,0.5,1.0")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--recall_target", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(Path(args.base_config))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    baseline_summary: dict | None = None
    if args.fixed_config:
        fixed_cfg = load_config(Path(args.fixed_config))
        if args.backend:
            fixed_cfg.setdefault("local_ann", {})["backend"] = args.backend
        fixed_run_dir = run_one(fixed_cfg, "fixed_baseline", out_root, args.dry_run)
        if not args.dry_run:
            baseline_summary = summarize_run(fixed_run_dir, {"kind": "fixed_baseline"})
            rows.append(baseline_summary)

    values = product(
        parse_float_list(args.min_net_gain),
        parse_float_list(args.cover_penalty_weight),
        parse_float_list(args.max_cover_growth),
    )
    for idx, (min_gain, penalty_weight, max_growth) in enumerate(values):
        if args.limit and idx >= args.limit:
            break
        cfg = copy.deepcopy(base_cfg)
        split = cfg.setdefault("split", {})
        split["enabled"] = True
        split["min_net_gain"] = min_gain
        split["cover_penalty_weight"] = penalty_weight
        split["max_cover_growth"] = max_growth
        if args.backend:
            cfg.setdefault("local_ann", {})["backend"] = args.backend
        run_id = (
            f"{args.run_prefix}_gain{min_gain:g}"
            f"_pen{penalty_weight:g}"
            f"_growth{str(max_growth).replace('.', 'p')}"
        )
        run_dir = run_one(cfg, run_id, out_root, args.dry_run)
        if args.dry_run:
            continue
        rows.append(
            summarize_run(
                run_dir,
                {
                    "kind": "adaptive_penalty",
                    "min_net_gain": min_gain,
                    "cover_penalty_weight": penalty_weight,
                    "max_cover_growth": max_growth,
                },
                baseline=baseline_summary,
            )
        )

    if args.dry_run:
        return

    write_csv(out_root / "sweep_summary.csv", rows)
    candidates = [row for row in rows if row.get("kind") == "adaptive_penalty" and row["recall_mean"] >= args.recall_target]
    best_distance = min(candidates, key=lambda row: row["cumulative_distance_count"], default=None)
    best_latency = min(candidates, key=lambda row: row["p95_latency_ms"], default=None)
    best_balanced = min(
        candidates,
        key=lambda row: (
            row["p95_latency_ms"],
            row["covered_cell_count_mean"],
            row["cumulative_distance_count"],
        ),
        default=None,
    )
    best = {
        "best_distance": best_distance,
        "best_latency": best_latency,
        "best_balanced": best_balanced,
        "summary_csv": str(out_root / "sweep_summary.csv"),
    }
    (out_root / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
