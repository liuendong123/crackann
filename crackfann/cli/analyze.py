from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--recall_target", type=float, default=0.95)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = read_csv(run_dir / "summary_by_phase.csv")
    actions = read_csv(run_dir / "action_log.csv")
    queries = read_csv(run_dir / "query_log.csv")
    if not queries:
        print("No query_log.csv rows found.")
        return
    recall_values = [float(row["recall"]) for row in queries]
    recall_mean = sum(recall_values) / len(recall_values)
    recall_min = min(recall_values)
    recall_violations = sum(1 for value in recall_values if value < args.recall_target)
    distance_total = sum(float(row["distance_count"]) for row in queries)
    quality_risks = sum(int(row["quality_risk"]) for row in queries)
    print(f"Run: {run_dir}")
    print(f"Queries: {len(queries)}")
    print(f"Actions: {len(actions)}")
    print(f"Recall mean: {recall_mean:.4f}")
    print(f"Recall min: {recall_min:.4f}")
    print(f"Recall target pass (mean): {recall_mean >= args.recall_target}")
    print(f"Recall violations: {recall_violations}")
    print(f"Cumulative distance count: {distance_total:.0f}")
    print(f"Quality risk count: {quality_risks}")
    if summary:
        print("Phases:")
        for row in summary:
            print(
                f"  {row['phase']}: q={row['queries']} recall={float(row['recall_mean']):.4f} "
                f"p95_ms={float(row['p95_latency_ms']):.3f} dist_mean={float(row['distance_count_mean']):.1f}"
            )


if __name__ == "__main__":
    main()
