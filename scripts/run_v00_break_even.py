from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crackfann.core.dataset import Dataset
from crackfann.core.distance import topk_from_ids
from crackfann.core.types import FilteredQuery, RangePredicate
from crackfann.materialization.local_ann_store import MissingANNDependencyError, create_local_ann_store
from crackfann.predicate.tree import PredicateTree


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/v00_break_even")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--queries_per_cell", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--range_width", type=float, default=0.06)
    parser.add_argument("--num_planes", type=int, default=10)
    parser.add_argument("--probe_buckets", type=int, default=64)
    parser.add_argument("--candidate_fraction", type=float, default=0.75)
    parser.add_argument("--backend", default="rpann", choices=["exact", "rpann", "faiss", "faiss_hnsw", "hnswlib"])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=100)
    parser.add_argument("--ef_search", type=int, default=64)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.synthetic(n=args.n, d=args.d, seed=args.seed, attr_mode="correlated", noise=0.04)
    tree = PredicateTree.from_quantiles(dataset.attr_values(0), args.cells, attr_id=0)
    values = dataset.attr_values(0)
    rng = np.random.default_rng(args.seed + 1)
    ann = create_local_ann_store(
        {
            "backend": args.backend,
            "num_planes": args.num_planes,
            "probe_buckets": args.probe_buckets,
            "candidate_fraction": args.candidate_fraction,
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
        },
        seed=args.seed,
    )
    try:
        run_break_even(args, out, dataset, tree, values, rng, ann)
    except MissingANNDependencyError as exc:
        payload = {"out": str(out), "backend": args.backend, "error": str(exc)}
        (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2) from exc


def run_break_even(args, out: Path, dataset: Dataset, tree: PredicateTree, values: np.ndarray, rng, ann) -> None:
    break_even_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for cell in tree.cells.values():
        mask = tree.mask_for_cell(values, cell)
        ids = dataset.ids[mask]
        cell.data_count = int(ids.size)
        if ids.size == 0:
            continue
        report = ann.build(cell.cell_id, ids, dataset.vectors[ids], {})
        queries = []
        for i in range(args.queries_per_cell):
            anchor = int(rng.choice(ids))
            vector = dataset.vectors[anchor] + rng.normal(0.0, 0.05, size=dataset.d).astype(np.float32)
            queries.append(
                FilteredQuery(
                    query_id=i,
                    vector=vector.astype(np.float32, copy=False),
                    predicates=(RangePredicate(0, cell.low, cell.high),),
                    k=args.k,
                    timestamp=i,
                    phase="break_even",
                )
            )
        base_costs = []
        l1_costs = []
        l3_costs = []
        l3_recalls = []
        for query in queries:
            predicate = query.predicates[0]
            query_ids = dataset.ids_for_predicates((RangePredicate(0, predicate.low, predicate.high),))
            truth = topk_from_ids(dataset.vectors, query_ids, query.vector, query.k, source="truth")
            l1 = topk_from_ids(dataset.vectors, query_ids, query.vector, query.k, source="l1")
            l3 = ann.search(report.handle, query.vector, query.k)
            base_costs.append(dataset.n)
            l1_costs.append(l1.distance_computations)
            l3_costs.append(l3.distance_computations)
            expected = set(truth.ids.tolist())
            actual = set(l3.ids.tolist())
            l3_recalls.append(1.0 if not expected else len(expected & actual) / len(expected))

        base_mean = float(np.mean(base_costs))
        l1_mean = float(np.mean(l1_costs))
        l3_mean = float(np.mean(l3_costs))
        saving_l1 = max(0.0, base_mean - l1_mean)
        saving_l3_over_l1 = max(0.0, l1_mean - l3_mean)
        build_cost_units = float(ids.size * max(1.0, math.log2(ids.size + 1.0)))
        break_even_l3 = build_cost_units / saving_l3_over_l1 if saving_l3_over_l1 > 0 else float("inf")
        break_even_rows.append(
            {
                "cell_id": cell.cell_id,
                "data_count": int(ids.size),
                "base_distance_per_query": base_mean,
                "l1_distance_per_query": l1_mean,
                "l3_distance_per_query": l3_mean,
                "l1_saving_vs_base": saving_l1,
                "l3_saving_vs_l1": saving_l3_over_l1,
                "l3_build_ms": report.build_ms,
                "l3_build_cost_units": build_cost_units,
                "predicted_break_even_queries": break_even_l3,
                "l3_recall_mean": float(np.mean(l3_recalls)),
            }
        )

    if break_even_rows:
        comparison_rows.append(
            {
                "level": "L1",
                "mean_distance_per_query": float(np.mean([row["l1_distance_per_query"] for row in break_even_rows])),
                "mean_saving_vs_base": float(np.mean([row["l1_saving_vs_base"] for row in break_even_rows])),
                "mean_recall": 1.0,
            }
        )
        comparison_rows.append(
            {
                "level": f"L3_{args.backend}",
                "mean_distance_per_query": float(np.mean([row["l3_distance_per_query"] for row in break_even_rows])),
                "mean_saving_vs_base": float(np.mean([row["base_distance_per_query"] - row["l3_distance_per_query"] for row in break_even_rows])),
                "mean_recall": float(np.mean([row["l3_recall_mean"] for row in break_even_rows])),
            }
        )

    broad_rows = []
    for width_cells in (2, 4, 8, args.cells):
        selected = [tree.cells[i] for i in range(min(width_cells, args.cells))]
        low = selected[0].low
        high = selected[-1].high
        ids = dataset.ids_for_predicates((RangePredicate(0, low, high),))
        broad_rows.append(
            {
                "covered_cells": width_cells,
                "candidate_count": int(ids.size),
                "base_path_distance_count": dataset.n,
                "naive_cell_exact_distance_count": int(ids.size),
                "base_protects_broad_query": int(width_cells >= max(4, args.cells // 2)),
            }
        )

    write_csv(out / "break_even.csv", break_even_rows)
    write_csv(out / "level_comparison.csv", comparison_rows)
    write_csv(out / "broad_query.csv", broad_rows)
    summary = {
        "cells": args.cells,
        "dataset_n": dataset.n,
        "queries_per_cell": args.queries_per_cell,
        "num_planes": args.num_planes,
        "probe_buckets": args.probe_buckets,
        "candidate_fraction": args.candidate_fraction,
        "backend": args.backend,
        "M": args.M,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "go_l3_cells_under_50000_queries": sum(
            1 for row in break_even_rows if float(row["predicted_break_even_queries"]) <= 50000.0
        ),
        "mean_l3_recall": float(np.mean([row["l3_recall_mean"] for row in break_even_rows])) if break_even_rows else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
