from __future__ import annotations

import argparse
import json

from crackfann.cli.common import ensure_output_dir, load_config
from crackfann.core.dataset import Dataset
from crackfann.metrics.logger import RunLogger
from crackfann.metrics.recall import recall_at_k
from crackfann.system import CrackFANNSystem
from crackfann.workload.generator import WorkloadGenerator


def build_workload(generator: WorkloadGenerator, cfg: dict):
    kind = cfg.get("kind", "long_tail")
    n_queries = int(cfg.get("n_queries", 1000))
    if kind == "long_tail":
        return generator.long_tail(
            n_queries=n_queries,
            zipf_s=float(cfg.get("zipf_s", 1.2)),
            hot_regions=int(cfg.get("hot_regions", 8)),
            range_width=float(cfg.get("range_width", 0.05)),
            k=int(cfg.get("k", 10)),
            broad_ratio=float(cfg.get("broad_ratio", 0.0)),
        )
    if kind == "emerging":
        return generator.emerging(
            cold_queries=int(cfg.get("cold_queries", n_queries // 4)),
            ramp_queries=int(cfg.get("ramp_queries", n_queries // 4)),
            stable_queries=int(cfg.get("stable_queries", n_queries // 2)),
            range_width=float(cfg.get("range_width", 0.05)),
            k=int(cfg.get("k", 10)),
        )
    if kind == "drift":
        return generator.drift(
            n_queries=n_queries,
            range_width=float(cfg.get("range_width", 0.05)),
            k=int(cfg.get("k", 10)),
            abrupt=bool(cfg.get("abrupt", True)),
        )
    if kind == "recurring":
        return generator.recurring(
            n_queries=n_queries,
            period=int(cfg.get("period", 500)),
            range_width=float(cfg.get("range_width", 0.05)),
            k=int(cfg.get("k", 10)),
        )
    if kind == "mixed":
        return generator.mixed(
            n_queries=n_queries,
            range_width=float(cfg.get("range_width", 0.05)),
            k=int(cfg.get("k", 10)),
            broad_ratio=float(cfg.get("broad_ratio", 0.15)),
        )
    raise ValueError(f"Unknown workload kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_output_dir(args.run_id, args.out)
    dataset_cfg = cfg.get("dataset", {})
    dataset = Dataset.synthetic(
        n=int(dataset_cfg.get("n", 10000)),
        d=int(dataset_cfg.get("d", 32)),
        seed=int(cfg.get("seed", 42)),
        attr_mode=dataset_cfg.get("attr_mode", "correlated"),
        noise=float(dataset_cfg.get("noise", 0.05)),
    )
    system = CrackFANNSystem.from_config(cfg)
    system.load_dataset(dataset)
    system.build_base_index()
    system.initialize_fixed_cells(num_cells=int(cfg.get("predicate_tree", {}).get("num_cells", 16)))

    generator = WorkloadGenerator(dataset, seed=int(cfg.get("seed", 42)) + 1)
    workload = build_workload(generator, cfg.get("workload", {}))
    logger = RunLogger()
    for query in workload:
        result = system.search(query)
        truth = dataset.exact_search(query)
        recall = recall_at_k(truth.ids, result.ids, query.k)
        result.recall = recall
        candidate_count = dataset.count_for_predicates(query.predicates)
        logger.record_query(query, result, candidate_count, recall)
    logger.extend_actions([record.to_row() for record in system.action_log])
    logger.extend_cells(system.snapshot_cells(workload[-1].timestamp if workload else 0))
    system.save_run(output_dir)
    logger.write(
        output_dir,
        manifest={
            "run_id": args.run_id,
            "config": args.config,
            "dataset_n": dataset.n,
            "dataset_d": dataset.d,
            "queries": len(workload),
        },
    )
    print(json.dumps({"run_dir": str(output_dir), "queries": len(workload), "actions": len(system.action_log)}, indent=2))


if __name__ == "__main__":
    main()

