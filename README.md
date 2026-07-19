# CrackFANN

CrackFANN is an executable research prototype for workload-driven physical
design in filtered approximate nearest-neighbor search.

This first implementation is intentionally small and dependency-light. It uses:

- a global exact base index as the safety path;
- a quantile Predicate Cell Tree over one numeric attribute;
- L1 bitmap/exact candidate materialization for every fixed cell;
- an optional random-projection local ANN prototype for L3 cells;
- online promotion based on measured/query-cost break-even;
- CSV/JSON logs for kill-tests and paper-style analysis.

The high-performance FAISS/HNSW backends in the design document can replace the
current pure NumPy backends through the existing interfaces.

## Quick Start

```powershell
python -m unittest discover -s tests/unit
python scripts/run_v00_break_even.py --out outputs/v00_break_even
python -m crackfann.cli.run_workload --config configs/paper/synthetic_longtail.json --run_id smoke
python -m crackfann.cli.analyze --run_dir outputs/smoke
```

`configs/paper/synthetic_longtail.json` keeps L3 in safe local-exact mode, so it
is a correctness and logging smoke test. Use
`configs/paper/synthetic_l3_experimental.json` to evaluate the approximate
random-projection L3 path; that run can trade recall for fewer distance
computations and should be treated as a kill-test, not as the default system.

Check optional ANN backend availability:

```powershell
python -m crackfann.cli.check_backends
```

When installed, use the real local ANN backends:

```powershell
python scripts/run_v00_break_even.py --backend faiss_hnsw --ef_search 64 --out outputs/v00_faiss
python scripts/run_v00_break_even.py --backend hnswlib --ef_search 64 --out outputs/v00_hnswlib
python -m crackfann.cli.run_workload --config configs/paper/synthetic_faiss_hnsw.json --run_id faiss_hnsw
python -m crackfann.cli.run_workload --config configs/paper/synthetic_hnswlib.json --run_id hnswlib
```

The FAISS/HNSW paths are optional: if `faiss` or `hnswlib` is missing, selecting
that backend fails loudly with an install hint instead of silently falling back.

## v0.2 Adaptive Tree

The v0.2 prototype can split predicate cells from repeated query-boundary
observations before promoting hot cells to L3. Run it on a Linux server with
`hnswlib` or FAISS available:

```bash
python -m crackfann.cli.run_workload \
  --config configs/paper/synthetic_adaptive_hnswlib.json \
  --run_id adaptive_hnswlib

python -m crackfann.cli.analyze \
  --run_dir outputs/adaptive_hnswlib \
  --recall_target 0.95
```

Compare it against a fixed-cell run with the same backend. The first v0.2 gate is
whether `action_log.csv` contains useful `SPLIT` actions before `PROMOTE`, while
keeping recall at target and lowering cumulative distance count or memory-time.

Fixed coarse baselines with the same template workload are provided at
`configs/paper/synthetic_fixed4_template_hnswlib.json` and
`configs/paper/synthetic_fixed4_template_faiss_hnsw.json`.

## v0.3 Scheduler Metrics And Split Penalty

The v0.3 gate compares:

- fixed coarse cells;
- adaptive splitting without penalty;
- adaptive splitting with scan-saving minus cover-growth penalty.

Example HNSWLIB run:

```bash
python -m crackfann.cli.run_workload \
  --config configs/paper/synthetic_fixed4_template_hnswlib.json \
  --run_id fixed4_template_hnswlib

python -m crackfann.cli.run_workload \
  --config configs/paper/synthetic_adaptive_no_penalty_hnswlib.json \
  --run_id adaptive_no_penalty_hnswlib

python -m crackfann.cli.run_workload \
  --config configs/paper/synthetic_adaptive_hnswlib.json \
  --run_id adaptive_penalty_hnswlib
```

`query_log.csv` and `summary_by_phase.csv` now include scheduler diagnostics:
`covered_cell_count`, `scheduler_steps`, `l3_cells`,
`exact_residual_cells`, `l3_distance_count`, and
`exact_residual_distance_count`. The v0.3 target is to keep recall at target,
reduce cumulative distance count, and keep `covered_cell_count_mean` and P95
latency close to or below the no-penalty adaptive run.

Run the penalty sweep:

```bash
python scripts/run_v03_penalty_sweep.py \
  --base_config configs/paper/synthetic_adaptive_hnswlib.json \
  --fixed_config configs/paper/synthetic_fixed4_template_hnswlib.json \
  --out_root outputs/v03_penalty_sweep_hnswlib \
  --min_net_gain 25,50,100 \
  --cover_penalty_weight 200,400,800 \
  --max_cover_growth 0.35,0.5,1.0
```

The sweep writes `sweep_summary.csv` plus `best.json`. Use `--limit 2` for a
quick smoke test before running the full grid.

## Current Scope

This is v0.0/v0.1 of the plan:

- fixed quantile cells;
- one numeric filter attribute;
- online L1-to-L3 promotion;
- long-tail, emerging, drift, recurring, and mixed synthetic workloads;
- reproducible logs and summaries.

The next research step is to run the break-even test before implementing a more
complex adaptive tree. If local materialization cannot beat both no-materialize
and all-materialize extremes in cumulative cost, stop or pivot to lighter L1/L2.
