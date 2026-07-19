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
