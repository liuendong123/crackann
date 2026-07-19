from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from crackfann.core.distance import squared_l2_batch
from crackfann.core.types import BuildReport, CandidateBatch


class LocalANNBackend(Protocol):
    def build(self, cell_id: int, ids: np.ndarray, vectors: np.ndarray, params: dict | None = None) -> BuildReport:
        ...

    def search(self, handle: str, q: np.ndarray, k: int, budget: object | None = None) -> CandidateBatch:
        ...

    def estimate_query_cost(self, data_count: int, k: int = 10) -> int:
        ...

    def memory_bytes(self, handle: str) -> int:
        ...


class MissingANNDependencyError(RuntimeError):
    pass


@dataclass
class _LocalIndex:
    ids: np.ndarray
    vectors: np.ndarray
    planes: np.ndarray
    buckets: dict[int, np.ndarray]
    memory_bytes: int


@dataclass
class _ExactLocalIndex:
    ids: np.ndarray
    vectors: np.ndarray
    memory_bytes: int


class ExactLocalANNStore:
    """Exact local cell index used as a safe L3 reference."""

    def __init__(self) -> None:
        self._indexes: dict[str, _ExactLocalIndex] = {}
        self._next_generation = 0

    def build(self, cell_id: int, ids: np.ndarray, vectors: np.ndarray, params: dict | None = None) -> BuildReport:
        start = time.perf_counter()
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float32)
        handle = f"exact:{cell_id}:{self._next_generation}"
        self._next_generation += 1
        memory_bytes = int(ids.nbytes + vectors.nbytes)
        self._indexes[handle] = _ExactLocalIndex(ids=ids, vectors=vectors, memory_bytes=memory_bytes)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return BuildReport(handle=handle, build_ms=elapsed_ms, memory_bytes=memory_bytes, build_work=float(ids.size))

    def search(self, handle: str, q: np.ndarray, k: int, budget: object | None = None) -> CandidateBatch:
        index = self._indexes[handle]
        distances = squared_l2_batch(index.vectors, q)
        take = min(k, index.ids.size)
        if take < index.ids.size:
            chosen = np.argpartition(distances, take - 1)[:take]
        else:
            chosen = np.arange(index.ids.size)
        order = np.argsort(distances[chosen], kind="stable")
        chosen = chosen[order]
        return CandidateBatch(
            ids=index.ids[chosen],
            distances=distances[chosen].astype(np.float32, copy=False),
            distance_computations=int(index.ids.size),
            source="local_exact_l3",
        )

    def estimate_query_cost(self, data_count: int, k: int = 10) -> int:
        return int(data_count)

    def memory_bytes(self, handle: str) -> int:
        return self._indexes[handle].memory_bytes


class RandomProjectionANNStore:
    """Dependency-free local ANN prototype.

    It is deliberately simple: random hyperplane signatures route a query to a
    small set of nearby buckets, then exact distances are computed in that
    candidate subset. This gives a real approximate path without FAISS/HNSW.
    """

    def __init__(
        self,
        seed: int = 42,
        num_planes: int = 10,
        default_probe_buckets: int = 64,
        candidate_fraction: float = 0.75,
    ) -> None:
        self.seed = seed
        self.num_planes = num_planes
        self.default_probe_buckets = default_probe_buckets
        self.candidate_fraction = candidate_fraction
        self._indexes: dict[str, _LocalIndex] = {}
        self._next_generation = 0

    def build(self, cell_id: int, ids: np.ndarray, vectors: np.ndarray, params: dict | None = None) -> BuildReport:
        params = params or {}
        start = time.perf_counter()
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float32)
        if ids.size != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same row count")
        planes_count = int(params.get("num_planes", self.num_planes))
        rng = np.random.default_rng(self.seed + int(cell_id))
        planes = rng.normal(0.0, 1.0, size=(planes_count, vectors.shape[1])).astype(np.float32)
        signatures = self._sign_many(vectors, planes)
        bucket_lists: dict[int, list[int]] = {}
        for pos, signature in enumerate(signatures.tolist()):
            bucket_lists.setdefault(int(signature), []).append(pos)
        buckets = {key: np.array(pos_list, dtype=np.int64) for key, pos_list in bucket_lists.items()}
        memory_bytes = int(ids.nbytes + vectors.nbytes + planes.nbytes + signatures.nbytes)
        handle = f"rpann:{cell_id}:{self._next_generation}"
        self._next_generation += 1
        self._indexes[handle] = _LocalIndex(
            ids=ids,
            vectors=vectors,
            planes=planes,
            buckets=buckets,
            memory_bytes=memory_bytes,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        build_work = float(ids.size * max(1.0, math.log2(ids.size + 1.0)))
        return BuildReport(handle=handle, build_ms=elapsed_ms, memory_bytes=memory_bytes, build_work=build_work)

    def search(self, handle: str, q: np.ndarray, k: int, budget: object | None = None) -> CandidateBatch:
        index = self._indexes[handle]
        max_candidates = getattr(budget, "max_candidates", None)
        probe_buckets = int(getattr(budget, "probe_buckets", self.default_probe_buckets))
        if max_candidates is None:
            max_candidates = self.estimate_query_cost(index.ids.size, k)
        max_candidates = max(int(max_candidates), k)

        if max_candidates >= index.ids.size:
            positions = np.arange(index.ids.size, dtype=np.int64)
            candidate_ids = index.ids
            candidate_vectors = index.vectors
            distances = squared_l2_batch(candidate_vectors, q)
            take = min(k, candidate_ids.size)
            if take < candidate_ids.size:
                chosen = np.argpartition(distances, take - 1)[:take]
            else:
                chosen = np.arange(candidate_ids.size)
            order = np.argsort(distances[chosen], kind="stable")
            chosen = chosen[order]
            return CandidateBatch(
                ids=candidate_ids[chosen],
                distances=distances[chosen].astype(np.float32, copy=False),
                distance_computations=int(positions.size),
                predicate_checks=0,
                source="local_exact_l3",
            )

        query_sig = int(self._sign_one(q.astype(np.float32, copy=False), index.planes))
        keys = sorted(index.buckets, key=lambda key: ((key ^ query_sig).bit_count(), key))
        pos_chunks: list[np.ndarray] = []
        total = 0
        for key in keys[: max(1, probe_buckets)]:
            chunk = index.buckets[key]
            pos_chunks.append(chunk)
            total += int(chunk.size)
            if total >= max_candidates:
                break
        if not pos_chunks:
            return CandidateBatch(
                ids=np.empty(0, dtype=np.int64),
                distances=np.empty(0, dtype=np.float32),
                distance_computations=0,
                source="rpann",
            )
        positions = np.concatenate(pos_chunks)
        if positions.size > max_candidates:
            positions = positions[:max_candidates]
        candidate_ids = index.ids[positions]
        candidate_vectors = index.vectors[positions]
        distances = squared_l2_batch(candidate_vectors, q)
        take = min(k, candidate_ids.size)
        if take < candidate_ids.size:
            chosen = np.argpartition(distances, take - 1)[:take]
        else:
            chosen = np.arange(candidate_ids.size)
        order = np.argsort(distances[chosen], kind="stable")
        chosen = chosen[order]
        return CandidateBatch(
            ids=candidate_ids[chosen],
            distances=distances[chosen].astype(np.float32, copy=False),
            distance_computations=int(positions.size),
            predicate_checks=0,
            source="rpann",
        )

    def estimate_query_cost(self, data_count: int, k: int = 10) -> int:
        if data_count <= 0:
            return 0
        fraction_cost = int(math.ceil(data_count * self.candidate_fraction))
        sqrt_cost = int(math.ceil(8 * math.sqrt(data_count)))
        return int(min(data_count, max(4 * k, sqrt_cost, fraction_cost)))

    def memory_bytes(self, handle: str) -> int:
        return self._indexes[handle].memory_bytes

    @staticmethod
    def _sign_many(vectors: np.ndarray, planes: np.ndarray) -> np.ndarray:
        bits = vectors @ planes.T >= 0
        signatures = np.zeros(vectors.shape[0], dtype=np.int64)
        for bit_idx in range(bits.shape[1]):
            signatures |= bits[:, bit_idx].astype(np.int64) << bit_idx
        return signatures

    @staticmethod
    def _sign_one(vector: np.ndarray, planes: np.ndarray) -> int:
        bits = vector @ planes.T >= 0
        signature = 0
        for bit_idx, bit in enumerate(bits.tolist()):
            if bit:
                signature |= 1 << bit_idx
        return int(signature)


@dataclass
class _FaissIndex:
    ids: np.ndarray
    index: object
    memory_bytes: int
    ef_search: int


class FaissLocalANNStore:
    """FAISS local HNSW backend.

    Uses `faiss.IndexHNSWFlat` when the optional `faiss` package is installed.
    """

    def __init__(self, m: int = 16, ef_construction: int = 100, ef_search: int = 64) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._indexes: dict[str, _FaissIndex] = {}
        self._next_generation = 0

    def build(self, cell_id: int, ids: np.ndarray, vectors: np.ndarray, params: dict | None = None) -> BuildReport:
        params = params or {}
        faiss = _import_faiss()
        if not hasattr(faiss, "IndexHNSWFlat"):
            raise MissingANNDependencyError(
                "FAISS is importable, but this installation does not expose IndexHNSWFlat. "
                "Use a FAISS build with HNSW support, or choose local_ann.backend='hnswlib', 'rpann', or 'exact'."
            )
        start = time.perf_counter()
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        m = int(params.get("M", params.get("m", self.m)))
        ef_construction = int(params.get("ef_construction", self.ef_construction))
        ef_search = int(params.get("ef_search", self.ef_search))
        index = faiss.IndexHNSWFlat(vectors.shape[1], m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(vectors)
        handle = f"faiss_hnsw:{cell_id}:{self._next_generation}"
        self._next_generation += 1
        memory_bytes = int(ids.nbytes + vectors.nbytes + ids.size * m * 8)
        self._indexes[handle] = _FaissIndex(ids=ids, index=index, memory_bytes=memory_bytes, ef_search=ef_search)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        build_work = float(ids.size * max(1.0, math.log2(ids.size + 1.0)) * max(1, m))
        return BuildReport(handle=handle, build_ms=elapsed_ms, memory_bytes=memory_bytes, build_work=build_work)

    def search(self, handle: str, q: np.ndarray, k: int, budget: object | None = None) -> CandidateBatch:
        index = self._indexes[handle]
        ef = int(getattr(budget, "ef", 0) or index.ef_search)
        index.index.hnsw.efSearch = max(ef, k)
        distances, positions = index.index.search(np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32), k)
        positions = positions[0]
        distances = distances[0]
        keep = positions >= 0
        positions = positions[keep].astype(np.int64, copy=False)
        distances = distances[keep].astype(np.float32, copy=False)
        return CandidateBatch(
            ids=index.ids[positions],
            distances=distances,
            distance_computations=min(int(index.ids.size), max(int(index.index.hnsw.efSearch), k)),
            source="faiss_hnsw",
        )

    def estimate_query_cost(self, data_count: int, k: int = 10) -> int:
        return int(min(data_count, max(k, self.ef_search)))

    def memory_bytes(self, handle: str) -> int:
        return self._indexes[handle].memory_bytes


@dataclass
class _HnswlibIndex:
    ids: np.ndarray
    index: object
    memory_bytes: int
    ef_search: int


class HnswlibLocalANNStore:
    """hnswlib local HNSW backend."""

    def __init__(self, m: int = 16, ef_construction: int = 100, ef_search: int = 64, seed: int = 42) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._indexes: dict[str, _HnswlibIndex] = {}
        self._next_generation = 0

    def build(self, cell_id: int, ids: np.ndarray, vectors: np.ndarray, params: dict | None = None) -> BuildReport:
        params = params or {}
        hnswlib = _import_hnswlib()
        start = time.perf_counter()
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        m = int(params.get("M", params.get("m", self.m)))
        ef_construction = int(params.get("ef_construction", self.ef_construction))
        ef_search = int(params.get("ef_search", self.ef_search))
        index = hnswlib.Index(space="l2", dim=vectors.shape[1])
        index.init_index(
            max_elements=ids.size,
            ef_construction=ef_construction,
            M=m,
            random_seed=self.seed + int(cell_id),
        )
        local_labels = np.arange(ids.size, dtype=np.int64)
        index.add_items(vectors, local_labels)
        index.set_ef(max(ef_search, 1))
        handle = f"hnswlib:{cell_id}:{self._next_generation}"
        self._next_generation += 1
        memory_bytes = int(ids.nbytes + vectors.nbytes + ids.size * m * 8)
        self._indexes[handle] = _HnswlibIndex(ids=ids, index=index, memory_bytes=memory_bytes, ef_search=ef_search)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        build_work = float(ids.size * max(1.0, math.log2(ids.size + 1.0)) * max(1, m))
        return BuildReport(handle=handle, build_ms=elapsed_ms, memory_bytes=memory_bytes, build_work=build_work)

    def search(self, handle: str, q: np.ndarray, k: int, budget: object | None = None) -> CandidateBatch:
        index = self._indexes[handle]
        ef = int(getattr(budget, "ef", 0) or index.ef_search)
        index.index.set_ef(max(ef, k))
        labels, distances = index.index.knn_query(np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32), k=k)
        labels = labels[0].astype(np.int64, copy=False)
        distances = distances[0].astype(np.float32, copy=False)
        return CandidateBatch(
            ids=index.ids[labels],
            distances=distances,
            distance_computations=min(int(index.ids.size), max(ef, k)),
            source="hnswlib",
        )

    def estimate_query_cost(self, data_count: int, k: int = 10) -> int:
        return int(min(data_count, max(k, self.ef_search)))

    def memory_bytes(self, handle: str) -> int:
        return self._indexes[handle].memory_bytes


def create_local_ann_store(config: dict | None = None, seed: int = 42) -> LocalANNBackend:
    config = config or {}
    backend = str(config.get("backend", config.get("type", "rpann"))).lower()
    if backend in {"exact", "local_exact"}:
        return ExactLocalANNStore()
    if backend in {"rpann", "random_projection", "random-projection"}:
        return RandomProjectionANNStore(
            seed=seed,
            num_planes=int(config.get("num_planes", 10)),
            default_probe_buckets=int(config.get("probe_buckets", 64)),
            candidate_fraction=float(config.get("candidate_fraction", 0.75)),
        )
    if backend in {"faiss", "faiss_hnsw", "faiss-hnsw"}:
        return FaissLocalANNStore(
            m=int(config.get("M", config.get("m", 16))),
            ef_construction=int(config.get("ef_construction", 100)),
            ef_search=int(config.get("ef_search", 64)),
        )
    if backend in {"hnsw", "hnswlib"}:
        return HnswlibLocalANNStore(
            m=int(config.get("M", config.get("m", 16))),
            ef_construction=int(config.get("ef_construction", 100)),
            ef_search=int(config.get("ef_search", 64)),
            seed=seed,
        )
    raise ValueError(f"Unknown local ANN backend: {backend}")


def available_backends() -> dict[str, bool]:
    _ensure_local_ann_deps_path()
    return {
        "exact": True,
        "rpann": True,
        "faiss": _faiss_has_hnsw(),
        "hnswlib": _has_module("hnswlib"),
    }


def _has_module(name: str) -> bool:
    import importlib.util

    _ensure_local_ann_deps_path()
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, OSError, PermissionError):
        return False


def _import_faiss():
    _ensure_local_ann_deps_path()
    try:
        import faiss
    except ImportError as exc:
        raise MissingANNDependencyError(
            "FAISS backend selected, but Python package 'faiss' is not installed. "
            "Install faiss-cpu or choose local_ann.backend='hnswlib', 'rpann', or 'exact'."
        ) from exc
    return faiss


def _faiss_has_hnsw() -> bool:
    if not _has_module("faiss"):
        return False
    try:
        faiss = _import_faiss()
    except (MissingANNDependencyError, ImportError, OSError, PermissionError):
        return False
    return hasattr(faiss, "IndexHNSWFlat")


def _import_hnswlib():
    _ensure_local_ann_deps_path()
    try:
        import hnswlib
    except ImportError as exc:
        raise MissingANNDependencyError(
            "hnswlib backend selected, but Python package 'hnswlib' is not installed. "
            "Install hnswlib or choose local_ann.backend='faiss', 'rpann', or 'exact'."
        ) from exc
    return hnswlib


def _ensure_local_ann_deps_path() -> None:
    deps = Path(__file__).resolve().parents[2] / ".ann_deps"
    if deps.exists():
        deps_str = str(deps)
        if deps_str not in sys.path:
            sys.path.insert(0, deps_str)
