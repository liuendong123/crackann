from __future__ import annotations

import numpy as np

from crackfann.core.types import CandidateBatch


def squared_l2_batch(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    diff = vectors - query.astype(vectors.dtype, copy=False)
    return np.einsum("ij,ij->i", diff, diff)


def topk_from_ids(
    vectors: np.ndarray,
    ids: np.ndarray,
    query: np.ndarray,
    k: int,
    source: str = "exact",
    predicate_checks: int = 0,
) -> CandidateBatch:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return CandidateBatch(
            ids=np.empty(0, dtype=np.int64),
            distances=np.empty(0, dtype=np.float32),
            distance_computations=0,
            predicate_checks=predicate_checks,
            source=source,
        )

    local_vectors = vectors[ids]
    distances = squared_l2_batch(local_vectors, query)
    take = min(k, ids.size)
    if take < ids.size:
        candidate_pos = np.argpartition(distances, take - 1)[:take]
    else:
        candidate_pos = np.arange(ids.size)
    order = np.argsort(distances[candidate_pos], kind="stable")
    pos = candidate_pos[order]
    return CandidateBatch(
        ids=ids[pos],
        distances=distances[pos].astype(np.float32, copy=False),
        distance_computations=int(ids.size),
        predicate_checks=predicate_checks,
        source=source,
    )


def merge_topk(batches: list[CandidateBatch], k: int) -> CandidateBatch:
    if not batches:
        return CandidateBatch(
            ids=np.empty(0, dtype=np.int64),
            distances=np.empty(0, dtype=np.float32),
            distance_computations=0,
            predicate_checks=0,
            source="merge",
        )

    best: dict[int, float] = {}
    distance_count = 0
    predicate_checks = 0
    sources = []
    for batch in batches:
        distance_count += batch.distance_computations
        predicate_checks += batch.predicate_checks
        sources.append(batch.source)
        for obj_id, distance in zip(batch.ids.tolist(), batch.distances.tolist()):
            previous = best.get(int(obj_id))
            if previous is None or distance < previous:
                best[int(obj_id)] = float(distance)

    if not best:
        ids = np.empty(0, dtype=np.int64)
        distances = np.empty(0, dtype=np.float32)
    else:
        sorted_items = sorted(best.items(), key=lambda item: (item[1], item[0]))[:k]
        ids = np.array([item[0] for item in sorted_items], dtype=np.int64)
        distances = np.array([item[1] for item in sorted_items], dtype=np.float32)

    return CandidateBatch(
        ids=ids,
        distances=distances,
        distance_computations=distance_count,
        predicate_checks=predicate_checks,
        source="+".join(sorted(set(sources))),
    )

