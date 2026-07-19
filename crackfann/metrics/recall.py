from __future__ import annotations

import numpy as np


def recall_at_k(expected_ids: np.ndarray, actual_ids: np.ndarray, k: int) -> float:
    expected = set(np.asarray(expected_ids, dtype=np.int64)[:k].tolist())
    if not expected:
        return 1.0
    actual = set(np.asarray(actual_ids, dtype=np.int64)[:k].tolist())
    return len(expected & actual) / float(len(expected))

