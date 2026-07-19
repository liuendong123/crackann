from __future__ import annotations

import numpy as np


class BitmapStore:
    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._next_generation = 0

    def build(self, cell_id: int, object_ids: np.ndarray) -> str:
        handle = f"bitmap:{cell_id}:{self._next_generation}"
        self._next_generation += 1
        self._arrays[handle] = np.asarray(object_ids, dtype=np.int64)
        return handle

    def get_ids(self, handle: str) -> np.ndarray:
        return self._arrays[handle]

    def cardinality(self, handle: str) -> int:
        return int(self._arrays[handle].size)

    def union(self, handles: list[str]) -> np.ndarray:
        if not handles:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate([self._arrays[handle] for handle in handles]))

    def memory_bytes(self, handle: str) -> int:
        return int(self._arrays[handle].nbytes)

