from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackfann.predicate.cell import PredicateCell


@dataclass(frozen=True)
class CoverPart:
    cell_id: int
    low: float
    high: float
    full: bool


class PredicateTree:
    def __init__(self, attr_id: int, cells: dict[int, PredicateCell]) -> None:
        self.attr_id = attr_id
        self.cells = cells

    @classmethod
    def from_quantiles(cls, values: np.ndarray, num_cells: int, attr_id: int = 0) -> "PredicateTree":
        if num_cells <= 0:
            raise ValueError("num_cells must be positive")
        values = np.asarray(values, dtype=np.float32)
        quantiles = np.linspace(0.0, 1.0, num_cells + 1)
        boundaries = np.quantile(values, quantiles)
        if np.unique(boundaries).size < num_cells + 1:
            boundaries = np.linspace(float(values.min()), float(values.max()), num_cells + 1)
        boundaries[0] = float(values.min()) - 1e-6
        boundaries[-1] = float(values.max()) + 1e-6
        cells: dict[int, PredicateCell] = {}
        for idx in range(num_cells):
            cells[idx] = PredicateCell(
                cell_id=idx,
                parent_id=None,
                left_child=None,
                right_child=None,
                low=float(boundaries[idx]),
                high=float(boundaries[idx + 1]),
                data_count=0,
            )
        tree = cls(attr_id=attr_id, cells=cells)
        tree.validate()
        return tree

    @property
    def leaf_ids(self) -> list[int]:
        return sorted(self.cells)

    def cover(self, low: float, high: float) -> list[CoverPart]:
        parts: list[CoverPart] = []
        for cell_id in self.leaf_ids:
            cell = self.cells[cell_id]
            if low <= cell.high and cell.low <= high:
                part_low = max(low, cell.low)
                part_high = min(high, cell.high)
                full = low <= cell.low and cell.high <= high
                parts.append(CoverPart(cell_id=cell_id, low=part_low, high=part_high, full=full))
        return parts

    def mask_for_cell(self, values: np.ndarray, cell: PredicateCell) -> np.ndarray:
        if cell.cell_id == self.leaf_ids[-1]:
            return (values >= cell.low) & (values <= cell.high)
        return (values >= cell.low) & (values < cell.high)

    def validate(self) -> None:
        previous_high: float | None = None
        for cell_id in self.leaf_ids:
            cell = self.cells[cell_id]
            if cell.high <= cell.low:
                raise ValueError(f"Cell {cell_id} has non-positive width")
            if previous_high is not None and cell.low < previous_high - 1e-9:
                raise ValueError("Cells overlap")
            previous_high = cell.high

