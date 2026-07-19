from __future__ import annotations

from dataclasses import dataclass

from crackfann.predicate.cell import MaterializationLevel, PredicateCell


@dataclass
class MaterializationCatalog:
    cells: dict[int, PredicateCell]

    def promote(self, cell_id: int, level: MaterializationLevel) -> None:
        cell = self.cells[cell_id]
        if level < cell.level:
            raise ValueError("Use demote for level decreases")
        cell.level = level
        cell.generation += 1

    def demote(self, cell_id: int, level: MaterializationLevel) -> None:
        cell = self.cells[cell_id]
        if level > cell.level:
            raise ValueError("Use promote for level increases")
        cell.level = level
        cell.generation += 1

