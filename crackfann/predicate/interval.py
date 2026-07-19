from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def overlaps(self, other: "Interval") -> bool:
        return self.low <= other.high and other.low <= self.high

    def contains(self, other: "Interval") -> bool:
        return self.low <= other.low and other.high <= self.high

    @property
    def width(self) -> float:
        return self.high - self.low

