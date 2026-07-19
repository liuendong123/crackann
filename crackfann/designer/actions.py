from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionRecord:
    ts: int
    action: str
    cell_id: int
    from_level: int
    to_level: int
    predicted_gain: float
    realized_gain: float
    build_ms: float
    reason: str

    def to_row(self) -> dict:
        return {
            "ts": self.ts,
            "action": self.action,
            "cell_id": self.cell_id,
            "from_level": self.from_level,
            "to_level": self.to_level,
            "predicted_gain": self.predicted_gain,
            "realized_gain": self.realized_gain,
            "build_ms": self.build_ms,
            "reason": self.reason,
        }

