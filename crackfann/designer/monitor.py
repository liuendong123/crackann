from __future__ import annotations


def ema(previous: float, value: float, alpha: float) -> float:
    if previous == 0.0:
        return float(value)
    return (1.0 - alpha) * previous + alpha * float(value)

