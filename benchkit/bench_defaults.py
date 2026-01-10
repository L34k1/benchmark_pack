from __future__ import annotations

DEFAULT_WINDOW_S = 60.0
DEFAULT_LOAD_DURATION_MULTIPLIER = 10.0
DEFAULT_STEPS = 200
DEFAULT_TARGET_INTERVAL_MS = 16.0


def default_load_duration_s(window_s: float, multiplier: float = DEFAULT_LOAD_DURATION_MULTIPLIER) -> float:
    return float(window_s) * float(multiplier)
