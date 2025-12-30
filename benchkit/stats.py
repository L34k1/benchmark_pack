from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def pctile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def summarize_latency_ms(lat_ms: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(lat_ms), dtype=float)
    a = a[np.isfinite(a)]
    return {
        "lat_p50_ms": pctile(a, 50),
        "lat_p95_ms": pctile(a, 95),
        "lat_max_ms": float(np.max(a)) if a.size else float("nan"),
        "n": float(a.size),
    }
