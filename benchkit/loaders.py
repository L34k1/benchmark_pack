from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LoadedSegment:
    """
    Canonical loaded segment for visualization benchmarks.

    - times_s: shape (n_samples,)
    - data:    shape (n_channels, n_samples), float32, normalized per-channel
    - fs_hz:   sampling frequency in Hz if known, else NaN
    - meta:    loader-specific metadata (selected channels, dataset name, etc.)
    """
    times_s: np.ndarray
    data: np.ndarray
    fs_hz: float
    meta: Dict[str, Any]


def _normalize_per_channel(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    out = np.empty_like(data, dtype=np.float32)
    for ch in range(data.shape[0]):
        x = data[ch].astype(np.float32, copy=False)
        x = x - float(x.mean())
        sd = float(x.std())
        if not sd:
            sd = 1.0
        out[ch] = x / sd
    return out


def load_edf_segment_pyedflib(
    path: Path,
    load_start_s: float,
    load_duration_s: float,
    n_channels_max: int,
    *,
    allow_resample: bool = False,
) -> LoadedSegment:
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        labels = [str(x).strip() for x in f.getSignalLabels()]
        ns_all = [int(x) for x in f.getNSamples()]
        fs_all = [float(f.getSampleFrequency(i)) for i in range(len(labels))]

        if not fs_all:
            raise RuntimeError(f"No signals found in EDF: {path}")

        fs0 = fs_all[0]
        cand = []
        for i, (lab, fs_i, ns_i) in enumerate(zip(labels, fs_all, ns_all)):
            if ns_i <= 1 or fs_i <= 0:
                continue
            if "annot" in lab.lower():
                continue
            if (not allow_resample) and abs(fs_i - fs0) > 1e-6:
                continue
            cand.append(i)

        if not cand:
            cand = [i for i, (fs_i, ns_i) in enumerate(zip(fs_all, ns_all)) if fs_i > 0 and ns_i > 1]
        if not cand:
            raise RuntimeError(f"No usable numeric channels found in EDF: {path}")

        cand = cand[: max(1, int(n_channels_max))]
        n_min_total = min(ns_all[i] for i in cand)
        total_dur = n_min_total / fs0

        start = max(0.0, min(float(load_start_s), max(0.0, total_dur - 1e-6)))
        start_samp = int(math.floor(start * fs0))

        n_avail = n_min_total - start_samp
        if n_avail <= 1:
            raise ValueError(
                f"Requested start={start:.6f}s leaves no data. total_dur={total_dur:.3f}s, fs0={fs0}"
            )

        max_dur = n_avail / fs0
        dur = min(float(load_duration_s), max_dur)
        n_samples = min(int(math.floor(dur * fs0)), n_avail)
        if n_samples <= 1:
            raise ValueError(f"Computed n_samples={n_samples} too small.")

        data = np.zeros((len(cand), n_samples), dtype=np.float32)
        for out_ch, ch in enumerate(cand):
            sig = np.asarray(f.readSignal(ch, start_samp, n_samples), dtype=np.float32)
            if sig.size < n_samples:
                sig = np.pad(sig, (0, n_samples - sig.size), mode="edge")
            data[out_ch, :] = sig

    data = _normalize_per_channel(data)
    times = (np.arange(n_samples, dtype=np.float32) / fs0) + float(start)
    return LoadedSegment(
        times_s=times,
        data=data,
        fs_hz=float(fs0),
        meta={
            "path": str(path),
            "selected_channels": cand,
            "labels": [labels[i] for i in cand],
            "n_ch_total": int(len(labels)),
            "n_ch_used": int(len(cand)),
            "effective_n_ch": int(len(cand)),
            "fs0": float(fs0),
            "start_s": float(start),
            "duration_s": float(n_samples / fs0),
        },
    )


def _iter_nwb_timeseries(nwbfile: Any) -> Iterable[Tuple[str, Any]]:
    for name, obj in getattr(nwbfile, "acquisition", {}).items():
        yield f"acquisition/{name}", obj

    processing = getattr(nwbfile, "processing", {})
    for mod_name, mod in processing.items():
        try:
            data_ifaces = getattr(mod, "data_interfaces", {})
        except Exception:
            data_ifaces = {}
        for name, obj in data_ifaces.items():
            yield f"processing/{mod_name}/{name}", obj


def _pick_nwb_series(nwbfile: Any, series_path: Optional[str]) -> Tuple[str, Any]:
    if series_path:
        parts = series_path.strip("/").split("/")
        if not parts:
            raise ValueError("Empty series_path")
        if parts[0] == "acquisition" and len(parts) == 2:
            return series_path, nwbfile.acquisition[parts[1]]
        if parts[0] == "processing" and len(parts) >= 3:
            mod = nwbfile.processing[parts[1]]
            key = "/".join(parts[2:])
            if hasattr(mod, "data_interfaces") and key in mod.data_interfaces:
                return series_path, mod.data_interfaces[key]
            return series_path, mod.data_interfaces[parts[2]]
        raise ValueError(f"Unsupported series_path format: {series_path}")

    best = None
    for path, obj in _iter_nwb_timeseries(nwbfile):
        if not hasattr(obj, "data"):
            continue
        try:
            shape = tuple(getattr(obj.data, "shape", ()))
        except Exception:
            shape = ()
        if len(shape) != 2:
            continue

        rate = getattr(obj, "rate", None)
        if rate is None:
            rate = getattr(obj, "sampling_rate", None)

        score = 0
        if rate is not None:
            score += 10
        if shape[0] >= shape[1]:
            score += 2
        if best is None or score > best[0]:
            best = (score, path, obj)

    if best is None:
        raise RuntimeError("No suitable 2D TimeSeries-like object found in NWB file. Provide --nwb-series-path.")
    return best[1], best[2]


def load_nwb_segment_pynwb(
    path: Path,
    load_start_s: float,
    load_duration_s: float,
    n_channels_max: int,
    *,
    series_path: Optional[str] = None,
    time_dim: str = "auto",  # auto|time_first|time_last
) -> LoadedSegment:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), mode="r") as io:
        nwbfile = io.read()
        sel_path, series = _pick_nwb_series(nwbfile, series_path)

        data_obj = series.data
        shape = tuple(getattr(data_obj, "shape", ()))
        if len(shape) != 2:
            raise RuntimeError(f"Selected series is not 2D: {sel_path} shape={shape}")

        fs = getattr(series, "rate", None)
        if fs is None:
            fs = getattr(series, "sampling_rate", None)
        if fs is None:
            ts = getattr(series, "timestamps", None)
            if ts is not None:
                ts = np.asarray(ts)
                if ts.size >= 2:
                    fs = 1.0 / float(np.median(np.diff(ts)))
        fs = float(fs) if fs is not None else float("nan")

        if time_dim == "time_first":
            time_first = True
        elif time_dim == "time_last":
            time_first = False
        else:
            time_first = (shape[0] >= shape[1])

        n_time = int(shape[0] if time_first else shape[1])
        n_ch_total = int(shape[1] if time_first else shape[0])
        n_ch = min(int(n_channels_max), n_ch_total)

        if math.isnan(fs) or fs <= 0:
            raise RuntimeError(
                f"Could not infer sampling rate for {sel_path}. Provide a series with .rate or timestamps."
            )

        total_dur = n_time / fs
        start = max(0.0, min(float(load_start_s), max(0.0, total_dur - 1e-6)))
        start_idx = int(math.floor(start * fs))
        n_avail = n_time - start_idx
        if n_avail <= 1:
            raise ValueError(f"Start index out of range: start={start}, start_idx={start_idx}, n_time={n_time}")

        dur = min(float(load_duration_s), n_avail / fs)
        n_samp = min(int(math.floor(dur * fs)), n_avail)
        if n_samp <= 1:
            raise ValueError(f"Too few samples: n_samp={n_samp}")

        if time_first:
            slab = np.asarray(data_obj[start_idx:start_idx + n_samp, :n_ch], dtype=np.float32)
            data = slab.T
        else:
            slab = np.asarray(data_obj[:n_ch, start_idx:start_idx + n_samp], dtype=np.float32)
            data = slab

    data = _normalize_per_channel(data)
    times = (np.arange(n_samp, dtype=np.float32) / fs) + float(start)
    return LoadedSegment(
        times_s=times,
        data=data,
        fs_hz=float(fs),
        meta={
            "path": str(path),
            "series_path": sel_path,
            "time_first": bool(time_first),
            "n_time": n_time,
            "n_ch_total": n_ch_total,
            "n_ch_used": n_ch,
            "effective_n_ch": n_ch,
            "start_s": float(start),
            "duration_s": float(n_samp / fs),
        },
    )


def decimate_for_display(times_s: np.ndarray, data: np.ndarray, max_points_per_trace: int) -> Tuple[np.ndarray, np.ndarray, int]:
    n = int(times_s.shape[0])
    if max_points_per_trace <= 0 or n <= max_points_per_trace:
        return times_s, data, 1
    factor = int(math.ceil(n / max_points_per_trace))
    idx = np.arange(0, n, factor, dtype=np.int64)
    return times_s[idx], data[:, idx], factor
