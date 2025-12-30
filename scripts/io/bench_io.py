#!/usr/bin/env python3
"""
scripts/io/bench_io.py

I/O benchmark for EDF files across three Python stacks:
- pyEDFlib
- Neo
- MNE

Profiles
- lazy    : minimal open + read a `window_s` slice from disk
- preload : full load at open, then slice in RAM

Outputs (under --out-root)
- outputs/IO/IO_STACKS/<tag>/io_raw.csv
- outputs/IO/IO_STACKS/<tag>/io_summary.csv
- outputs/IO/IO_STACKS/<tag>/manifest.json

Note
The historical column name T_read_10s_ch0_s is preserved for backward compatibility,
but it measures reading `window_s` seconds (not necessarily 10s). The canonical
name in this pack is T_read_window_ch0_s.
"""
from __future__ import annotations

import argparse
import pathlib
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from benchkit.common import out_dir, write_manifest, utc_stamp
from benchkit.lexicon import BENCH_IO, FMT_EDF


PROFILES = ("lazy", "preload")
LIBS = ("pyedflib", "neo", "mne")


def bench_pyedflib(path: pathlib.Path, profile: str, window_s: float, n_channels: int) -> dict:
    """
    pyEDFlib benchmark.

    lazy :
      - T_open_s  : EdfReader(...) only (metadata + header)
      - T_read    : readSignal(ch) for n_channels over `window_s` seconds

    preload :
      - T_open_s  : EdfReader(...) + full read of all channels
      - T_read    : slicing the first `window_s` seconds in RAM (channel 0)
    """
    if profile not in {"lazy", "preload"}:
        raise ValueError(f"Unknown profile for pyEDFlib: {profile}")

    import pyedflib

    t0 = time.perf_counter()
    f = pyedflib.EdfReader(str(path))
    t1 = time.perf_counter()

    fs = float(f.getSampleFrequency(0))
    n_available = int(f.signals_in_file)
    n_channels = min(int(n_channels), n_available)

    if profile == "lazy":
        # Close and reopen to isolate "open" from "read" more cleanly
        f.close()

        t_open0 = time.perf_counter()
        f = pyedflib.EdfReader(str(path))
        t_open1 = time.perf_counter()

        n_samples = min(int(window_s * fs), int(f.getNSamples()[0]))
        t_read0 = time.perf_counter()
        data = []
        for ch in range(n_channels):
            sig = f.readSignal(ch, start=0, n=n_samples)
            data.append(sig)
        t_read1 = time.perf_counter()
        f.close()

        data = np.array(data)
        return {
            "T_open_s": t_open1 - t_open0,
            "T_read_window_ch0_s": t_read1 - t_read0,
            "fs": fs,
            "n_channels": n_channels,
            "n_samples_ch0": int(n_samples),
        }

    # preload
    t_open0 = time.perf_counter()
    # Already opened as f; emulate "open+load" by reading full file here
    all_ch = []
    for ch in range(n_channels):
        sig = f.readSignal(ch)
        all_ch.append(sig)
    t_open1 = time.perf_counter()
    f.close()

    # slice in RAM (channel 0)
    ch0 = np.asarray(all_ch[0])
    n_samples = min(int(window_s * fs), int(ch0.shape[0]))
    t_read0 = time.perf_counter()
    _ = ch0[:n_samples]
    t_read1 = time.perf_counter()

    return {
        "T_open_s": t_open1 - t_open0,
        "T_read_window_ch0_s": t_read1 - t_read0,
        "fs": fs,
        "n_channels": n_channels,
        "n_samples_ch0": int(n_samples),
    }


def bench_neo(path: pathlib.Path, profile: str, window_s: float, n_channels: int) -> dict:
    """
    Neo benchmark (EDF via neo.io.EDFIO).

    This mirrors the original script's behavior, but records the same canonical fields.
    """
    if profile not in {"lazy", "preload"}:
        raise ValueError(f"Unknown profile for Neo: {profile}")

    try:
        from neo.io import EDFIO
    except Exception as e:
        return {"error": f"neo unavailable: {e}"}

    io = EDFIO(str(path))

    # Attempt "lazy" open through read_segment(lazy=True) if supported.
    t_open0 = time.perf_counter()
    seg = None
    if profile == "lazy":
        try:
            seg = io.read_segment(lazy=True)
        except TypeError:
            seg = io.read_segment()
    else:
        seg = io.read_segment()
    t_open1 = time.perf_counter()

    # Extract fs and slice semantics conservatively.
    # Neo may represent signals as AnalogSignal objects with sampling_rate.
    try:
        asig0 = seg.analogsignals[0]
        fs = float(asig0.sampling_rate)
    except Exception:
        fs = float("nan")

    # Read: force-load (if lazy) + slice channel 0
    t_read0 = time.perf_counter()
    try:
        asig0 = seg.analogsignals[0]
        if profile == "lazy":
            try:
                _ = asig0.load()
                asig0 = _
            except Exception:
                pass
        n_samples = int(min(int(window_s * fs), len(asig0)))
        _ = asig0[:n_samples]
    except Exception as e:
        t_read1 = time.perf_counter()
        return {
            "T_open_s": t_open1 - t_open0,
            "T_read_window_ch0_s": t_read1 - t_read0,
            "fs": fs,
            "n_channels": int(n_channels),
            "n_samples_ch0": 0,
            "error": str(e),
        }
    t_read1 = time.perf_counter()

    return {
        "T_open_s": t_open1 - t_open0,
        "T_read_window_ch0_s": t_read1 - t_read0,
        "fs": fs,
        "n_channels": int(n_channels),
        "n_samples_ch0": int(n_samples),
    }


def bench_mne(path: pathlib.Path, profile: str, window_s: float, n_channels: int) -> dict:
    """
    MNE benchmark (EDF only).

    lazy:
      - read_raw_edf(preload=False)
      - open cost: read_raw_edf
      - read cost: raw.get_data() for `window_s` seconds

    preload:
      - read_raw_edf(preload=True)
      - open cost includes full preload
      - read cost: slice in RAM
    """
    if profile not in {"lazy", "preload"}:
        raise ValueError(f"Unknown profile for MNE: {profile}")

    try:
        import mne
    except Exception as e:
        return {"error": f"mne unavailable: {e}"}

    preload = (profile == "preload")

    t_open0 = time.perf_counter()
    raw = mne.io.read_raw_edf(str(path), preload=preload, verbose=False)
    t_open1 = time.perf_counter()

    fs = float(raw.info["sfreq"])
    n_samples = int(min(int(window_s * fs), raw.n_times))

    # Select first n_channels (if fewer, take available)
    picks = list(range(min(int(n_channels), raw.info["nchan"])))

    t_read0 = time.perf_counter()
    data = raw.get_data(picks=picks, start=0, stop=n_samples)
    t_read1 = time.perf_counter()

    return {
        "T_open_s": t_open1 - t_open0,
        "T_read_window_ch0_s": t_read1 - t_read0,
        "fs": fs,
        "n_channels": len(picks),
        "n_samples_ch0": int(n_samples),
    }


def parse_windows(spec: str) -> List[float]:
    out: List[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty --windows")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-files", type=int, default=5, help="How many EDF files to benchmark (sorted).")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--windows", type=str, default="60,600,1800")
    ap.add_argument("--n-channels", type=int, default=64)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="edf_io")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"Missing data dir: {data_dir}")

    edfs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".edf"])
    if not edfs:
        raise SystemExit(f"No EDF files found in {data_dir}")
    edfs = edfs[: int(args.n_files)]

    windows = parse_windows(args.windows)

    out = out_dir(args.out_root, BENCH_IO, "IO_STACKS", args.tag)
    write_manifest(out, BENCH_IO, "IO_STACKS", vars(args), extra={"format": FMT_EDF, "profiles": PROFILES, "libs": LIBS})

    rows = []
    for f_idx, edf_path in enumerate(edfs, 1):
        scenario = f"S{f_idx:02d}"
        for window_s in windows:
            for lib in LIBS:
                for profile in PROFILES:
                    for run in range(int(args.runs)):
                        bench_fn = {
                            "pyedflib": bench_pyedflib,
                            "neo": bench_neo,
                            "mne": bench_mne,
                        }[lib]
                        res = bench_fn(edf_path, profile, window_s, int(args.n_channels))
                        row = {
                            "kind": "raw",
                            "format": FMT_EDF,
                            "lib": lib,
                            "profile": profile,
                            "scenario": scenario,
                            "filename": edf_path.name,
                            "window_s": float(window_s),
                            "run": int(run),
                        }
                        # Map canonical + legacy field names
                        if "T_read_window_ch0_s" in res:
                            row["T_read_window_ch0_s"] = float(res["T_read_window_ch0_s"])
                            row["T_read_10s_ch0_s"] = float(res["T_read_window_ch0_s"])  # legacy alias
                        if "T_open_s" in res:
                            row["T_open_s"] = float(res["T_open_s"])
                        for k in ["fs", "n_channels", "n_samples_ch0"]:
                            if k in res:
                                row[k] = res[k]
                        if "error" in res:
                            row["error"] = res["error"]
                        rows.append(row)

    df = pd.DataFrame(rows)

    # Summaries (median + max)
    grp = df[df["kind"] == "raw"].groupby(["format", "lib", "profile", "scenario", "filename", "window_s"], dropna=False)
    summary = grp.agg(
        T_open_s_median=("T_open_s", "median"),
        T_open_s_max=("T_open_s", "max"),
        T_read_window_ch0_s_median=("T_read_window_ch0_s", "median"),
        T_read_window_ch0_s_max=("T_read_window_ch0_s", "max"),
    ).reset_index()
    summary.insert(0, "kind", "summary")

    # Write outputs
    out_raw = out / "io_raw.csv"
    out_sum = out / "io_summary.csv"
    df.to_csv(out_raw, index=False)
    summary.to_csv(out_sum, index=False)

    print(f"Saved: {out_raw} ({len(df)} rows)")
    print(f"Saved: {out_sum} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
