#!/usr/bin/env python3
"""scripts/desktop/bench_datoviz.py

Datoviz benchmark entrypoint (EDF/NWB).
Implements TFFR/A1/A2 using line visuals and axis updates when available.
"""

from __future__ import annotations

import sys

import argparse
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_json, write_manifest
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    BENCH_TFFR,
    FMT_EDF,
    FMT_NWB,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_DATOVIZ,
)
from benchkit.loaders import load_edf_segment_pyedflib, load_nwb_segment_pynwb


def clamp_range(x0: float, x1: float, lo: float, hi: float) -> Tuple[float, float]:
    w = x1 - x0
    if w <= 0:
        w = 1e-6
    if x0 < lo:
        x0, x1 = lo, lo + w
    if x1 > hi:
        x1, x0 = hi, hi - w
    if x0 < lo:
        x0, x1 = lo, hi
    return x0, x1


def build_ranges(sequence: str, lo: float, hi: float, window_s: float, steps: int) -> List[Tuple[float, float]]:
    x0, x1 = lo, min(lo + window_s, hi)
    w = x1 - x0
    pan_step = w * 0.10
    rng: List[Tuple[float, float]] = []

    for i in range(steps):
        if sequence == SEQ_PAN:
            x0, x1 = x0 + pan_step, x1 + pan_step
            if x1 > hi or x0 < lo:
                pan_step *= -1.0
        elif sequence == SEQ_ZOOM_IN:
            cx = 0.5 * (x0 + x1)
            w = max(w * 0.90, window_s * 0.10)
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif sequence == SEQ_ZOOM_OUT:
            cx = 0.5 * (x0 + x1)
            w = min(w * 1.10, hi - lo)
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif sequence == SEQ_PAN_ZOOM:
            if i % 2 == 0:
                x0, x1 = x0 + pan_step, x1 + pan_step
                if x1 > hi or x0 < lo:
                    pan_step *= -1.0
            else:
                cx = 0.5 * (x0 + x1)
                w = max(min(w * 0.95, hi - lo), window_s * 0.10)
                x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        else:
            raise ValueError(sequence)

        x0, x1 = clamp_range(x0, x1, lo, hi)
        rng.append((x0, x1))
    return rng


def load_segment(args: argparse.Namespace) -> Tuple[List[float], List[List[float]]]:
    if args.format == FMT_EDF:
        seg = load_edf_segment_pyedflib(args.file, args.load_start_s, args.load_duration_s, args.n_ch)
    else:
        seg = load_nwb_segment_pynwb(
            args.file,
            args.load_start_s,
            args.load_duration_s,
            args.n_ch,
            series_path=args.nwb_series_path,
            time_dim=args.nwb_time_dim,
        )
    t = seg.times_s.astype(float).tolist()
    data = seg.data.astype(float).tolist()
    return t, data


def main() -> None:
    ap = argparse.ArgumentParser(description="Datoviz benchmark.")
    ap.add_argument("--bench-id", choices=[BENCH_TFFR, BENCH_A1, BENCH_A2], required=True)
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--target-interval-ms", type=float, default=16.0)
    ap.add_argument("--n-ch", type=int, default=16)
    ap.add_argument("--load-start-s", type=float, default=0.0)
    ap.add_argument("--load-duration-s", type=float, default=300.0)
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    args = ap.parse_args()

    out = out_dir(args.out_root, args.bench_id, TOOL_DATOVIZ, args.tag)
    write_manifest(out, args.bench_id, TOOL_DATOVIZ, args=vars(args), extra={"format": args.format})

    import datoviz as dv

    t, data = load_segment(args)
    canvas = dv.canvas()
    for ch in range(min(len(data), args.n_ch)):
        xy = dv.array([t, data[ch]]).T if hasattr(dv, "array") else list(zip(t, data[ch]))
        canvas.visual("line", xy)
    t0 = time.perf_counter()
    canvas.show()
    tffr_s = time.perf_counter() - t0

    if args.bench_id == BENCH_TFFR:
        write_json(out / "summary.json", {"bench_id": BENCH_TFFR, "tool_id": TOOL_DATOVIZ, "tffr_s": tffr_s})
        return

    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))
    lat_ms: List[float] = []
    interval_s = float(args.target_interval_ms) / 1000.0
    start = time.perf_counter()
    for idx, (x0, x1) in enumerate(ranges):
        if args.bench_id == BENCH_A2:
            target = start + idx * interval_s
            while time.perf_counter() < target:
                pass
        t_issue = time.perf_counter()
        if hasattr(canvas, "set_xrange"):
            canvas.set_xrange(x0, x1)
        t_paint = time.perf_counter()
        lat_ms.append((t_paint - t_issue) * 1000.0)

    write_json(
        out / "summary.json",
        {
            "bench_id": args.bench_id,
            "tool_id": TOOL_DATOVIZ,
            "sequence": args.sequence,
            "steps": int(args.steps),
            "latency_ms_mean": float(sum(lat_ms) / max(1, len(lat_ms))),
        },
    )


if __name__ == "__main__":
    main()
