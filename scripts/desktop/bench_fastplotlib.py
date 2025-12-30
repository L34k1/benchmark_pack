from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import fastplotlib as fpl

from benchkit.common import ensure_dir, env_info, out_dir, write_json, write_manifest
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_FASTPLOTLIB,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.stats import summarize_latency_ms


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
    t, d, dec = decimate_for_display(seg.times_s, seg.data, args.max_points_per_trace)
    meta = dict(seg.meta)
    meta["decim_factor"] = int(dec)
    meta["n_points_per_trace"] = int(t.shape[0])
    return t.astype(np.float32), d.astype(np.float32), meta


def _clamp_range(x0: float, x1: float, lo: float, hi: float) -> Tuple[float, float]:
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

        x0, x1 = _clamp_range(x0, x1, lo, hi)
        rng.append((x0, x1))
    return rng


def _set_axes_range(plot: fpl.Plot, x0: float, x1: float, y0: float, y1: float) -> None:
    axes = plot.axes
    if hasattr(axes, "set_range"):
        axes.set_range(x=(x0, x1), y=(y0, y1))
        return
    if hasattr(axes, "set_ranges"):
        axes.set_ranges(x=(x0, x1), y=(y0, y1))
        return
    if hasattr(axes, "x") and hasattr(axes, "y"):
        if hasattr(axes.x, "lim"):
            axes.x.lim = (x0, x1)
        elif hasattr(axes.x, "range"):
            axes.x.range = (x0, x1)
        if hasattr(axes.y, "lim"):
            axes.y.lim = (y0, y1)
        elif hasattr(axes.y, "range"):
            axes.y.range = (y0, y1)


def _render(plot: fpl.Plot) -> None:
    if hasattr(plot, "render"):
        plot.render()
        return
    if hasattr(plot, "canvas"):
        canvas = plot.canvas
        if hasattr(canvas, "request_draw"):
            canvas.request_draw()
        if hasattr(canvas, "draw"):
            canvas.draw()
            return
        if hasattr(canvas, "update"):
            canvas.update()
            return


def _add_overlay(plot: fpl.Plot, t: np.ndarray, y0: float, y1: float) -> None:
    if not hasattr(plot, "add_line"):
        return
    t0 = float(t[0])
    t1 = float(t[-1])
    for frac in (0.25, 0.5, 0.75):
        x = t0 + (t1 - t0) * frac
        pos = np.array([[x, y0], [x, y1]], dtype=np.float32)
        plot.add_line(pos, color="gray")


def _bench_a1(args: argparse.Namespace, plot: fpl.Plot, t: np.ndarray, meta: Dict[str, Any]) -> None:
    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    lat_ms: List[float] = []
    y0 = -1.0
    y1 = float(args.n_ch) + 1.0

    for x0, x1 in ranges:
        start = _now_ms()
        _set_axes_range(plot, x0, x1, y0, y1)
        _render(plot)
        end = _now_ms()
        lat_ms.append(float(end - start))

    summary = summarize_latency_ms(lat_ms)
    summary.update({
        "bench_id": BENCH_A1,
        "tool_id": TOOL_FASTPLOTLIB,
        "format": args.format,
        "sequence": args.sequence,
        "overlay": args.overlay,
        "window_s": float(args.window_s),
        "steps": int(args.steps),
        "meta": meta,
    })
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "latencies_ms.json", {"lat_ms": lat_ms})


def _bench_a2(args: argparse.Namespace, plot: fpl.Plot, t: np.ndarray, meta: Dict[str, Any]) -> None:
    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    lat_ms: List[float] = []
    lateness_ms: List[float] = []

    t0 = _now_ms()
    interval = float(args.target_interval_ms)
    y0 = -1.0
    y1 = float(args.n_ch) + 1.0

    for i, (x0, x1) in enumerate(ranges):
        scheduled = t0 + i * interval

        while True:
            now = _now_ms()
            if now >= scheduled:
                break
            remaining = scheduled - now
            if remaining > 2.0:
                time.sleep((remaining - 1.0) / 1000.0)

        start = _now_ms()
        _set_axes_range(plot, x0, x1, y0, y1)
        _render(plot)
        finish = _now_ms()
        lat_ms.append(float(finish - start))
        lateness_ms.append(float(finish - scheduled))

    arr = np.asarray(lateness_ms, dtype=float)
    summary = summarize_latency_ms(lat_ms)
    summary.update({
        "bench_id": BENCH_A2,
        "tool_id": TOOL_FASTPLOTLIB,
        "format": args.format,
        "sequence": args.sequence,
        "overlay": args.overlay,
        "window_s": float(args.window_s),
        "steps": int(args.steps),
        "target_interval_ms": float(args.target_interval_ms),
        "lateness_p50_ms": float(np.percentile(arr, 50)),
        "lateness_p95_ms": float(np.percentile(arr, 95)),
        "lateness_max_ms": float(np.max(arr)),
        "meta": meta,
    })
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "latencies_ms.json", {"lat_ms": lat_ms, "lateness_ms": lateness_ms})


def main() -> None:
    p = argparse.ArgumentParser(description="fastplotlib A1/A2 benchmark for EEG line stacks.")
    p.add_argument("--bench", choices=[BENCH_A1, BENCH_A2], required=True)
    p.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--tag", type=str, required=True)

    p.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN_ZOOM)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--target-interval-ms", type=float, default=16.0)

    p.add_argument("--n-ch", type=int, default=16)
    p.add_argument("--load-start-s", type=float, default=0.0)
    p.add_argument("--load-duration-s", type=float, default=1900.0)
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--max-points-per-trace", type=int, default=5000)

    p.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)

    p.add_argument("--nwb-series-path", type=str, default=None)
    p.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    args = p.parse_args()

    out = out_dir(args.out_root, args.bench, TOOL_FASTPLOTLIB, args.tag)
    ensure_dir(out)
    args.out_dir = out
    write_manifest(out, args.bench, TOOL_FASTPLOTLIB, args=vars(args), extra={"env": env_info()})

    t, d, meta = _load_segment(args)

    plot = fpl.Plot()
    plot.add_line_stack(d, x=t, spacing=1.0)
    plot.show()

    y0 = -1.0
    y1 = float(args.n_ch) + 1.0
    x0 = float(t[0])
    x1 = min(float(t[-1]), x0 + float(args.window_s))
    _set_axes_range(plot, x0, x1, y0, y1)
    _render(plot)

    if args.overlay == OVL_ON:
        _add_overlay(plot, t, y0, y1)
        _render(plot)

    if args.bench == BENCH_A1:
        _bench_a1(args, plot, t, meta)
    else:
        _bench_a2(args, plot, t, meta)

    if hasattr(plot, "close"):
        plot.close()


if __name__ == "__main__":
    main()
