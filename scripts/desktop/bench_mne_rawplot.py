from __future__ import annotations

import sys

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from benchkit.common import ensure_dir, env_info, out_dir, write_json, write_manifest
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    BENCH_TFFR,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_MNE_RAWPLOT,
)
from benchkit.stats import summarize_latency_ms


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _load_raw(args: argparse.Namespace):
    try:
        import mne
    except Exception as exc:
        raise RuntimeError(f"mne unavailable: {exc}") from exc

    if args.format == FMT_EDF:
        raw = mne.io.read_raw_edf(str(args.file), preload=False, verbose="ERROR")
    elif args.format == FMT_NWB:
        raw = mne.io.read_raw_nwb(str(args.file), preload=False, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    if args.load_duration_s > 0:
        tmin = float(args.load_start_s)
        tmax = tmin + float(args.load_duration_s)
        raw.crop(tmin=tmin, tmax=tmax)

    return raw


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


def _resolve_fig(plotter: Any):
    if hasattr(plotter, "fig"):
        return plotter.fig
    if hasattr(plotter, "_fig"):
        return plotter._fig
    raise RuntimeError("Unable to resolve RawPlotter figure handle")


def _set_plot_time(plotter: Any, time_s: float) -> None:
    if hasattr(plotter, "set_time"):
        plotter.set_time(time_s)
        return
    if hasattr(plotter, "_set_time"):
        plotter._set_time(time_s)
        return
    if hasattr(plotter, "_update_time"):
        plotter._update_time(time_s)
        return
    if hasattr(plotter, "first_time"):
        plotter.first_time = float(time_s)
        if hasattr(plotter, "_update_times"):
            plotter._update_times()
        if hasattr(plotter, "_update_data"):
            plotter._update_data()
        return
    raise RuntimeError("RawPlotter time update API not found")


def wait_next_draw(draw_times: List[float], prev_len: int, timeout_ms: int = 2000) -> float:
    t0 = _now_ms()
    while len(draw_times) <= prev_len and (_now_ms() - t0) < timeout_ms:
        plt.pause(0.001)
    return draw_times[-1] if len(draw_times) > prev_len else float("nan")


def run_tffr(raw, args: argparse.Namespace) -> Dict[str, Any]:
    picks = list(range(min(int(args.n_ch), raw.info["nchan"])))

    draw_times: List[float] = []
    start_ms = _now_ms()
    plotter = raw.plot(
        show=True,
        block=False,
        duration=float(args.window_s),
        n_channels=int(args.n_ch),
        picks=picks,
        scalings="auto",
    )
    fig = _resolve_fig(plotter)
    fig.canvas.mpl_connect("draw_event", lambda event: draw_times.append(_now_ms()))
    fig.canvas.draw_idle()

    end_ms = wait_next_draw(draw_times, 0, timeout_ms=5000)
    tffr_ms = float(end_ms - start_ms) if np.isfinite(end_ms) else float("nan")

    if args.pause_ms > 0:
        plt.pause(args.pause_ms / 1000.0)
    plt.close(fig)

    return {
        "bench_id": BENCH_TFFR,
        "tool_id": TOOL_MNE_RAWPLOT,
        "format": args.format,
        "overlay": args.overlay,
        "window_s": float(args.window_s),
        "n_ch": int(len(picks)),
        "tffr_ms": tffr_ms,
    }


def run_a1(raw, args: argparse.Namespace) -> Tuple[Dict[str, Any], List[float]]:
    picks = list(range(min(int(args.n_ch), raw.info["nchan"])))

    draw_times: List[float] = []
    plotter = raw.plot(
        show=True,
        block=False,
        duration=float(args.window_s),
        n_channels=int(args.n_ch),
        picks=picks,
        scalings="auto",
    )
    fig = _resolve_fig(plotter)
    fig.canvas.mpl_connect("draw_event", lambda event: draw_times.append(_now_ms()))
    fig.canvas.draw_idle()

    wait_next_draw(draw_times, 0, timeout_ms=5000)

    lo = float(raw.times[0])
    hi = float(raw.times[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    lat_ms: List[float] = []

    for x0, _x1 in ranges:
        start = _now_ms()
        prev_len = len(draw_times)
        _set_plot_time(plotter, float(x0))
        fig.canvas.draw_idle()
        end_paint = wait_next_draw(draw_times, prev_len)
        lat_ms.append(float(end_paint - start) if np.isfinite(end_paint) else float("nan"))

    plt.close(fig)

    summary = summarize_latency_ms(lat_ms)
    summary.update({
        "bench_id": BENCH_A1,
        "tool_id": TOOL_MNE_RAWPLOT,
        "format": args.format,
        "sequence": args.sequence,
        "overlay": args.overlay,
        "window_s": float(args.window_s),
        "steps": int(args.steps),
        "n_ch": int(len(picks)),
    })
    return summary, lat_ms


def run_a2(raw, args: argparse.Namespace) -> Tuple[Dict[str, Any], List[float], List[float]]:
    picks = list(range(min(int(args.n_ch), raw.info["nchan"])))

    draw_times: List[float] = []
    plotter = raw.plot(
        show=True,
        block=False,
        duration=float(args.window_s),
        n_channels=int(args.n_ch),
        picks=picks,
        scalings="auto",
    )
    fig = _resolve_fig(plotter)
    fig.canvas.mpl_connect("draw_event", lambda event: draw_times.append(_now_ms()))
    fig.canvas.draw_idle()

    wait_next_draw(draw_times, 0, timeout_ms=5000)

    lo = float(raw.times[0])
    hi = float(raw.times[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    lat_ms: List[float] = []
    lateness_ms: List[float] = []

    t0 = _now_ms()
    interval = float(args.target_interval_ms)

    for i, (x0, _x1) in enumerate(ranges):
        scheduled = t0 + i * interval

        while True:
            now = _now_ms()
            if now >= scheduled:
                break
            remaining = scheduled - now
            if remaining > 2.0:
                time.sleep((remaining - 1.0) / 1000.0)
            plt.pause(0.001)

        start = _now_ms()
        prev_len = len(draw_times)
        _set_plot_time(plotter, float(x0))
        fig.canvas.draw_idle()
        end_paint = wait_next_draw(draw_times, prev_len)

        finish = float(end_paint) if np.isfinite(end_paint) else _now_ms()
        lat_ms.append(float(finish - start))
        lateness_ms.append(float(finish - scheduled))

    plt.close(fig)

    arr = np.asarray(lateness_ms, dtype=float)
    summary = summarize_latency_ms(lat_ms)
    summary.update({
        "bench_id": BENCH_A2,
        "tool_id": TOOL_MNE_RAWPLOT,
        "format": args.format,
        "sequence": args.sequence,
        "overlay": args.overlay,
        "window_s": float(args.window_s),
        "steps": int(args.steps),
        "target_interval_ms": float(args.target_interval_ms),
        "lateness_p50_ms": float(np.percentile(arr, 50)),
        "lateness_p95_ms": float(np.percentile(arr, 95)),
        "lateness_max_ms": float(np.max(arr)),
        "n_ch": int(len(picks)),
    })
    return summary, lat_ms, lateness_ms


def main() -> None:
    p = argparse.ArgumentParser(description="MNE RawPlot benchmarking (EDF/NWB).")
    p.add_argument("--bench", choices=[BENCH_TFFR, BENCH_A1, BENCH_A2], required=True)
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

    p.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)

    p.add_argument("--pause-ms", type=int, default=250)
    args = p.parse_args()

    out = out_dir(args.out_root, args.bench, TOOL_MNE_RAWPLOT, args.tag)
    ensure_dir(out)
    write_manifest(out, args.bench, TOOL_MNE_RAWPLOT, args=vars(args), extra={"env": env_info()})

    raw = _load_raw(args)

    if args.bench == BENCH_TFFR:
        summary = run_tffr(raw, args)
        write_json(out / "summary.json", summary)
    elif args.bench == BENCH_A1:
        summary, lat_ms = run_a1(raw, args)
        write_json(out / "summary.json", summary)
        write_json(out / "latencies_ms.json", {"lat_ms": lat_ms})
    elif args.bench == BENCH_A2:
        summary, lat_ms, lateness_ms = run_a2(raw, args)
        write_json(out / "summary.json", summary)
        write_json(out / "latencies_ms.json", {"lat_ms": lat_ms, "lateness_ms": lateness_ms})
    else:
        raise ValueError(args.bench)


if __name__ == "__main__":
    main()
