from __future__ import annotations

import sys

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from benchkit.common import ensure_dir, env_info, out_dir, write_json, write_manifest
from benchkit.lexicon import (
    BENCH_A2, FMT_EDF, FMT_NWB, OVL_OFF, OVL_ON,
    SEQ_PAN, SEQ_PAN_ZOOM, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, TOOL_PG
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.stats import summarize_latency_ms


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class _Plot(pg.PlotWidget):
    def __init__(self) -> None:
        super().__init__()
        self.paint_count = 0
        self.last_paint_ms = float("nan")

    def paintEvent(self, ev) -> None:  # type: ignore[override]
        self.paint_count += 1
        self.last_paint_ms = _now_ms()
        return super().paintEvent(ev)


def _load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if args.format == FMT_EDF:
        seg = load_edf_segment_pyedflib(args.file, args.load_start_s, args.load_duration_s, args.n_ch)
    else:
        seg = load_nwb_segment_pynwb(
            args.file, args.load_start_s, args.load_duration_s, args.n_ch,
            series_path=args.nwb_series_path, time_dim=args.nwb_time_dim
        )
    t, d, dec = decimate_for_display(seg.times_s, seg.data, args.max_points_per_trace)
    meta = dict(seg.meta)
    meta["decim_factor"] = int(dec)
    meta["n_points_per_trace"] = int(t.shape[0])
    meta["fs_hz"] = float(seg.fs_hz)
    return t, d, meta


def _add_overlay(plot: pg.PlotWidget, t: np.ndarray) -> None:
    for frac in (0.25, 0.5, 0.75):
        x = float(t[int(frac * (len(t) - 1))])
        plot.addItem(pg.InfiniteLine(pos=x, angle=90, movable=False))


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


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="PyQtGraph A2 (cadenced) benchmark.")
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
    p.add_argument("--paint-timeout-ms", type=int, default=250)
    p.add_argument("--step-timeout-ms", type=int, default=5000)
    p.add_argument("--run-timeout-s", type=float, default=180.0)
    p.add_argument("--step-delay-ms", type=int, default=0)
    args = p.parse_args()

    out = out_dir(args.out_root, BENCH_A2, TOOL_PG, args.tag)
    ensure_dir(out)
    write_manifest(out, BENCH_A2, TOOL_PG, args=vars(args), extra={"env": env_info()})

    _log("phase=load_start")
    t, d, meta = _load_segment(args)
    _log("phase=load_done")

    requested_n_ch = int(args.n_ch)
    available_n_ch = int(meta.get("n_ch_total", d.shape[0]))
    effective_n_ch = int(meta.get("effective_n_ch", meta.get("n_ch_used", d.shape[0])))
    n_points_per_trace = int(meta.get("n_points_per_trace", t.shape[0]))
    total_points = int(n_points_per_trace * effective_n_ch)
    meta.update(
        {
            "requested_n_ch": requested_n_ch,
            "available_n_ch": available_n_ch,
            "effective_n_ch": effective_n_ch,
            "sequence": args.sequence,
            "window_s": float(args.window_s),
            "fs_hz": float(meta.get("fs_hz", float("nan"))),
            "total_points_rendered": total_points,
        }
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = _Plot()
    plot.setWindowTitle("PyQtGraph A2")
    plot.setBackground("w")
    plot.show()

    _log("phase=plot_init")
    offsets = np.arange(d.shape[0], dtype=np.float32)[:, None]
    y = d + offsets
    for ch in range(y.shape[0]):
        plot.plot(t, y[ch], pen=pg.mkPen(width=1))
    if args.overlay == OVL_ON:
        _add_overlay(plot, t)

    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    lat_ms: List[float] = []
    lateness_ms: List[float] = []
    prev_paints = plot.paint_count
    step_idx = 0
    step_start_ms = float("nan")
    step_deadline_ms = float("nan")
    base_start_ms = _now_ms()
    interval = float(args.target_interval_ms)
    failed = False
    exit_code = 0
    start_ms = _now_ms()

    def finalize() -> None:
        _log("phase=finalize_start")
        steps_csv = out / "steps.csv"
        with steps_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step_id", "latency_ms", "lateness_ms"])
            for i, (lat, late) in enumerate(zip(lat_ms, lateness_ms)):
                w.writerow([i, f"{lat:.3f}", f"{late:.3f}"])

        arr = np.asarray(lateness_ms, dtype=float)
        summary = summarize_latency_ms(lat_ms)
        summary.update({
            "bench_id": BENCH_A2,
            "tool_id": TOOL_PG,
            "format": args.format,
            "sequence": args.sequence,
            "overlay": args.overlay,
            "window_s": float(args.window_s),
            "steps": int(args.steps),
            "target_interval_ms": float(args.target_interval_ms),
            "lateness_p50_ms": float(np.percentile(arr, 50)),
            "lateness_p95_ms": float(np.percentile(arr, 95)),
            "lateness_max_ms": float(np.max(arr)),
            "requested_n_ch": requested_n_ch,
            "available_n_ch": available_n_ch,
            "effective_n_ch": effective_n_ch,
            "fs_hz": float(meta.get("fs_hz", float("nan"))),
            "total_points_rendered": total_points,
            "meta": meta,
        })
        write_json(out / "summary.json", summary)
        write_json(out / "latencies_ms.json", {"lat_ms": lat_ms, "lateness_ms": lateness_ms})
        _log("phase=finalize_done")

    def schedule_next(delay_ms: float = 0.0) -> None:
        QtCore.QTimer.singleShot(int(max(0, delay_ms)), run_step)

    def fail_run(reason: str) -> None:
        nonlocal failed, exit_code
        if failed:
            return
        failed = True
        exit_code = 2
        _log(f"error={reason}")
        app.quit()

    def check_paint(scheduled_ms: float) -> None:
        nonlocal step_idx, prev_paints
        if failed:
            return
        if plot.paint_count > prev_paints:
            end_paint = plot.last_paint_ms
            prev_paints = plot.paint_count
            finish = float(end_paint) if np.isfinite(end_paint) else _now_ms()
            lat_ms.append(float(finish - step_start_ms))
            lateness_ms.append(float(finish - scheduled_ms))
            step_idx += 1
            schedule_next(float(args.step_delay_ms))
            return
        if _now_ms() >= step_deadline_ms:
            fail_run(f"step_timeout step={step_idx} timeout_ms={args.step_timeout_ms}")
            return
        QtCore.QTimer.singleShot(1, lambda: check_paint(scheduled_ms))

    def run_step() -> None:
        nonlocal step_idx, step_start_ms, step_deadline_ms
        if failed:
            return
        if step_idx >= len(ranges):
            finalize()
            app.quit()
            return
        scheduled_ms = base_start_ms + step_idx * interval
        now_ms = _now_ms()
        if now_ms < scheduled_ms:
            schedule_next(scheduled_ms - now_ms)
            return
        x0, x1 = ranges[step_idx]
        _log(f"phase=step idx={step_idx} x0={x0:.6f} x1={x1:.6f}")
        step_start_ms = _now_ms()
        step_deadline_ms = step_start_ms + float(args.step_timeout_ms)
        plot.setXRange(x0, x1, padding=0.0)
        check_paint(scheduled_ms)

    def heartbeat() -> None:
        if failed:
            return
        elapsed_s = (_now_ms() - start_ms) / 1000.0
        _log(f"heartbeat elapsed_s={elapsed_s:.1f} step={step_idx}/{len(ranges)}")
        QtCore.QTimer.singleShot(1000, heartbeat)

    def run_watchdog() -> None:
        if failed:
            return
        if float(args.run_timeout_s) > 0 and (_now_ms() - start_ms) > float(args.run_timeout_s) * 1000.0:
            fail_run(f"run_timeout timeout_s={args.run_timeout_s}")
            return
        QtCore.QTimer.singleShot(200, run_watchdog)

    _log("phase=run_start")
    heartbeat()
    run_watchdog()
    schedule_next(0.0)
    app.exec_()
    if failed:
        raise SystemExit(exit_code)
    return


if __name__ == "__main__":
    main()
