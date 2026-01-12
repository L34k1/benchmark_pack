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
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from benchkit.common import ensure_dir, env_info, out_dir
from benchkit.bench_defaults import (
    DEFAULT_STEPS,
    DEFAULT_TARGET_INTERVAL_MS,
    DEFAULT_WINDOW_S,
    PAN_STEP_FRACTION,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    default_load_duration_s,
)
from benchkit.lexicon import (
    BENCH_A2, FMT_EDF, FMT_NWB, OVL_OFF, OVL_ON,
    SEQ_PAN, SEQ_PAN_ZOOM, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, TOOL_PG
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.output_contract import write_manifest_contract, write_steps_csv, write_steps_summary


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


def _apply_pan(x0: float, x1: float, pan_step: float, lo: float, hi: float) -> Tuple[float, float, float]:
    next_x0, next_x1 = x0 + pan_step, x1 + pan_step
    if next_x0 < lo or next_x1 > hi:
        pan_step *= -1.0
        next_x0, next_x1 = x0 + pan_step, x1 + pan_step
    return next_x0, next_x1, pan_step


def build_ranges(sequence: str, lo: float, hi: float, window_s: float, steps: int) -> List[Tuple[float, float]]:
    x0, x1 = lo, min(lo + window_s, hi)
    w = x1 - x0
    pan_step = w * PAN_STEP_FRACTION
    rng: List[Tuple[float, float]] = []
    min_span = window_s * 0.10
    eps = 1e-9

    for i in range(steps):
        if sequence == SEQ_PAN:
            x0, x1, pan_step = _apply_pan(x0, x1, pan_step, lo, hi)

        elif sequence == SEQ_ZOOM_IN:
            cx = 0.5 * (x0 + x1)
            next_w = max(w * ZOOM_IN_FACTOR, min_span)
            if abs(next_w - w) <= eps * max(1.0, w):
                x0, x1, pan_step = _apply_pan(x0, x1, pan_step, lo, hi)
            else:
                w = next_w
                x0, x1 = cx - 0.5 * w, cx + 0.5 * w

        elif sequence == SEQ_ZOOM_OUT:
            cx = 0.5 * (x0 + x1)
            next_w = min(w * ZOOM_OUT_FACTOR, hi - lo)
            if abs(next_w - w) <= eps * max(1.0, w):
                x0, x1, pan_step = _apply_pan(x0, x1, pan_step, lo, hi)
            else:
                w = next_w
                x0, x1 = cx - 0.5 * w, cx + 0.5 * w

        elif sequence == SEQ_PAN_ZOOM:
            if i % 2 == 0:
                x0, x1, pan_step = _apply_pan(x0, x1, pan_step, lo, hi)
            else:
                cx = 0.5 * (x0 + x1)
                next_w = max(min(w * ZOOM_IN_FACTOR, hi - lo), min_span)
                if abs(next_w - w) <= eps * max(1.0, w):
                    x0, x1, pan_step = _apply_pan(x0, x1, pan_step, lo, hi)
                else:
                    w = next_w
                    x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        else:
            raise ValueError(sequence)

        x0, x1 = _clamp_range(x0, x1, lo, hi)
        rng.append((x0, x1))
    return rng


def _range_close(a: Tuple[float, float], b: Tuple[float, float], rel: float = 1e-6) -> bool:
    wa = abs(a[1] - a[0])
    wb = abs(b[1] - b[0])
    scale = max(wa, wb, 1.0)
    return abs(a[0] - b[0]) <= rel * scale and abs(a[1] - b[1]) <= rel * scale


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
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--target-interval-ms", type=float, default=DEFAULT_TARGET_INTERVAL_MS)

    p.add_argument("--n-ch", type=int, default=16)
    p.add_argument("--load-start-s", type=float, default=0.0)
    p.add_argument("--load-duration-s", type=float, default=None)
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--max-points-per-trace", type=int, default=5000)

    p.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)

    p.add_argument("--nwb-series-path", type=str, default=None)
    p.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    p.add_argument("--paint-timeout-ms", type=int, default=250)
    p.add_argument("--step-timeout-ms", type=int, default=5000)
    p.add_argument("--run-timeout-s", type=float, default=180.0)
    p.add_argument("--step-delay-ms", type=int, default=0)
    args = p.parse_args()
    # A2 runs are fixed at 200 steps for consistent cadence targets.
    if args.steps != DEFAULT_STEPS:
        _log(f"config=force_steps value={DEFAULT_STEPS} requested={args.steps}")
    args.steps = DEFAULT_STEPS
    if args.load_duration_s is None:
        args.load_duration_s = default_load_duration_s(args.window_s)

    if args.runs != 1:
        _log(f"config=force_runs value=1 requested={args.runs}")
        args.runs = 1
    out = out_dir(args.out_root, BENCH_A2, TOOL_PG, args.tag)
    ensure_dir(out)
    write_manifest_contract(
        out,
        bench_id=BENCH_A2,
        tool_id=TOOL_PG,
        fmt=args.format,
        file_path=args.file,
        window_s=float(args.window_s),
        n_channels=int(args.n_ch),
        sequence=args.sequence,
        overlay=args.overlay,
        run_id=0,
        steps_target=int(args.steps),
        extra={"env": env_info()},
    )

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
    view_box = plot.getViewBox()

    _log("phase=plot_init")
    offsets = np.arange(d.shape[0], dtype=np.float32)[:, None]
    y = d + offsets
    for ch in range(y.shape[0]):
        plot.plot(t, y[ch], pen=pg.mkPen(width=1))
    if args.overlay == OVL_ON:
        _add_overlay(plot, t)

    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    steps_rows: List[Dict[str, object]] = []
    prev_paints = plot.paint_count
    step_idx = 0
    step_start_ms = float("nan")
    step_deadline_ms = float("nan")
    base_start_ms = _now_ms()
    interval = float(args.target_interval_ms)
    expected_range: Tuple[float, float] = (float("nan"), float("nan"))
    range_updated = False
    failed = False
    exit_code = 0
    start_ms = _now_ms()

    def on_range_changed(_view_box, ranges) -> None:
        nonlocal range_updated
        if expected_range[0] != expected_range[0]:
            return
        x0, x1 = float(ranges[0][0]), float(ranges[0][1])
        if _range_close((x0, x1), expected_range):
            range_updated = True

    view_box.sigRangeChanged.connect(on_range_changed)

    def finalize() -> None:
        _log("phase=finalize_start")
        write_steps_csv(
            out,
            steps_rows,
            fieldnames=[
                "step_id",
                "status",
                "noop",
                "latency_ms",
                "lateness_ms",
                "xmin",
                "xmax",
                "span",
                "paint_delta",
                "range_signal",
            ],
        )
        lateness_ms = [row["lateness_ms"] for row in steps_rows if row["status"] == "OK"]
        arr = np.asarray(lateness_ms, dtype=float)
        if arr.size:
            lateness_p50_ms = float(np.percentile(arr, 50))
            lateness_p95_ms = float(np.percentile(arr, 95))
            lateness_max_ms = float(np.max(arr))
        else:
            lateness_p50_ms = float("nan")
            lateness_p95_ms = float("nan")
            lateness_max_ms = float("nan")
        write_steps_summary(
            out,
            steps_rows,
            extra={
                "bench_id": BENCH_A2,
                "tool_id": TOOL_PG,
                "format": args.format,
                "sequence": args.sequence,
                "overlay": args.overlay,
                "window_s": float(args.window_s),
                "steps": int(args.steps),
                "target_interval_ms": float(args.target_interval_ms),
                "lateness_p50_ms": lateness_p50_ms,
                "lateness_p95_ms": lateness_p95_ms,
                "lateness_max_ms": lateness_max_ms,
                "requested_n_ch": requested_n_ch,
                "available_n_ch": available_n_ch,
                "effective_n_ch": effective_n_ch,
                "fs_hz": float(meta.get("fs_hz", float("nan"))),
                "total_points_rendered": total_points,
                "meta": meta,
            },
        )
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
        if plot.paint_count > prev_paints or range_updated:
            end_paint = plot.last_paint_ms
            paint_delta = plot.paint_count - prev_paints
            prev_paints = plot.paint_count
            finish = float(end_paint) if np.isfinite(end_paint) else _now_ms()
            view_range = view_box.viewRange()[0]
            xmin, xmax = float(view_range[0]), float(view_range[1])
            span = float(xmax - xmin)
            steps_rows.append(
                {
                    "step_id": step_idx,
                    "status": "OK",
                    "noop": False,
                    "latency_ms": float(finish - step_start_ms),
                    "lateness_ms": float(finish - scheduled_ms),
                    "xmin": xmin,
                    "xmax": xmax,
                    "span": span,
                    "paint_delta": int(paint_delta),
                    "range_signal": bool(range_updated),
                }
            )
            _log(
                "phase=step_done idx=%d status=OK paint_delta=%d range_signal=%s"
                % (step_idx, paint_delta, bool(range_updated))
            )
            step_idx += 1
            schedule_next(float(args.step_delay_ms))
            return
        if _now_ms() >= step_deadline_ms:
            _log(
                "phase=step_timeout idx=%d paint_delta=%d range_signal=%s"
                % (step_idx, plot.paint_count - prev_paints, bool(range_updated))
            )
            fail_run(f"step_timeout step={step_idx} timeout_ms={args.step_timeout_ms}")
            return
        QtCore.QTimer.singleShot(1, lambda: check_paint(scheduled_ms))

    def run_step() -> None:
        nonlocal step_idx, step_start_ms, step_deadline_ms, expected_range, range_updated
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
        view_range = view_box.viewRange()[0]
        old_x0, old_x1 = float(view_range[0]), float(view_range[1])
        requested = ranges[step_idx]
        clamped = _clamp_range(requested[0], requested[1], lo, hi)
        old_span = float(old_x1 - old_x0)
        new_span = float(clamped[1] - clamped[0])
        eps = 1e-9 * max(1.0, old_span)
        noop = abs(clamped[0] - old_x0) <= eps and abs(clamped[1] - old_x1) <= eps
        clamped_reason = "normal"
        if not _range_close(requested, clamped):
            clamped_reason = "clamped"
        if noop:
            pan_step = old_span * PAN_STEP_FRACTION
            nudged_x0, nudged_x1, _ = _apply_pan(old_x0, old_x1, pan_step, lo, hi)
            if abs(nudged_x0 - old_x0) > eps or abs(nudged_x1 - old_x1) > eps:
                clamped = (nudged_x0, nudged_x1)
                new_span = float(clamped[1] - clamped[0])
                noop = False
                clamped_reason = "nudged"
            else:
                clamped_reason = "noop"
        _log(
            "phase=step idx=%d seq=%s old_x=(%.6f,%.6f) req_x=(%.6f,%.6f) clamp_x=(%.6f,%.6f) "
            "span_old=%.6f span_new=%.6f reason=%s"
            % (
                step_idx,
                args.sequence,
                old_x0,
                old_x1,
                requested[0],
                requested[1],
                clamped[0],
                clamped[1],
                old_span,
                new_span,
                clamped_reason,
            )
        )
        range_updated = False
        expected_range = clamped
        step_start_ms = _now_ms()
        step_deadline_ms = step_start_ms + float(args.step_timeout_ms)
        if noop:
            finish = _now_ms()
            steps_rows.append(
                {
                    "step_id": step_idx,
                    "status": "NOOP",
                    "noop": True,
                    "latency_ms": 0.0,
                    "lateness_ms": float(finish - scheduled_ms),
                    "xmin": float(clamped[0]),
                    "xmax": float(clamped[1]),
                    "span": float(new_span),
                    "paint_delta": 0,
                    "range_signal": False,
                }
            )
            _log("phase=step_done idx=%d status=NOOP paint_delta=0 range_signal=False" % step_idx)
            step_idx += 1
            schedule_next(float(args.step_delay_ms))
            return
        plot.setXRange(clamped[0], clamped[1], padding=0.0)
        plot.repaint()
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
