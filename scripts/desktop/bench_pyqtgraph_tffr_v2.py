from __future__ import annotations

import sys

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from benchkit.common import ensure_dir, env_info, out_dir
from benchkit.bench_defaults import DEFAULT_WINDOW_S, default_load_duration_s
from benchkit.lexicon import (
    BENCH_TFFR,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    TOOL_PG,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.output_contract import write_manifest_contract, write_tffr_csv, write_tffr_summary


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


def _load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    if args.format == FMT_EDF:
        seg = load_edf_segment_pyedflib(args.file, args.load_start_s, args.load_duration_s, args.n_ch)
    elif args.format == FMT_NWB:
        seg = load_nwb_segment_pynwb(
            args.file, args.load_start_s, args.load_duration_s, args.n_ch,
            series_path=args.nwb_series_path, time_dim=args.nwb_time_dim
        )
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    t, d, dec = decimate_for_display(seg.times_s, seg.data, args.max_points_per_trace)
    meta = dict(seg.meta)
    meta["decim_factor"] = int(dec)
    meta["n_points_per_trace"] = int(t.shape[0])
    return t, d, float(seg.fs_hz), meta


def _add_overlay(plot: pg.PlotWidget, t: np.ndarray) -> None:
    for frac in (0.25, 0.5, 0.75):
        x = float(t[int(frac * (len(t) - 1))])
        plot.addItem(pg.InfiniteLine(pos=x, angle=90, movable=False))


def main() -> None:
    p = argparse.ArgumentParser(description="PyQtGraph Time-to-First-Render benchmark (EDF/NWB).")
    p.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    p.add_argument("--file", type=Path, required=True)

    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--tag", type=str, required=True)

    p.add_argument("--n-ch", type=int, default=16)
    p.add_argument("--load-start-s", type=float, default=0.0)
    p.add_argument("--load-duration-s", type=float, default=None)
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--max-points-per-trace", type=int, default=5000)
    p.add_argument("--sequence", choices=[SEQ_PAN], default=SEQ_PAN)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--runs", type=int, default=1)

    p.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)

    p.add_argument("--nwb-series-path", type=str, default=None)
    p.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])

    p.add_argument("--pause-ms", type=int, default=250)
    args = p.parse_args()
    if args.load_duration_s is None:
        args.load_duration_s = default_load_duration_s(args.window_s)

    if args.runs != 1:
        print(f"[WARN] forcing runs=1 (requested {args.runs})")
        args.runs = 1
    out = out_dir(args.out_root, BENCH_TFFR, TOOL_PG, args.tag)
    ensure_dir(out)
    write_manifest_contract(
        out,
        bench_id=BENCH_TFFR,
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

    t, d, fs, meta = _load_segment(args)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = _Plot()
    plot.setWindowTitle("PyQtGraph TFFR")
    plot.setBackground("w")
    plot.show()

    base_paints = plot.paint_count
    t0 = _now_ms()
    while plot.paint_count == base_paints and (_now_ms() - t0) < 2000:
        app.processEvents(QtCore.QEventLoop.AllEvents, 1)

    offsets = np.arange(d.shape[0], dtype=np.float32)[:, None]
    y = d + offsets

    base_paints = plot.paint_count
    start_ms = _now_ms()
    for ch in range(y.shape[0]):
        plot.plot(t, y[ch], pen=pg.mkPen(width=1))
    if args.overlay == OVL_ON:
        _add_overlay(plot, t)

    while plot.paint_count == base_paints and (_now_ms() - start_ms) < 5000:
        app.processEvents(QtCore.QEventLoop.AllEvents, 1)

    end_ms = plot.last_paint_ms
    tffr_ms = float(end_ms - start_ms) if np.isfinite(end_ms) else float("nan")

    write_tffr_csv(out, run_id=0, tffr_ms=tffr_ms)
    write_tffr_summary(
        out,
        bench_id=BENCH_TFFR,
        tool_id=TOOL_PG,
        fmt=args.format,
        window_s=float(args.window_s),
        n_channels=int(args.n_ch),
        tffr_ms=tffr_ms,
        extra={"overlay": args.overlay, "meta": meta},
    )

    if args.pause_ms > 0:
        QtCore.QTimer.singleShot(int(args.pause_ms), app.quit)
        app.exec_()


if __name__ == "__main__":
    main()
