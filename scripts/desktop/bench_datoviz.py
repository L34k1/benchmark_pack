#!/usr/bin/env python3
"""scripts/desktop/bench_datoviz.py

Datoviz desktop benchmarks:
- TFFR: time-to-first-render (first draw event)
- A1_THROUGHPUT: as-fast-as-possible interactions (issue next command on paint)
- A2_CADENCED: cadenced interactions (issue commands on a fixed interval)

Outputs (per bench):
  outputs/<BENCH_ID>/VIS_DATOVIZ/<tag>/
    - manifest.json
    - datoviz_<bench>_steps.csv
    - datoviz_<bench>_summary.csv   (interaction benches)
    - datoviz_tffr.csv              (TFFR bench)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_manifest
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    BENCH_TFFR,
    CACHE_WARM,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_DATOVIZ,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb


@dataclass
class Cmd:
    step_idx: int
    x0: float
    x1: float
    t_sched_s: float
    t_cmd_s: float


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


class DatovizBench:
    """
    A Datoviz scene that supports:
    - first draw timestamp capture (TFFR)
    - panel range updates
    - draw-event callback for measuring command->draw latency
    """

    def __init__(self, t: np.ndarray, data: np.ndarray, width: int, height: int):
        import datoviz as dv

        self.dv = dv
        self.canvas = dv.canvas(show=False, size=(width, height))
        self.panel = self.canvas.panel()

        self._n_ch = int(data.shape[0])
        self._lane_h = 1.0 / max(1, self._n_ch)
        self._amp = 0.35 * self._lane_h
        self._line_visuals: List[Any] = []

        for i in range(self._n_ch):
            y0 = (i + 0.5) * self._lane_h
            y = y0 + self._amp * data[i]
            pos = np.column_stack([t, y, np.zeros_like(t)]).astype(np.float32, copy=False)
            line = self.panel.visual("line")
            line.data("pos", pos)
            line.data("color", np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32))
            self._line_visuals.append(line)

        self._first_draw_s: Optional[float] = None
        self._t0_s: Optional[float] = None
        self._draw_cb: Optional[callable] = None

        @self.canvas.connect
        def _on_event(ev) -> None:
            if getattr(ev, "type", "draw") != "draw":
                return
            if self._first_draw_s is None:
                self._first_draw_s = time.perf_counter()
            if self._draw_cb is not None:
                self._draw_cb()

    def show_and_start(self) -> None:
        self.canvas.show()
        self._t0_s = time.perf_counter()

    @property
    def tffr_s(self) -> Optional[float]:
        if self._t0_s is None or self._first_draw_s is None:
            return None
        return self._first_draw_s - self._t0_s

    def _set_panel_range(self, x0: float, x1: float, y0: float, y1: float) -> None:
        if hasattr(self.panel, "set_range"):
            self.panel.set_range(x=(float(x0), float(x1)), y=(float(y0), float(y1)))
            return
        camera = getattr(self.panel, "camera", None)
        if callable(camera):
            camera = camera()
        if camera is not None and hasattr(camera, "set_range"):
            camera.set_range(x=(float(x0), float(x1)), y=(float(y0), float(y1)))
            return
        if camera is not None and hasattr(camera, "set"):
            camera.set(x=(float(x0), float(x1)), y=(float(y0), float(y1)))

    def set_xrange(self, x0: float, x1: float) -> None:
        self._set_panel_range(x0, x1, 0.0, 1.0)

    def set_draw_callback(self, fn: Optional[callable]) -> None:
        self._draw_cb = fn

    def request_draw(self) -> None:
        if hasattr(self.canvas, "update"):
            self.canvas.update()

    def close(self) -> None:
        if hasattr(self.canvas, "close"):
            self.canvas.close()


def add_overlay(panel: Any, x0: float, x1: float, y0: float, y1: float) -> None:
    for frac in (0.25, 0.5, 0.75):
        x = x0 + (x1 - x0) * frac
        pos = np.array([[x, y0, 0.0], [x, y1, 0.0]], dtype=np.float32)
        line = panel.visual("line")
        line.data("pos", pos)
        line.data("color", np.array([[0.6, 0.6, 0.6, 1.0]], dtype=np.float32))


def load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    if args.format == FMT_EDF:
        seg = load_edf_segment_pyedflib(args.file, args.load_start_s, args.load_duration_s, args.n_ch)
    elif args.format == FMT_NWB:
        seg = load_nwb_segment_pynwb(
            args.file,
            args.load_start_s,
            args.load_duration_s,
            args.n_ch,
            series_path=args.nwb_series_path,
            time_dim=args.nwb_time_dim,
        )
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    t, d, dec = decimate_for_display(seg.times_s, seg.data, args.max_points_per_trace)
    meta = dict(seg.meta)
    meta["decim_factor"] = int(dec)
    meta["n_points_per_trace"] = int(t.shape[0])
    return t.astype(np.float32), d.astype(np.float32), meta, float(seg.fs_hz)


def summarize_step_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    gcols = [
        "bench_id",
        "backend",
        "format",
        "file",
        "sequence",
        "n_channels",
        "window_s",
        "load_duration_s",
        "target_interval_ms",
    ]
    for key, grp in df.groupby(gcols):
        g = grp.copy()
        lat = g.loc[(g["was_dropped"] == 0) & np.isfinite(g["latency_ms"]), "latency_ms"].to_numpy(dtype=float)
        if lat.size == 0:
            p50 = p95 = p99 = mx = float("nan")
        else:
            p50 = float(np.percentile(lat, 50))
            p95 = float(np.percentile(lat, 95))
            p99 = float(np.percentile(lat, 99))
            mx = float(np.max(lat))

        drop_rate = float((g["was_dropped"] == 1).mean()) if "was_dropped" in g.columns else 0.0

        out = dict(zip(gcols, key))
        out.update(
            {
                "n_steps": int(len(g)),
                "lat_p50_ms": p50,
                "lat_p95_ms": p95,
                "lat_p99_ms": p99,
                "lat_max_ms": mx,
                "drop_rate": drop_rate,
            }
        )

        if "issue_late_ms" in g.columns:
            il = g["issue_late_ms"].to_numpy(dtype=float)
            il = il[np.isfinite(il)]
            out["issue_late_p95_ms"] = float(np.percentile(il, 95)) if il.size else float("nan")
            out["issue_late_max_ms"] = float(np.max(il)) if il.size else float("nan")

        rows.append(out)
    return pd.DataFrame(rows)


def _dv_run(dv: Any) -> None:
    if hasattr(dv, "run"):
        dv.run()
        return
    if hasattr(dv, "app"):
        app = dv.app()
        if hasattr(app, "run"):
            app.run()
            return
    raise RuntimeError("Datoviz run() API not found")


def _dv_quit(dv: Any) -> None:
    if hasattr(dv, "quit"):
        dv.quit()
        return
    if hasattr(dv, "app"):
        app = dv.app()
        if hasattr(app, "quit"):
            app.quit()
            return


def run_tffr_one(t: np.ndarray, data: np.ndarray, fs: float, args: argparse.Namespace) -> Dict[str, Any]:
    bench = DatovizBench(t, data, width=int(args.width), height=int(args.height))
    bench.show_and_start()

    t_init = float(t[0])
    bench.set_xrange(t_init, t_init + float(args.window_s))
    if args.overlay == OVL_ON:
        add_overlay(bench.panel, t_init, t_init + float(args.window_s), 0.0, 1.0)

    def stop_on_timeout() -> None:
        if bench.tffr_s is None:
            _dv_quit(bench.dv)

    timer = threading.Timer(float(args.hard_timeout_s), stop_on_timeout)
    timer.start()

    def on_draw() -> None:
        if bench.tffr_s is not None:
            _dv_quit(bench.dv)

    bench.set_draw_callback(on_draw)
    bench.request_draw()
    _dv_run(bench.dv)
    timer.cancel()

    out = {
        "bench_id": BENCH_TFFR,
        "tool": TOOL_DATOVIZ,
        "format": args.format,
        "file": args.file.name,
        "run": int(args.run_idx),
        "window_s": float(args.window_s),
        "n_channels": int(data.shape[0]),
        "overlay": args.overlay,
        "fs": float(fs),
        "T_first_render_s": float(bench.tffr_s) if bench.tffr_s is not None else None,
    }
    bench.close()
    return out


def run_interactions_one(
    t: np.ndarray,
    data: np.ndarray,
    fs: float,
    bench_id: str,
    sequence: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    bench = DatovizBench(t, data, width=int(args.width), height=int(args.height))
    bench.show_and_start()

    t_init = float(t[0])
    bench.set_xrange(t_init, t_init + float(args.window_s))
    if args.overlay == OVL_ON:
        add_overlay(bench.panel, t_init, t_init + float(args.window_s), 0.0, 1.0)

    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(sequence, lo, hi, float(args.window_s), int(args.steps))

    interval_s = float(args.target_interval_ms) / 1000.0
    t_start: Optional[float] = None

    pending: List[Cmd] = []
    results: List[Dict[str, Any]] = []

    next_step = {"idx": 0}
    done = {"flag": False}

    def issue_one(step_idx: int, now: float) -> None:
        x0, x1 = ranges[step_idx]
        x0 = max(float(t[0]), x0)
        x1 = min(float(t[-1]), x1)
        sched = (t_start or now) + (step_idx * interval_s if bench_id == BENCH_A2 else 0.0)
        pending.append(Cmd(step_idx=step_idx, x0=x0, x1=x1, t_sched_s=sched, t_cmd_s=now))
        bench.set_xrange(x0, x1)
        bench.request_draw()

    def maybe_issue(now: float) -> None:
        if t_start is None:
            return
        while next_step["idx"] < len(ranges):
            if bench_id == BENCH_A1:
                issue_one(next_step["idx"], now)
                next_step["idx"] += 1
                break
            if (now - t_start) >= next_step["idx"] * interval_s:
                issue_one(next_step["idx"], now)
                next_step["idx"] += 1
                continue
            break

    def finish_if_done() -> None:
        if done["flag"]:
            return
        if next_step["idx"] >= len(ranges) and not pending:
            done["flag"] = True
            _dv_quit(bench.dv)

    def on_draw() -> None:
        nonlocal t_start
        now = time.perf_counter()
        if t_start is None:
            t_start = now

        if pending:
            dropped = pending[:-1]
            matched = pending[-1]

            if bench_id == BENCH_A2 and dropped:
                for d in dropped:
                    results.append(
                        {
                            "step_idx": d.step_idx,
                            "x0": d.x0,
                            "x1": d.x1,
                            "t_sched_s": d.t_sched_s,
                            "t_cmd_s": d.t_cmd_s,
                            "t_paint_s": float("nan"),
                            "latency_ms": float("nan"),
                            "was_dropped": 1,
                            "dropped_before": None,
                            "issue_late_ms": (d.t_cmd_s - d.t_sched_s) * 1000.0,
                        }
                    )

            t_paint = time.perf_counter()
            results.append(
                {
                    "step_idx": matched.step_idx,
                    "x0": matched.x0,
                    "x1": matched.x1,
                    "t_sched_s": matched.t_sched_s,
                    "t_cmd_s": matched.t_cmd_s,
                    "t_paint_s": t_paint,
                    "latency_ms": (t_paint - matched.t_cmd_s) * 1000.0,
                    "was_dropped": 0,
                    "dropped_before": len(dropped),
                    "issue_late_ms": (matched.t_cmd_s - matched.t_sched_s) * 1000.0,
                }
            )
            pending[:] = []

        maybe_issue(now)
        finish_if_done()

    bench.set_draw_callback(on_draw)

    def stop_on_timeout() -> None:
        done["flag"] = True
        _dv_quit(bench.dv)

    timer = threading.Timer(float(args.hard_timeout_s), stop_on_timeout)
    timer.start()

    bench.request_draw()
    _dv_run(bench.dv)
    timer.cancel()

    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "bench_id": bench_id,
                "backend": TOOL_DATOVIZ,
                "format": args.format,
                "n_channels": int(data.shape[0]),
                "overlay_state": args.overlay,
                "cache_state": args.cache_state,
                "file": args.file.name,
                "run": int(args.run_idx),
                "sequence": sequence,
                "target_interval_ms": float(args.target_interval_ms),
                "window_s": float(args.window_s),
                "load_start_s": float(args.load_start_s),
                "load_duration_s": float(args.load_duration_s),
                "fs": float(fs),
                **r,
            }
        )
    bench.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-id", type=str, required=True, choices=[BENCH_TFFR, BENCH_A1, BENCH_A2])
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-ch", type=int, default=16)
    ap.add_argument("--load-start-s", type=float, default=0.0)
    ap.add_argument("--load-duration-s", type=float, default=300.0)
    ap.add_argument("--max-points-per-trace", type=int, default=0)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN_ZOOM)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--target-interval-ms", type=float, default=16.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--hard-timeout-s", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--out-root", type=str, default="outputs")
    ap.add_argument("--tag", type=str, default="datoviz")
    ap.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    args = ap.parse_args()

    t, data, meta, fs = load_segment(args)

    bench_id = args.bench_id
    out_base = out_dir(Path(args.out_root), bench_id, TOOL_DATOVIZ, args.tag)
    write_manifest(out_base, bench_id, TOOL_DATOVIZ, vars(args), extra={"format": args.format, "meta": meta})

    if bench_id == BENCH_TFFR:
        rows = []
        for run_idx in range(int(args.runs)):
            args.run_idx = run_idx
            rows.append(run_tffr_one(t, data, fs, args))
        df = pd.DataFrame(rows)
        out_csv = out_base / "datoviz_tffr.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} ({len(df)} rows)")
        return

    all_rows: List[Dict[str, Any]] = []
    for run_idx in range(int(args.runs)):
        args.run_idx = run_idx
        print(f"[{args.file.name}] {args.sequence} {bench_id} run {run_idx} (steps={args.steps})")
        rows = run_interactions_one(t, data, fs, bench_id, args.sequence, args)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_steps = out_base / f"datoviz_{bench_id}_steps.csv"
    df.to_csv(out_steps, index=False)
    print(f"Saved {out_steps} ({len(df)} rows)")

    out_sum = out_base / f"datoviz_{bench_id}_summary.csv"
    if df.empty:
        summary = pd.DataFrame()
        summary.to_csv(out_sum, index=False)
        print(f"Saved {out_sum} (no rows)")
        return

    summary = summarize_step_df(df)
    summary.to_csv(out_sum, index=False)
    print(f"Saved {out_sum}")
    if "lat_p95_ms" in summary.columns:
        print(summary.groupby(["sequence"])["lat_p95_ms"].median().sort_values())


if __name__ == "__main__":
    main()
