#!/usr/bin/env python3
"""scripts/desktop/bench_vispy.py

VisPy desktop benchmarks (Qt backend):
- TFFR: time-to-first-render (first draw event)
- A1_THROUGHPUT: as-fast-as-possible interactions (issue next command on paint)
- A2_CADENCED: cadenced interactions (issue commands on a fixed interval)

This script follows the folder/manifest conventions used in the existing pack.

Outputs (per bench):
  outputs/<BENCH_ID>/VIS_VISPY/<tag>/
    - manifest.json
    - vispy_<bench>_steps.csv
    - vispy_<bench>_summary.csv   (interaction benches)
    - vispy_tffr.csv              (TFFR bench)
"""

from __future__ import annotations

import sys

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir
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
    TOOL_VISPY,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.output_contract import (
    write_manifest_contract,
    write_steps_csv,
    write_steps_summary,
    write_tffr_csv,
    write_tffr_summary,
)


def normalize_bench_id(bench_id: str) -> str:
    if bench_id == "A1":
        return BENCH_A1
    if bench_id == "A2":
        return BENCH_A2
    return bench_id


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
    pan_step = w * PAN_STEP_FRACTION
    rng: List[Tuple[float, float]] = []

    for i in range(steps):
        if sequence == SEQ_PAN:
            x0, x1 = x0 + pan_step, x1 + pan_step
            if x1 > hi or x0 < lo:
                pan_step *= -1.0
        elif sequence == SEQ_ZOOM_IN:
            cx = 0.5 * (x0 + x1)
            w = max(w * ZOOM_IN_FACTOR, window_s * 0.10)
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif sequence == SEQ_ZOOM_OUT:
            cx = 0.5 * (x0 + x1)
            w = min(w * ZOOM_OUT_FACTOR, hi - lo)
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif sequence == SEQ_PAN_ZOOM:
            if i % 2 == 0:
                x0, x1 = x0 + pan_step, x1 + pan_step
                if x1 > hi or x0 < lo:
                    pan_step *= -1.0
            else:
                cx = 0.5 * (x0 + x1)
                w = max(min(w * ZOOM_IN_FACTOR, hi - lo), window_s * 0.10)
                x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        else:
            raise ValueError(sequence)

        x0, x1 = clamp_range(x0, x1, lo, hi)
        rng.append((x0, x1))
    return rng


def add_overlay(view, x0: float, x1: float, y0: float, y1: float) -> None:
    from vispy import scene

    for frac in (0.25, 0.5, 0.75):
        x = x0 + (x1 - x0) * frac
        pos = np.array([[x, y0], [x, y1]], dtype=np.float32)
        scene.visuals.Line(pos=pos, color="gray", width=1, parent=view.scene)


class VispyBench:
    """
    A VisPy scene that supports:
    - first draw timestamp capture (TFFR)
    - camera x-range updates
    - draw-event callback for measuring command->draw latency
    """

    def __init__(self, t: np.ndarray, data: np.ndarray, width: int, height: int):
        from vispy import app, scene

        self.app = app
        self.scene = scene

        self.canvas = scene.SceneCanvas(keys=None, size=(width, height), show=False, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=None)
        self.view.camera.set_range(x=(float(t[0]), float(t[-1])), y=(0.0, 1.0))

        self._n_ch = int(data.shape[0])

        # layout: stack traces in [0..1] with constant lane heights
        self._lane_h = 1.0 / max(1, self._n_ch)
        self._amp = 0.35 * self._lane_h

        for i in range(self._n_ch):
            y0 = (i + 0.5) * self._lane_h
            y = y0 + self._amp * data[i]
            xy = np.column_stack([t, y]).astype(np.float32, copy=False)
            line = scene.visuals.Line(xy, color="white", parent=self.view.scene, method="gl")
            line.set_data(pos=xy)

        self._first_draw_s: Optional[float] = None
        self._t0_s: Optional[float] = None

        @self.canvas.events.draw.connect
        def _on_draw(ev):
            if self._first_draw_s is None:
                self._first_draw_s = time.perf_counter()
            if self._draw_cb is not None:
                self._draw_cb()

        self._draw_cb: Optional[callable] = None

    def show_and_start(self) -> None:
        self.canvas.show()
        # keep consistent with existing TFFR script style: start timing right after show()
        self._t0_s = time.perf_counter()

    @property
    def tffr_s(self) -> Optional[float]:
        if self._t0_s is None or self._first_draw_s is None:
            return None
        return self._first_draw_s - self._t0_s

    def set_xrange(self, x0: float, x1: float) -> None:
        # keep y range stable; only change x range deterministically
        y0, y1 = 0.0, 1.0
        self.view.camera.set_range(x=(float(x0), float(x1)), y=(y0, y1))

    def get_xrange(self) -> Tuple[float, float]:
        rect = self.view.camera.rect
        return float(rect.left), float(rect.right)

    def set_draw_callback(self, fn: Optional[callable]) -> None:
        self._draw_cb = fn

    def close(self) -> None:
        self.canvas.close()


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
    """
    Match the summary style used by existing desktop scripts: percentiles over latency_ms,
    plus drop rate for A2.
    """
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

        # cadence adherence proxies (A2 only; columns are present there)
        if "issue_late_ms" in g.columns:
            il = g["issue_late_ms"].to_numpy(dtype=float)
            il = il[np.isfinite(il)]
            out["issue_late_p95_ms"] = float(np.percentile(il, 95)) if il.size else float("nan")
            out["issue_late_max_ms"] = float(np.max(il)) if il.size else float("nan")

        rows.append(out)
    return pd.DataFrame(rows)


def run_tffr_one(t: np.ndarray, data: np.ndarray, fs: float, args: argparse.Namespace) -> Dict[str, Any]:
    from vispy import app as visapp

    bench = VispyBench(t, data, width=int(args.width), height=int(args.height))
    bench.show_and_start()

    t_init = float(t[0])
    bench.set_xrange(t_init, t_init + float(args.window_s))
    if args.overlay == OVL_ON:
        add_overlay(bench.view, t_init, t_init + float(args.window_s), 0.0, 1.0)

    t0 = time.perf_counter()
    while bench.tffr_s is None and (time.perf_counter() - t0) < float(args.hard_timeout_s):
        visapp.process_events()

    out = {
        "bench_id": BENCH_TFFR,
        "tool": TOOL_VISPY,
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
    visapp.process_events()
    return out


def run_interactions_one(
    t: np.ndarray,
    data: np.ndarray,
    fs: float,
    bench_id: str,
    sequence: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Return step-level rows (same schema shape as existing desktop scripts).
    """
    from vispy import app as visapp

    bench = VispyBench(t, data, width=int(args.width), height=int(args.height))
    bench.show_and_start()

    # initial view: [t0, t0+window]
    t_init = float(t[0])
    bench.set_xrange(t_init, t_init + float(args.window_s))
    if args.overlay == OVL_ON:
        add_overlay(bench.view, t_init, t_init + float(args.window_s), 0.0, 1.0)

    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(sequence, lo, hi, float(args.window_s), int(args.steps))

    # Controller state
    eps = float(args.eps)
    t_start = time.perf_counter()
    interval_s = float(args.target_interval_ms) / 1000.0

    pending: List[Cmd] = []
    results: List[Dict[str, Any]] = []

    next_step = {"idx": 0}
    done = {"flag": False}

    def issue_one(step_idx: int) -> None:
        x0, x1 = ranges[step_idx]
        # clamp to loaded segment
        x0 = max(float(t[0]), x0)
        x1 = min(float(t[-1]), x1)
        now = time.perf_counter()
        sched = t_start + (step_idx * interval_s if bench_id == BENCH_A2 else 0.0)
        pending.append(Cmd(step_idx=step_idx, x0=x0, x1=x1, t_sched_s=sched, t_cmd_s=now))
        bench.set_xrange(x0, x1)
        bench.canvas.update()

    def finish_if_done() -> None:
        if done["flag"]:
            return
        if next_step["idx"] >= len(ranges) and not pending:
            done["flag"] = True

    # A2 issuance timer
    timer = {"obj": None}

    def on_timer(ev=None) -> None:
        if next_step["idx"] >= len(ranges):
            # stop issuing
            if timer["obj"] is not None:
                timer["obj"].stop()
            return
        issue_one(next_step["idx"])
        next_step["idx"] += 1

    # Draw callback: treat the latest pending command as presented
    def on_draw() -> None:
        if not pending:
            return

        # Treat all but the latest as dropped (A2) or ignored (A1).
        dropped = pending[:-1]
        matched = pending[-1]
        after: List[Cmd] = []

        if dropped:
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
                        "issue_late_ms": (d.t_cmd_s - d.t_sched_s) * 1000.0 if bench_id == BENCH_A2 else float("nan"),
                    }
                )

        # Matched row
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
                "issue_late_ms": (matched.t_cmd_s - matched.t_sched_s) * 1000.0 if bench_id == BENCH_A2 else float("nan"),
            }
        )

        # update pending
        pending[:] = after

        if bench_id == BENCH_A1:
            # issue next immediately (throughput-only, no sleeps)
            if next_step["idx"] < len(ranges):
                issue_one(next_step["idx"])
                next_step["idx"] += 1

        finish_if_done()

    bench.set_draw_callback(on_draw)

    # Drive the event loop with timers so updates run even when the window has focus.
    hard_timeout = float(args.hard_timeout_s)
    loop_start = time.perf_counter()
    started = {"flag": False}

    def pump(ev=None) -> None:
        if done["flag"]:
            if timer["obj"] is not None:
                timer["obj"].stop()
            bench.close()
            visapp.quit()
            return

        if (time.perf_counter() - loop_start) > hard_timeout:
            done["flag"] = True
            if timer["obj"] is not None:
                timer["obj"].stop()
            bench.close()
            visapp.quit()
            return

        # Wait for the first draw before issuing commands.
        if bench.tffr_s is None:
            return

        if started["flag"]:
            return
        started["flag"] = True

        if bench_id == BENCH_A1:
            next_step["idx"] = 0
            issue_one(0)
            next_step["idx"] = 1
        else:
            next_step["idx"] = 0
            timer["obj"] = visapp.Timer(interval=interval_s, connect=on_timer, start=True)

    pump_timer = visapp.Timer(interval=0.01, connect=pump, start=True)
    visapp.run()

    if pending:
        for d in pending:
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
                    "issue_late_ms": (d.t_cmd_s - d.t_sched_s) * 1000.0 if bench_id == BENCH_A2 else float("nan"),
                }
            )

    if len(results) < len(ranges):
        seen = {row["step_idx"] for row in results}
        for step_idx in range(len(ranges)):
            if step_idx in seen:
                continue
            x0, x1 = ranges[step_idx]
            sched = t_start + (step_idx * interval_s if bench_id == BENCH_A2 else 0.0)
            results.append(
                {
                    "step_idx": step_idx,
                    "x0": x0,
                    "x1": x1,
                    "t_sched_s": sched,
                    "t_cmd_s": float("nan"),
                    "t_paint_s": float("nan"),
                    "latency_ms": float("nan"),
                    "was_dropped": 1,
                    "dropped_before": None,
                    "issue_late_ms": float("nan"),
                }
            )

    # Build rows, including metadata
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "bench_id": bench_id,
                "backend": TOOL_VISPY,
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
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-id", type=str, required=True, choices=[BENCH_TFFR, BENCH_A1, BENCH_A2, "A1", "A2"])
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--n-ch", type=int, default=16)
    ap.add_argument("--load-start-s", type=float, default=0.0)
    ap.add_argument("--load-duration-s", type=float, default=None)
    ap.add_argument("--max-points-per-trace", type=int, default=0)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN_ZOOM)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--target-interval-ms", type=float, default=DEFAULT_TARGET_INTERVAL_MS)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--hard-timeout-s", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--out-root", type=str, default="outputs")
    ap.add_argument("--tag", type=str, default="vispy")
    ap.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    args = ap.parse_args()
    args.bench_id = normalize_bench_id(args.bench_id)
    if args.load_duration_s is None:
        args.load_duration_s = default_load_duration_s(args.window_s)

    t, data, meta, fs = load_segment(args)

    bench_id = args.bench_id
    if args.runs != 1:
        print(f"[WARN] forcing runs=1 (requested {args.runs})")
        args.runs = 1
    out_base = out_dir(Path(args.out_root), bench_id, TOOL_VISPY, args.tag)
    write_manifest_contract(
        out_base,
        bench_id=bench_id,
        tool_id=TOOL_VISPY,
        fmt=args.format,
        file_path=args.file,
        window_s=float(args.window_s),
        n_channels=int(args.n_ch),
        sequence=args.sequence,
        overlay=args.overlay,
        run_id=0,
        steps_target=int(args.steps),
        extra={"meta": meta, "cache_state": args.cache_state},
    )

    if bench_id == BENCH_TFFR:
        args.run_idx = 0
        row = run_tffr_one(t, data, fs, args)
        tffr_s = row.get("T_first_render_s")
        tffr_ms = float(tffr_s) * 1000.0 if tffr_s is not None else float("nan")
        write_tffr_csv(out_base, run_id=0, tffr_ms=tffr_ms)
        write_tffr_summary(
            out_base,
            bench_id=BENCH_TFFR,
            tool_id=TOOL_VISPY,
            fmt=args.format,
            window_s=float(args.window_s),
            n_channels=int(args.n_ch),
            tffr_ms=tffr_ms,
            extra={"overlay": args.overlay, "meta": meta},
        )
        return

    # interactions
    args.run_idx = 0
    print(f"[{args.file.name}] {args.sequence} {bench_id} run 0 (steps={args.steps})")
    rows = run_interactions_one(t, data, fs, bench_id, args.sequence, args)
    steps_rows: List[Dict[str, Any]] = []
    for row in rows:
        latency = row.get("latency_ms", float("nan"))
        ok = np.isfinite(latency) and int(row.get("was_dropped", 0)) == 0
        steps_rows.append(
            {
                "step_id": int(row.get("step_idx", 0)),
                "latency_ms": float(latency) if ok else 0.0,
                "noop": False,
                "status": "OK" if ok else "FAIL",
                "x0": row.get("x0"),
                "x1": row.get("x1"),
                "issue_late_ms": row.get("issue_late_ms"),
                "was_dropped": row.get("was_dropped"),
            }
        )
    steps_rows.sort(key=lambda r: r["step_id"])
    write_steps_csv(out_base, steps_rows)
    write_steps_summary(
        out_base,
        steps_rows,
        extra={
            "bench_id": bench_id,
            "tool_id": TOOL_VISPY,
            "format": args.format,
            "sequence": args.sequence,
            "overlay": args.overlay,
            "window_s": float(args.window_s),
            "steps": int(args.steps),
            "target_interval_ms": float(args.target_interval_ms),
            "n_ch": int(args.n_ch),
            "fs_hz": float(fs),
            "meta": meta,
        },
    )


if __name__ == "__main__":
    main()
