from __future__ import annotations

import sys

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from vispy import app, scene

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import ensure_dir, env_info, out_dir
from benchkit.bench_defaults import (
    DEFAULT_STEPS,
    DEFAULT_WINDOW_S,
    PAN_STEP_FRACTION,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    default_load_duration_s,
)
from benchkit.lexicon import (
    BENCH_A1,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    TOOL_VISPY,
    SEQ_PAN,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    SEQ_PAN_ZOOM,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.output_contract import write_manifest_contract, write_steps_csv, write_steps_summary


def now_ms() -> float:
    return time.perf_counter() * 1000.0


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


def add_overlay(view: scene.widgets.ViewBox, x0: float, x1: float, y0: float, y1: float) -> None:
    for frac in (0.25, 0.5, 0.75):
        x = x0 + (x1 - x0) * frac
        pos = np.array([[x, y0], [x, y1]], dtype=np.float32)
        scene.visuals.Line(pos=pos, color="gray", width=1, parent=view.scene)


def load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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


def main() -> None:
    p = argparse.ArgumentParser(description="VisPy A1 throughput-only benchmark (coalescing-aware).")
    p.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--tag", type=str, required=True)

    p.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN_ZOOM)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--runs", type=int, default=1)

    p.add_argument("--n-ch", type=int, default=16)
    p.add_argument("--load-start-s", type=float, default=0.0)
    p.add_argument("--load-duration-s", type=float, default=None)
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--max-points-per-trace", type=int, default=5000)
    p.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)

    p.add_argument("--nwb-series-path", type=str, default=None)
    p.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    args = p.parse_args()
    if args.load_duration_s is None:
        args.load_duration_s = default_load_duration_s(args.window_s)

    if args.runs != 1:
        print(f"[WARN] forcing runs=1 (requested {args.runs})")
        args.runs = 1
    out = out_dir(args.out_root, BENCH_A1, TOOL_VISPY, args.tag)
    ensure_dir(out)
    write_manifest_contract(
        out,
        bench_id=BENCH_A1,
        tool_id=TOOL_VISPY,
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

    t, d, meta = load_segment(args)

    offsets = np.arange(d.shape[0], dtype=np.float32)[:, None]
    y = d + offsets
    x = t[None, :].repeat(y.shape[0], axis=0)

    canvas = scene.SceneCanvas(keys="interactive", size=(1200, 800), show=True, bgcolor="white")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.PanZoomCamera()

    for ch in range(y.shape[0]):
        pos = np.column_stack([x[ch], y[ch]]).astype(np.float32)
        scene.visuals.Line(pos=pos, color="black", width=1, parent=view.scene)

    lo, hi = float(t[0]), float(t[-1])
    x0 = lo
    x1 = min(hi, lo + float(args.window_s))
    y0 = -2.0
    y1 = float(y.shape[0]) + 2.0
    view.camera.rect = (x0, y0, x1 - x0, y1 - y0)

    if args.overlay == OVL_ON:
        add_overlay(view, x0, x1, y0, y1)

    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    issued_ms: List[float] = []
    latest_issued_id = -1
    last_presented_id = -1

    presented_id: List[int] = []
    presented_at_ms: List[float] = []
    latency_ms: List[float] = []
    dropped_before: List[int] = []

    def on_draw(event) -> None:
        nonlocal last_presented_id
        if latest_issued_id < 0:
            return
        if latest_issued_id == last_presented_id:
            return

        drop = max(0, latest_issued_id - last_presented_id - 1)
        pid = latest_issued_id
        t_present = now_ms()
        t_issue = issued_ms[pid]

        presented_id.append(pid)
        presented_at_ms.append(t_present)
        latency_ms.append(float(t_present - t_issue))
        dropped_before.append(int(drop))
        last_presented_id = pid

    canvas.events.draw.connect(on_draw)

    for i, (rx0, rx1) in enumerate(ranges):
        latest_issued_id = i
        issued_ms.append(now_ms())
        view.camera.rect = (float(rx0), y0, float(rx1 - rx0), y1 - y0)
        canvas.update()

    t0 = time.time()
    while last_presented_id < (len(ranges) - 1) and (time.time() - t0) < 10.0:
        app.process_events()

    presented_lookup = {
        pid: (presented_at_ms[idx], latency_ms[idx], dropped_before[idx]) for idx, pid in enumerate(presented_id)
    }

    steps_rows: List[Dict[str, Any]] = []
    for step_idx in range(len(ranges)):
        issued = issued_ms[step_idx] if step_idx < len(issued_ms) else float("nan")
        if step_idx in presented_lookup:
            presented, latency, drop = presented_lookup[step_idx]
            steps_rows.append(
                {
                    "step_id": step_idx,
                    "latency_ms": float(latency),
                    "noop": False,
                    "status": "OK",
                    "issued_ms": float(issued),
                    "presented_ms": float(presented),
                    "dropped_before": int(drop),
                    "was_dropped": 0,
                }
            )
        else:
            steps_rows.append(
                {
                    "step_id": step_idx,
                    "latency_ms": 0.0,
                    "noop": False,
                    "status": "FAIL",
                    "issued_ms": float(issued) if issued == issued else float("nan"),
                    "presented_ms": float("nan"),
                    "dropped_before": None,
                    "was_dropped": 1,
                }
            )

    write_steps_csv(out, steps_rows)
    total_time_s = (
        max(1e-6, (max(presented_at_ms) - min(issued_ms)) / 1000.0) if presented_at_ms else float("nan")
    )
    fps = (len(presented_id) / total_time_s) if total_time_s == total_time_s else float("nan")
    write_steps_summary(
        out,
        steps_rows,
        extra={
            "bench_id": BENCH_A1,
            "tool_id": TOOL_VISPY,
            "format": args.format,
            "sequence": args.sequence,
            "overlay": args.overlay,
            "window_s": float(args.window_s),
            "steps_issued": int(args.steps),
            "frames_presented": int(len(presented_id)),
            "fps_presented": float(fps),
            "meta": meta,
        },
    )

    canvas.close()


if __name__ == "__main__":
    main()
