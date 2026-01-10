from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from benchkit.common import ensure_dir, out_dir, write_json, write_manifest, env_info
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
    BENCH_A1, BENCH_A2, BENCH_TFFR,
    FMT_EDF, FMT_NWB, OVL_OFF, OVL_ON,
    SEQ_PAN, SEQ_PAN_ZOOM, SEQ_ZOOM_IN, SEQ_ZOOM_OUT,
    TOOL_PLOTLY,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


PLOTLY_COMMON_JS = r"""
function nowMs(){ return performance.now(); }
async function sleepRaf(n=1){
  for (let i=0;i<n;i++){
    await new Promise(r => requestAnimationFrame(r));
  }
}
"""


JS_RUNNER_TEMPLATE = r"""
{COMMON_JS}

const PAYLOAD = __PAYLOAD__;

async function run() {{
  const gd = document.getElementById('plot');
  const tStart = nowMs();

  await Plotly.newPlot(gd, PAYLOAD.traces, PAYLOAD.layout, {{displayModeBar: false}});
  await sleepRaf(1);
  const tAfter = nowMs();

  if (PAYLOAD.bench_id === '__BENCH_TFFR__') {{
    const out = {{
      bench_id: PAYLOAD.bench_id,
      tool_id: PAYLOAD.tool_id,
      format: PAYLOAD.format,
      overlay: PAYLOAD.overlay,
      window_s: PAYLOAD.window_s,
      n_ch: PAYLOAD.n_ch,
      tffr_ms: (tAfter - tStart),
      meta: PAYLOAD.meta,
    }};
    console.log('BENCH_JSON:' + JSON.stringify(out));
    return;
  }}

  const lat = [];
  const lateness = [];
  const ranges = PAYLOAD.ranges;
  const t0 = nowMs();
  const interval = PAYLOAD.target_interval_ms || 0;

  for (let i = 0; i < ranges.length; i++) {{
    const sched = t0 + i * interval;

    if (PAYLOAD.bench_id === '__BENCH_A2__') {{
      while (nowMs() < sched) {{
        await sleepRaf(1);
      }}
    }}

    const stepStart = nowMs();
    await Plotly.relayout(gd, {{'xaxis.range': ranges[i]}});
    await sleepRaf(1);
    const stepEnd = nowMs();

    lat.push(stepEnd - stepStart);
    if (PAYLOAD.bench_id === '__BENCH_A2__') {{
      lateness.push(stepEnd - sched);
    }}
  }}

  function pct(arr, q) {{
    const a = arr.slice().sort((a,b)=>a-b);
    if (a.length === 0) return NaN;
    const idx = (q/100)*(a.length-1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo===hi) return a[lo];
    return a[lo]*(hi-idx) + a[hi]*(idx-lo);
  }}

  const out = {{
    bench_id: PAYLOAD.bench_id,
    tool_id: PAYLOAD.tool_id,
    format: PAYLOAD.format,
    overlay: PAYLOAD.overlay,
    sequence: PAYLOAD.sequence,
    window_s: PAYLOAD.window_s,
    steps: PAYLOAD.steps,
    target_interval_ms: PAYLOAD.target_interval_ms,
    lat_p50_ms: pct(lat, 50),
    lat_p95_ms: pct(lat, 95),
    lat_max_ms: Math.max(...lat),
    n: lat.length,
    lateness_p50_ms: PAYLOAD.bench_id === '__BENCH_A2__' ? pct(lateness, 50) : null,
    lateness_p95_ms: PAYLOAD.bench_id === '__BENCH_A2__' ? pct(lateness, 95) : null,
    lateness_max_ms: PAYLOAD.bench_id === '__BENCH_A2__' ? Math.max(...lateness) : null,
    lat_ms: lat,
    lateness_ms: lateness,
    meta: PAYLOAD.meta,
  }};
  console.log('BENCH_JSON:' + JSON.stringify(out));
}}

run().catch(e => {{
  console.error('BENCH_ERROR', e);
}});
"""


def build_ranges(sequence: str, lo: float, hi: float, window_s: float, steps: int) -> List[Tuple[float, float]]:
    x0, x1 = lo, min(lo + window_s, hi)
    w = x1 - x0
    pan_step = w * PAN_STEP_FRACTION
    out: List[Tuple[float, float]] = []

    def clamp(a: float, b: float) -> Tuple[float, float]:
        width = b - a
        if width <= 0:
            width = 1e-6
        if a < lo:
            a, b = lo, lo + width
        if b > hi:
            b, a = hi, hi - width
        if a < lo:
            a, b = lo, hi
        return a, b

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
        x0, x1 = clamp(x0, x1)
        out.append((x0, x1))
    return out


def load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
    return t, d, meta


def make_traces(t: np.ndarray, d: np.ndarray) -> List[Dict[str, Any]]:
    offsets = np.arange(d.shape[0], dtype=np.float32)[:, None]
    y = d + offsets
    t_list = t.astype(float).tolist()
    traces: List[Dict[str, Any]] = []
    for ch in range(y.shape[0]):
        traces.append({
            "type": "scattergl",
            "mode": "lines",
            "x": t_list,
            "y": y[ch].astype(float).tolist(),
            "name": f"ch{ch}",
            "line": {"width": 1},
        })
    return traces


def make_layout(t0: float, t1: float, overlay: str) -> Dict[str, Any]:
    layout: Dict[str, Any] = {
        "title": "",
        "showlegend": False,
        "margin": {"l": 40, "r": 10, "t": 10, "b": 35},
        "xaxis": {"range": [t0, t1]},
        "yaxis": {"title": "", "showticklabels": False},
    }
    if overlay == OVL_ON:
        shapes = []
        for frac in (0.25, 0.5, 0.75):
            x = t0 + (t1 - t0) * frac
            shapes.append({"type": "line", "x0": x, "x1": x, "y0": 0, "y1": 1, "xref": "x", "yref": "paper", "line": {"width": 1}})
        layout["shapes"] = shapes
    return layout


def html_template(payload: Dict[str, Any]) -> str:
    js_payload = json.dumps(payload)
    js = JS_RUNNER_TEMPLATE.format(COMMON_JS=PLOTLY_COMMON_JS).replace("__PAYLOAD__", js_payload)
    js = js.replace("__BENCH_TFFR__", BENCH_TFFR).replace("__BENCH_A2__", BENCH_A2)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Plotly Benchmark</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #plot {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
{js}
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Plotly benchmark HTML (TFFR/A1/A2).")
    p.add_argument("--bench-id", choices=[BENCH_TFFR, BENCH_A1, BENCH_A2], required=True)
    p.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    p.add_argument("--file", type=Path, required=True)

    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--tag", type=str, required=True)

    p.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN_ZOOM)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--target-interval-ms", type=float, default=DEFAULT_TARGET_INTERVAL_MS)

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

    out = out_dir(args.out_root, args.bench_id, TOOL_PLOTLY, args.tag)
    ensure_dir(out)
    write_manifest(out, args.bench_id, TOOL_PLOTLY, args=vars(args), extra={"env": env_info()})

    t, d, meta = load_segment(args)
    lo, hi = float(t[0]), float(t[-1])

    payload: Dict[str, Any] = {
        "bench_id": args.bench_id,
        "tool_id": TOOL_PLOTLY,
        "format": args.format,
        "overlay": args.overlay,
        "sequence": args.sequence,
        "window_s": float(args.window_s),
        "steps": int(args.steps),
        "target_interval_ms": float(args.target_interval_ms) if args.bench_id == BENCH_A2 else 0.0,
        "n_ch": int(args.n_ch),
        "traces": make_traces(t, d),
        "layout": make_layout(lo, min(lo + float(args.window_s), hi), args.overlay),
        "meta": meta,
    }

    if args.bench_id in (BENCH_A1, BENCH_A2):
        payload["ranges"] = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))

    html = html_template(payload)
    html_path = out / "plotly_bench.html"
    html_path.write_text(html, encoding="utf-8")

    write_json(out / "html_manifest.json", {"html": str(html_path), "payload_keys": list(payload.keys())})
    print(str(html_path))


if __name__ == "__main__":
    main()
