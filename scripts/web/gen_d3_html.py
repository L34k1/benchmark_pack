from __future__ import annotations

import sys

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import ensure_dir, env_info, out_dir, write_json, write_manifest
from benchkit.lexicon import (
    BENCH_TFFR,
    BENCH_A1,
    BENCH_A2,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    SEQ_PAN_ZOOM,
    TOOL_D3,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb


D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"


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


def make_html(payload: Dict[str, Any]) -> str:
    js_payload = json.dumps(payload)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>D3 Benchmark</title>
  <script src="{D3_CDN}"></script>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #root {{ width: 100%; height: 100%; display: flex; }}
    canvas {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
<div id="root"><canvas id="c"></canvas></div>
<script>
const PAYLOAD = {js_payload};

function nowMs() {{ return performance.now(); }}
function nextFrame() {{ return new Promise(r => requestAnimationFrame(r)); }}

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d', {{ alpha: false }});

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || window.innerWidth;
  const h = canvas.clientHeight || window.innerHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}
window.addEventListener('resize', resize);
resize();

const margin = {{l: 50, r: 10, t: 10, b: 35}};

function clear() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}}

const t = PAYLOAD.t;             // [n]
const y = PAYLOAD.y;             // [n_ch][n]
const nCh = y.length;

function yDomain() {{
  return [-2, nCh + 2];
}}

function draw(range) {{
  const width = (canvas.clientWidth || window.innerWidth);
  const height = (canvas.clientHeight || window.innerHeight);

  const xScale = d3.scaleLinear().domain(range).range([margin.l, width - margin.r]);
  const yScale = d3.scaleLinear().domain(yDomain()).range([height - margin.b, margin.t]);

  clear();

  if (PAYLOAD.overlay === 'OVL_ON') {{
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    const fx = [0.25, 0.5, 0.75];
    for (const f of fx) {{
      const x0 = range[0] + (range[1]-range[0])*f;
      const xp = xScale(x0);
      ctx.beginPath();
      ctx.moveTo(xp, margin.t);
      ctx.lineTo(xp, height - margin.b);
      ctx.stroke();
    }}
  }}

  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1;

  for (let ch=0; ch<nCh; ch++) {{
    ctx.beginPath();
    const yy = y[ch];
    for (let i=0; i<t.length; i++) {{
      const xp = xScale(t[i]);
      const yp = yScale(yy[i]);
      if (i===0) ctx.moveTo(xp, yp);
      else ctx.lineTo(xp, yp);
    }}
    ctx.stroke();
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

async function run() {{
  const ranges = PAYLOAD.ranges || [];
  const lat = [];
  const lateness = [];

  const tStart = nowMs();
  await nextFrame();
  draw(PAYLOAD.initial_range);
  await nextFrame();
  const tAfter = nowMs();

  if (PAYLOAD.bench_id === 'TFFR') {{
    console.log('BENCH_JSON:' + JSON.stringify({{
      bench_id: PAYLOAD.bench_id,
      tool_id: PAYLOAD.tool_id,
      format: PAYLOAD.format,
      overlay: PAYLOAD.overlay,
      window_s: PAYLOAD.window_s,
      n_ch: PAYLOAD.n_ch,
      tffr_ms: (tAfter - tStart),
      meta: PAYLOAD.meta,
    }}));
    return;
  }}

  let latestIssued = -1;
  let lastPresented = -1;
  const issuedMs = [];
  const schedMs = [];
  let currentRange = PAYLOAD.initial_range;

  async function presentFrame() {{
    await nextFrame();
    draw(currentRange);
    await nextFrame();
    const tPres = nowMs();

    if (latestIssued <= lastPresented) return;

    const pid = latestIssued;
    const drop = Math.max(0, latestIssued - lastPresented - 1);
    const tIss = issuedMs[pid];

    lat.push(tPres - tIss);
    if (PAYLOAD.bench_id === 'A2_CADENCED') {{
      lateness.push(tPres - schedMs[pid]);
    }}
    lastPresented = pid;
  }}

  const t0 = nowMs();
  const interval = PAYLOAD.target_interval_ms || 0;

  if (PAYLOAD.bench_id === 'A1_THROUGHPUT') {{
    for (let i=0; i<ranges.length; i++) {{
      latestIssued = i;
      issuedMs.push(nowMs());
      currentRange = ranges[i];
    }}

    const deadline = nowMs() + 10000;
    while (lastPresented < ranges.length - 1 && nowMs() < deadline) {{
      await presentFrame();
    }}
  }} else {{
    for (let i=0; i<ranges.length; i++) {{
      const sched = t0 + i*interval;
      schedMs.push(sched);
      while (nowMs() < sched) {{
        await nextFrame();
      }}
      latestIssued = i;
      issuedMs.push(nowMs());
      currentRange = ranges[i];
      await presentFrame();
    }}
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
    frames_presented: lat.length,
    lat_p50_ms: pct(lat, 50),
    lat_p95_ms: pct(lat, 95),
    lat_max_ms: Math.max(...lat),
    lateness_p50_ms: (PAYLOAD.bench_id === 'A2_CADENCED') ? pct(lateness, 50) : null,
    lateness_p95_ms: (PAYLOAD.bench_id === 'A2_CADENCED') ? pct(lateness, 95) : null,
    lateness_max_ms: (PAYLOAD.bench_id === 'A2_CADENCED') ? Math.max(...lateness) : null,
    meta: PAYLOAD.meta,
  }};

  console.log('BENCH_JSON:' + JSON.stringify(out));
}}

run().catch(e => console.error('BENCH_ERROR', e));
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Generate D3.js benchmark HTML (TFFR/A1/A2).")
    p.add_argument("--bench-id", choices=[BENCH_TFFR, BENCH_A1, BENCH_A2], required=True)
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

    out = out_dir(args.out_root, args.bench_id, TOOL_D3, args.tag)
    ensure_dir(out)
    write_manifest(out, args.bench_id, TOOL_D3, args=vars(args), extra={"env": env_info()})

    t, d, meta = load_segment(args)
    offsets = np.arange(d.shape[0], dtype=np.float32)[:, None]
    y = (d + offsets).astype(np.float32)

    lo, hi = float(t[0]), float(t[-1])
    initial_range = [lo, min(hi, lo + float(args.window_s))]

    payload: Dict[str, Any] = {
        "bench_id": args.bench_id,
        "tool_id": TOOL_D3,
        "format": args.format,
        "overlay": args.overlay,
        "sequence": args.sequence,
        "window_s": float(args.window_s),
        "steps": int(args.steps),
        "target_interval_ms": float(args.target_interval_ms) if args.bench_id == BENCH_A2 else 0.0,
        "n_ch": int(args.n_ch),
        "t": t.astype(float).tolist(),
        "y": y.astype(float).tolist(),
        "initial_range": initial_range,
        "meta": meta,
    }

    if args.bench_id in (BENCH_A1, BENCH_A2):
        payload["ranges"] = [list(r) for r in build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))]

    html = make_html(payload)
    html_path = out / "d3_bench.html"
    html_path.write_text(html, encoding="utf-8")
    write_json(out / "html_manifest.json", {"html": str(html_path)})

    print(str(html_path))


if __name__ == "__main__":
    main()
