#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bench_plotly_interactions_v2.py

Deterministic (Option A) pan/zoom benchmark for Plotly, meant to be portable across machines.

Key properties:
- Generates a self-contained HTML (no server required).
- Loads EDF via pyEDFlib, normalizes channels, and DECIMATES to keep the HTML sane.
- Uses Plotly scattergl by default.
- Runs deterministic interaction sequences by calling Plotly.relayout() at a fixed cadence.
- Measures "command -> relayout resolved -> next rAF" latency per step (approx. visible update).
- Tracks "missed slots" when the cadence cannot be respected.

Run:
  python bench_plotly_interactions_v2.py --data-dir data --edf 230302e-b_0000.edf

Then open plotly_interactions.html and read the console output:
  BENCH_JSON: {...}
"""

from __future__ import annotations

import sys

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_manifest
from benchkit.bench_defaults import (
    DEFAULT_STEPS,
    DEFAULT_TARGET_INTERVAL_MS,
    DEFAULT_WINDOW_S,
    default_load_duration_s,
)
from benchkit.lexicon import (
    BENCH_A2,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    CACHE_WARM,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_PLOTLY,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb



def load_segment(
    path: Path,
    fmt: str,
    load_start_s: float,
    load_duration_s: float,
    n_channels_max: int,
    max_points_per_trace: int,
    nwb_series_path: str | None,
    nwb_time_dim: str,
) -> Tuple[List[float], List[List[float]], float, int, float, float, int]:
    """
    Returns:
      (times_list, data_list[ch], fs, n_ch, start, dur, decim_factor)

    Robust EDF reader:
    - skips annotation-like channels
    - keeps channels that match channel-0 sampling frequency (no resampling)
    - clamps to the shortest available length across chosen channels
    - decimates to keep generated HTML responsive
    """
    if fmt == FMT_EDF:
        seg = load_edf_segment_pyedflib(path, load_start_s, load_duration_s, n_channels_max)
    elif fmt == FMT_NWB:
        seg = load_nwb_segment_pynwb(
            path,
            load_start_s,
            load_duration_s,
            n_channels_max,
            series_path=nwb_series_path,
            time_dim=nwb_time_dim,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    times_np, data_np, decim = decimate_for_display(seg.times_s, seg.data, max_points_per_trace)
    times = times_np.astype(float).tolist()
    data = [data_np[i, :].astype(float).tolist() for i in range(data_np.shape[0])]
    return (
        times,
        data,
        float(seg.fs_hz),
        int(data_np.shape[0]),
        float(seg.meta.get("start_s", load_start_s)),
        float(seg.meta.get("duration_s", load_duration_s)),
        int(decim),
    )


def render_html(
    *,
    times: List[float],
    data: List[List[float]],
    meta: dict,
    window_s: float,
    target_interval_ms: float,
    steps: int,
    sequences: List[str],
    out_path: Path,
) -> None:
    # Keep layout minimal to reduce overhead.
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Plotly Interactions Bench</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #gd {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
<div id="gd"></div>

<script>
const META = {json.dumps(meta)};
const TIMES = {json.dumps(times)};
const DATA = {json.dumps(data)};

function percentile(arr, p) {{
  if (!arr.length) return null;
  const a = Array.from(arr).sort((x,y)=>x-y);
  const idx = (p/100) * (a.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return a[lo];
  return a[lo] + (a[hi]-a[lo])*(idx-lo);
}}

function sleep(ms) {{
  return new Promise(r => setTimeout(r, ms));
}}

async function relayoutAndMeasure(gd, update) {{
  const t0 = performance.now();
  await Plotly.relayout(gd, update);
  await new Promise(requestAnimationFrame);
  const t1 = performance.now();
  return t1 - t0;
}}

function clamp(x, a, b) {{
  return Math.max(a, Math.min(b, x));
}}

function makeTraces() {{
  const traces = [];
  for (let i = 0; i < DATA.length; i++) {{
    traces.push({{
      x: TIMES,
      y: DATA[i],
      type: "scattergl",
      mode: "lines",
      name: "ch" + i,
      hoverinfo: "skip",
      line: {{ width: 1 }}
    }});
  }}
  return traces;
}}

function initialLayout() {{
  const x0 = META.start_s;
  const x1 = META.start_s + Math.min({window_s}, META.dur_s);
  return {{
    margin: {{l: 40, r: 10, t: 30, b: 30}},
    title: "Plotly pan/zoom deterministic bench",
    showlegend: false,
    xaxis: {{ range: [x0, x1], fixedrange: false }},
    yaxis: {{ fixedrange: false }},
  }};
}}

async function runSequence(gd, name, steps, intervalMs, windowS) {{
  const rows = [];
  const lat = [];
  let missedTotal = 0;

  const startRange = gd.layout.xaxis.range.slice();
  const fullMin = META.start_s;
  const fullMax = META.start_s + META.dur_s;

  // Determine target path for ranges
  // PAN: shift window to the right across available duration
  // ZOOM_IN: shrink window around center
  // ZOOM_OUT: grow window around center
  // PAN_ZOOM: shift + shrink

  const center0 = (startRange[0] + startRange[1]) / 2;

  function rangeForStep(i) {{
    const t = (steps <= 1) ? 1.0 : (i / (steps - 1));
    let a = startRange[0];
    let b = startRange[1];

    if (name === "PAN_60S") {{
      // pan right by up to (dur - window)
      const span = windowS;
      const maxShift = Math.max(0, (fullMax - fullMin) - span);
      const shift = t * maxShift;
      a = fullMin + shift;
      b = a + span;
    }} else if (name === "ZOOM_IN") {{
      const span0 = windowS;
      const span1 = Math.max(0.5, windowS / 10); // down to 1s if window=10s
      const span = span0 + (span1 - span0) * t;
      a = center0 - span/2;
      b = center0 + span/2;
    }} else if (name === "ZOOM_OUT") {{
      const span0 = Math.max(0.5, windowS / 10);
      const span1 = Math.min(META.dur_s, windowS * 3); // up to 30s if window=10s
      const span = span0 + (span1 - span0) * t;
      a = center0 - span/2;
      b = center0 + span/2;
    }} else if (name === "PAN_ZOOM") {{
      const span0 = windowS;
      const span1 = Math.max(0.75, windowS / 5);
      const span = span0 + (span1 - span0) * t;

      const maxShift = Math.max(0, (fullMax - fullMin) - span);
      const shift = t * maxShift;
      a = fullMin + shift;
      b = a + span;
    }}

    // Clamp inside full extent
    const spanFinal = (b - a);
    if (spanFinal <= 0) {{
      a = fullMin;
      b = fullMin + Math.min(windowS, META.dur_s);
    }}
    if (a < fullMin) {{ a = fullMin; b = a + spanFinal; }}
    if (b > fullMax) {{ b = fullMax; a = b - spanFinal; }}
    a = clamp(a, fullMin, fullMax);
    b = clamp(b, fullMin, fullMax);
    return [a, b];
  }}

  const tStart = performance.now();
  for (let i = 0; i < steps; i++) {{
    const ideal = tStart + i * intervalMs;
    let now = performance.now();
    if (now < ideal) {{
      await sleep(ideal - now);
      now = performance.now();
    }}
    // missed cadence slots
    if (now > ideal + intervalMs) {{
      const missed = Math.floor((now - ideal) / intervalMs);
      missedTotal += missed;
    }}

    const r = rangeForStep(i);
    const latencyMs = await relayoutAndMeasure(gd, {{"xaxis.range": r}});
    lat.push(latencyMs);
    rows.push({{
      seq: name,
      step: i,
      ideal_ms: ideal,
      start_ms: now,
      latency_ms: latencyMs
    }});
  }}

  const res = {{
    seq: name,
    interval_ms: intervalMs,
    steps: steps,
    missed_total: missedTotal,
    latency_ms: {{
      mean: lat.reduce((a,b)=>a+b,0) / (lat.length || 1),
      p50: percentile(lat, 50),
      p95: percentile(lat, 95),
      max: lat.length ? Math.max(...lat) : null
    }},
    rows: rows
  }};
  return res;
}}

async function main() {{
  const gd = document.getElementById("gd");
  const traces = makeTraces();
  const layout = initialLayout();
  const config = {{
    responsive: true,
    displayModeBar: true
  }};

  const t0 = performance.now();
  await Plotly.newPlot(gd, traces, layout, config);
  await new Promise(requestAnimationFrame);
  const t1 = performance.now();

  const firstRenderMs = t1 - t0;

  const intervalMs = {target_interval_ms};
  const steps = {steps};
  const windowS = {window_s};

  const sequences = {json.dumps(sequences)};
  const results = [];

  for (const seq of sequences) {{
    // small pause between sequences
    await sleep(200);
    results.push(await runSequence(gd, seq, steps, intervalMs, windowS));
  }}

  // build compact summary
  const summary = results.map(r => {{
    return {{
      seq: r.seq,
      steps: r.steps,
      interval_ms: r.interval_ms,
      missed_total: r.missed_total,
      mean_ms: r.latency_ms.mean,
      p50_ms: r.latency_ms.p50,
      p95_ms: r.latency_ms.p95,
      max_ms: r.latency_ms.max
    }};
  }});

  const out = {{
    meta: META,
    first_render_ms: firstRenderMs,
    summary: summary,
    // step-level rows are large; keep them but you can delete if you want smaller output
    results: results
  }};

  console.log("BENCH_JSON:", JSON.stringify(out));
}}

main().catch(e => console.error(e));
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], default=FMT_EDF)
    ap.add_argument("--edf", type=str, default=None)
    ap.add_argument("--file", type=Path, default=None)
    ap.add_argument("--n-channels", "--n-ch", dest="n_channels", type=int, default=8)
    ap.add_argument("--load-start", "--load-start-s", dest="load_start", type=float, default=0.0)
    ap.add_argument("--load-duration", "--load-duration-s", dest="load_duration", type=float, default=None)
    ap.add_argument(
        "--max-points-per-trace",
        type=int,
        default=20000,
        help="Decimate each trace to at most this many points (keeps HTML responsive).",
    )
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--target-interval-ms", type=float, default=DEFAULT_TARGET_INTERVAL_MS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="edf_A2")
    ap.add_argument("--overlay-state", "--overlay", dest="overlay_state", type=str, default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)
    ap.add_argument(
        "--sequence",
        choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM, "ALL"],
        default="ALL",
    )
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])

    args = ap.parse_args()
    if args.load_duration is None:
        args.load_duration = default_load_duration_s(args.window_s)

    out_base = out_dir(args.out_root, BENCH_A2, TOOL_PLOTLY, args.tag)
    write_manifest(out_base, BENCH_A2, TOOL_PLOTLY, vars(args), extra={'format': args.format})
    out_html = out_base / 'plotly_A2_interactions.html'

    data_path = args.file or (args.data_dir / args.edf if args.edf else None)
    if data_path is None:
        raise SystemExit("Provide --file or --edf with --data-dir.")
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    times, data, fs, n_ch, start, dur, decim = load_segment(
        data_path,
        args.format,
        args.load_start,
        args.load_duration,
        args.n_channels,
        args.max_points_per_trace,
        args.nwb_series_path,
        args.nwb_time_dim,
    )

    meta = {
        "bench_id": BENCH_A2,
        "tool": TOOL_PLOTLY,
        "format": args.format,
        "overlay_state": args.overlay_state,
        "cache_state": args.cache_state,
        "edf": data_path.name,
        "fs_hz": fs,
        "n_ch": n_ch,
        "effective_n_ch": n_ch,
        "start_s": start,
        "dur_s": dur,
        "decim_factor": decim,
        "n_points_per_trace": len(times),
        "total_points": len(times) * n_ch,
        "window_s": args.window_s,
        "target_interval_ms": args.target_interval_ms,
        "steps": args.steps,
    }

    seq_map = {
        SEQ_PAN: "PAN_60S",
        SEQ_ZOOM_IN: "ZOOM_IN",
        SEQ_ZOOM_OUT: "ZOOM_OUT",
        SEQ_PAN_ZOOM: "PAN_ZOOM",
    }
    if args.sequence == "ALL":
        sequences = [seq_map[SEQ_PAN], seq_map[SEQ_ZOOM_IN], seq_map[SEQ_ZOOM_OUT], seq_map[SEQ_PAN_ZOOM]]
    else:
        sequences = [seq_map[args.sequence]]

    render_html(
        times=times,
        data=data,
        meta=meta,
        window_s=args.window_s,
        target_interval_ms=args.target_interval_ms,
        steps=args.steps,
        sequences=sequences,
        out_path=out_html,
    )
    print(f"Wrote {out_html} (open in browser; console prints BENCH_JSON).")  # noqa: T201


if __name__ == "__main__":
    main()
