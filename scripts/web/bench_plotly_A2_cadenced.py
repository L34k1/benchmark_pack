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
import pyedflib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_manifest
from benchkit.lexicon import BENCH_A2, FMT_EDF, TOOL_PLOTLY, OVL_OFF, CACHE_WARM



def load_edf_segment(
    path: Path,
    load_start_s: float,
    load_duration_s: float,
    n_channels_max: int,
    max_points_per_trace: int,
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
    with pyedflib.EdfReader(str(path)) as f:
        labels = [str(x).strip() for x in f.getSignalLabels()]
        ns_all = [int(x) for x in f.getNSamples()]
        fs_all = [float(f.getSampleFrequency(i)) for i in range(len(labels))]

        if not fs_all:
            raise RuntimeError(f"No signals found in EDF: {path}")

        fs0 = fs_all[0]

        cand: List[int] = []
        for i, (lab, fs_i, ns_i) in enumerate(zip(labels, fs_all, ns_all)):
            if ns_i <= 1 or fs_i <= 0:
                continue
            if "annot" in lab.lower():
                continue
            if abs(fs_i - fs0) > 1e-6:
                continue
            cand.append(i)

        # Fallback if filtering removed everything
        if not cand:
            cand = [
                i
                for i, (fs_i, ns_i) in enumerate(zip(fs_all, ns_all))
                if fs_i > 0 and ns_i > 1
            ]

        if not cand:
            raise RuntimeError(f"No usable numeric channels found in EDF: {path}")

        cand = cand[: max(1, int(n_channels_max))]

        n_min_total = min(ns_all[i] for i in cand)
        total_dur = n_min_total / fs0

        start = max(0.0, min(float(load_start_s), max(0.0, total_dur - 1e-6)))
        start_samp = int(math.floor(start * fs0))

        n_avail = n_min_total - start_samp
        if n_avail <= 1:
            raise ValueError(
                f"Requested start={start:.6f}s (sample {start_samp}) leaves no data. "
                f"total_dur={total_dur:.3f}s, n_min_total={n_min_total}, fs0={fs0}"
            )

        max_dur = n_avail / fs0
        dur = min(float(load_duration_s), max_dur)

        n_samples = min(int(math.floor(dur * fs0)), n_avail)
        if n_samples <= 1:
            raise ValueError(
                f"Computed n_samples={n_samples} (dur={dur:.6f}s) is too small. "
                f"start={start:.6f}s, total_dur={total_dur:.3f}s"
            )

        data_np = np.zeros((len(cand), n_samples), dtype=np.float32)
        for out_ch, ch in enumerate(cand):
            sig = np.asarray(f.readSignal(ch, start_samp, n_samples), dtype=np.float32)
            if sig.size == 0:
                raise ValueError(
                    f"readSignal returned 0 samples for ch={ch} label='{labels[ch]}' "
                    f"(start_samp={start_samp}, n_samples={n_samples})."
                )
            if sig.size < n_samples:
                sig = np.pad(sig, (0, n_samples - sig.size), mode="edge")

            # normalize (not part of interaction timing)
            sig = sig - float(sig.mean())
            sd = float(sig.std()) or 1.0
            sig = sig / sd
            data_np[out_ch, :] = sig

    times_np = (np.arange(n_samples, dtype=np.float32) / fs0) + float(start)

    # Decimate to keep HTML size reasonable.
    decim = 1
    if max_points_per_trace and max_points_per_trace > 0 and times_np.size > max_points_per_trace:
        decim = int(math.ceil(times_np.size / max_points_per_trace))
        times_np = times_np[::decim]
        data_np = data_np[:, ::decim]

    times = times_np.astype(float).tolist()
    data = [data_np[i, :].astype(float).tolist() for i in range(data_np.shape[0])]
    return times, data, fs0, int(data_np.shape[0]), float(start), float(dur), int(decim)


def render_html(
    *,
    times: List[float],
    data: List[List[float]],
    meta: dict,
    window_s: float,
    target_interval_ms: float,
    steps: int,
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

  const sequences = ["PAN_60S", "ZOOM_IN", "ZOOM_OUT", "PAN_ZOOM"];
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
    ap.add_argument("--edf", type=str, required=True)
    ap.add_argument("--n-channels", type=int, default=8)
    ap.add_argument("--load-start", type=float, default=0.0)
    ap.add_argument("--load-duration", type=float, default=75.0)
    ap.add_argument(
        "--max-points-per-trace",
        type=int,
        default=20000,
        help="Decimate each trace to at most this many points (keeps HTML responsive).",
    )
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--target-interval-ms", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="edf_A2")
    ap.add_argument("--overlay-state", type=str, default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)

    args = ap.parse_args()

    out_base = out_dir(args.out_root, BENCH_A2, TOOL_PLOTLY, args.tag)
    write_manifest(out_base, BENCH_A2, TOOL_PLOTLY, vars(args), extra={'format': FMT_EDF})
    out_html = out_base / 'plotly_A2_interactions.html'


    edf_path = args.data_dir / args.edf
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    times, data, fs, n_ch, start, dur, decim = load_edf_segment(
        edf_path,
        args.load_start,
        args.load_duration,
        args.n_channels,
        args.max_points_per_trace,
    )

    meta = {
        "bench_id": BENCH_A2,
        "tool": TOOL_PLOTLY,
        "format": FMT_EDF,
        "overlay_state": args.overlay_state,
        "cache_state": args.cache_state,
        "edf": edf_path.name,
        "fs_hz": fs,
        "n_ch": n_ch,
        "start_s": start,
        "dur_s": dur,
        "decim_factor": decim,
        "n_points_per_trace": len(times),
        "total_points": len(times) * n_ch,
        "window_s": args.window_s,
        "target_interval_ms": args.target_interval_ms,
        "steps": args.steps,
    }

    render_html(
        times=times,
        data=data,
        meta=meta,
        window_s=args.window_s,
        target_interval_ms=args.target_interval_ms,
        steps=args.steps,
        out_path=out_html,
    )
    print(f"Wrote {out_html} (open in browser; console prints BENCH_JSON).")  # noqa: T201


if __name__ == "__main__":
    main()
