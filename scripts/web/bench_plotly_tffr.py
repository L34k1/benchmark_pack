#!/usr/bin/env python3
"""
scripts/web/bench_plotly_tffr.py

Generates an HTML file for a Plotly WebGL plot and logs a JSON payload to the browser console:
    BENCH_JSON: {...}

The JSON includes:
- meta (file, fs, n_channels, window_s, decimation)
- first_render_ms (time until Plotly emits 'plotly_afterplot' + one rAF)

Outputs:
  outputs/TFFR/VIS_PLOTLY/<tag>/plotly_tffr.html
  outputs/TFFR/VIS_PLOTLY/<tag>/manifest.json

Optional:
  Use scripts/web/collect_plotly_console_playwright.py to capture BENCH_JSON automatically
  (requires playwright + a Chromium installation).
"""
from __future__ import annotations

import sys

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyedflib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_manifest
from benchkit.lexicon import BENCH_TFFR, FMT_EDF, TOOL_PLOTLY, OVL_OFF, CACHE_WARM


def load_edf_window(path: Path, n_channels: int, window_s: float, max_points_per_trace: int) -> Tuple[List[float], List[List[float]], float, int, int]:
    f = pyedflib.EdfReader(str(path))
    fs = float(f.getSampleFrequency(0))
    n_available = int(f.signals_in_file)
    n_channels = min(int(n_channels), n_available)

    n_samples = min(int(window_s * fs), int(f.getNSamples()[0]))

    # Uniform decimation to cap points per trace
    decim = max(1, int(np.ceil(n_samples / max_points_per_trace)))
    idx = np.arange(0, n_samples, decim, dtype=np.int64)

    times = (idx / fs).astype(np.float64)

    data: List[List[float]] = []
    for ch in range(n_channels):
        sig = f.readSignal(ch, start=0, n=n_samples).astype(np.float64)
        data.append(sig[idx].tolist())
    f.close()

    return times.tolist(), data, fs, n_channels, decim


def write_html(out_html: Path, times: List[float], data: List[List[float]], meta: dict) -> None:
    # Keep HTML self-contained (CDN for Plotly).
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Plotly TFFR Bench</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; }}
    #gd {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
<div id="gd"></div>
<script>
const META = {json.dumps(meta)};
const TIMES = {json.dumps(times)};
const DATA = {json.dumps(data)};

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

const layout = {{
  margin: {{l: 40, r: 20, t: 20, b: 30}},
  xaxis: {{ title: "Time (s)", showgrid: false }},
  yaxis: {{ showgrid: false }},
  showlegend: false
}};

const config = {{
  responsive: true,
  scrollZoom: true,
  displayModeBar: true
}};

(async () => {{
  const gd = document.getElementById("gd");
  const t0 = performance.now();

  // Wait for first afterplot and one rAF to approximate a "presented" frame.
  gd.on('plotly_afterplot', async () => {{
    await new Promise(requestAnimationFrame);
    const t1 = performance.now();
    const out = {{
      meta: META,
      first_render_ms: t1 - t0
    }};
    console.log("BENCH_JSON:", JSON.stringify(out));
  }});

  await Plotly.newPlot(gd, makeTraces(), layout, config);
}})().catch(e => console.error(e));
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--edf", type=str, required=True, help="EDF filename inside --data-dir.")
    ap.add_argument("--n-channels", type=int, default=8)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--max-points-per-trace", type=int, default=20000)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="edf_tffr")
    ap.add_argument("--overlay-state", type=str, default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)
    args = ap.parse_args()

    edf_path = args.data_dir / args.edf
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    out_base = out_dir(args.out_root, BENCH_TFFR, TOOL_PLOTLY, args.tag)
    write_manifest(out_base, BENCH_TFFR, TOOL_PLOTLY, vars(args), extra={"format": FMT_EDF})
    out_html = out_base / "plotly_tffr.html"

    times, data, fs, n_ch, decim = load_edf_window(edf_path, args.n_channels, args.window_s, args.max_points_per_trace)
    meta = {
        "bench_id": BENCH_TFFR,
        "tool": TOOL_PLOTLY,
        "format": FMT_EDF,
        "file": edf_path.name,
        "fs_hz": fs,
        "n_channels": n_ch,
        "window_s": args.window_s,
        "decim_factor": decim,
        "n_points_per_trace": len(times),
        "total_points": len(times) * n_ch,
        "overlay_state": args.overlay_state,
        "cache_state": args.cache_state,
    }
    write_html(out_html, times, data, meta)
    print(f"Wrote {out_html}. Open it and copy the console line starting with BENCH_JSON.")  # noqa: T201


if __name__ == "__main__":
    main()
