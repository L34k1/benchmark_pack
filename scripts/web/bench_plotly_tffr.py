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
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_manifest
from benchkit.bench_defaults import DEFAULT_WINDOW_S
from benchkit.lexicon import BENCH_TFFR, FMT_EDF, FMT_NWB, TOOL_PLOTLY, OVL_OFF, CACHE_WARM
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb


def load_window(
    path: Path,
    fmt: str,
    n_channels: int,
    window_s: float,
    max_points_per_trace: int,
    nwb_series_path: str | None,
    nwb_time_dim: str,
) -> Tuple[List[float], List[List[float]], float, int, int]:
    if fmt == FMT_EDF:
        seg = load_edf_segment_pyedflib(path, 0.0, window_s, n_channels)
    elif fmt == FMT_NWB:
        seg = load_nwb_segment_pynwb(
            path,
            0.0,
            window_s,
            n_channels,
            series_path=nwb_series_path,
            time_dim=nwb_time_dim,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    times, data, decim = decimate_for_display(seg.times_s, seg.data, max_points_per_trace)
    times_list = times.astype(np.float64).tolist()
    data_list = data.astype(np.float64).tolist()
    return times_list, data_list, float(seg.fs_hz), int(data.shape[0]), int(decim)


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
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], default=FMT_EDF)
    ap.add_argument("--edf", type=str, default=None, help="EDF/NWB filename inside --data-dir.")
    ap.add_argument("--file", type=Path, default=None, help="Full path to EDF/NWB file.")
    ap.add_argument("--n-channels", "--n-ch", dest="n_channels", type=int, default=8)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--max-points-per-trace", type=int, default=20000)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="edf_tffr")
    ap.add_argument("--overlay-state", "--overlay", dest="overlay_state", type=str, default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    args = ap.parse_args()

    data_path = args.file or (args.data_dir / args.edf if args.edf else None)
    if data_path is None:
        raise SystemExit("Provide --file or --edf with --data-dir.")
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    out_base = out_dir(args.out_root, BENCH_TFFR, TOOL_PLOTLY, args.tag)
    write_manifest(out_base, BENCH_TFFR, TOOL_PLOTLY, vars(args), extra={"format": args.format})
    out_html = out_base / "plotly_tffr.html"

    times, data, fs, n_ch, decim = load_window(
        data_path,
        args.format,
        args.n_channels,
        args.window_s,
        args.max_points_per_trace,
        args.nwb_series_path,
        args.nwb_time_dim,
    )
    meta = {
        "bench_id": BENCH_TFFR,
        "tool": TOOL_PLOTLY,
        "format": args.format,
        "file": data_path.name,
        "fs_hz": fs,
        "n_channels": n_ch,
        "effective_n_ch": n_ch,
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
