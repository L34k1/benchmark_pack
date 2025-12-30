# Status (What was done vs what remains)

This status is based on the files you provided in this conversation (IO CSV, PyQtGraph CSVs, and Plotly logs). It is therefore a *best-effort reconstruction*, not a guarantee that nothing else was executed elsewhere.

## Done

### IO (EDF)
- **IO** executed for **EDF** on 5 files (**S1..S5**), for windows **60s / 600s / 1800s**.
- Libraries and profiles covered:
  - **IO_PYEDFLIB**: lazy + preload
  - **IO_MNE**: lazy + preload
  - **IO_NEO**: lazy + preload

### VIS_PYQTGRAPH
- **TFFR** executed for EDF (window **10s**) on 5 files.
- **A2_CADENCED** executed for EDF (target interval **16 ms**) on 5 files, sequences:
  - **PAN**, **ZOOM_IN**, **ZOOM_OUT**, **PAN_ZOOM**
- Current A2 coverage is limited to **window 10s**, and channel-count coverage is unclear in the logged CSV (the script likely used “all available channels up to a cap”, but it is not recorded).

### VIS_PLOTLY
- Evidence of at least one **A2_CADENCED** run exists (console log), but the JSON is not reliably parseable from the provided log, so it is not counted as a completed, reproducible dataset.

## Remaining (minimum to satisfy CDC)

### Format coverage
- **NWB**: implement + run **IO**, **TFFR**, **A1**, **A2** for NWB (tool-by-tool).

### Window and channel matrix
For each tool and format:
- Windows: **60s / 600s / 1800s**
- Channels: **8 / 16 / 32 / 64**

### Benchmark phases
- **A1_THROUGHPUT**: not evidenced in the provided results (must be run).
- **A2_CADENCED**: currently partial (PyQtGraph only, 10s window; Plotly not captured cleanly).

### Overlays
- **OVL_OFF vs OVL_ON**: not evidenced; requires at least one clearly defined overlay condition and its performance impact.

### Consistency / metadata
- Ensure every output row records:
  - bench_id, tool_id, format, window_s, n_channels, sequence, overlay_state, cache_state, file_id, run_id, step_id
