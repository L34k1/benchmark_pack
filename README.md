# benchmark_pack

A minimal, reproducible folder structure for EEG benchmarking scripts with consistent outputs.

## Preconditions

- Put EDF files under `./data/` (this pack does not ship data).
- Run commands from the project root (so `benchkit` imports resolve).

## Quick commands

### IO (EDF)
```bash
python -m scripts.io.bench_io --data-dir data --n-files 5 --runs 5 --windows 60,600,1800 --n-channels 64 --tag edf_io
```

### PyQtGraph — TFFR
```bash
python -m scripts.desktop.bench_pyqtgraph_tffr --data-dir data --n-files 5 --runs 3 --window-s 10 --n-channels 64 --tag edf_tffr
```

### PyQtGraph — A1 throughput-only
```bash
python -m scripts.desktop.bench_pyqtgraph_A1_throughput --data-dir data --n-files 5 --runs 3 --window-s 60 --n-channels 64 --tag edf_A1
```

### PyQtGraph — A2 cadenced
```bash
python -m scripts.desktop.bench_pyqtgraph_A2_cadenced --data-dir data --n-files 5 --runs 3 --window-s 60 --n-channels 64 --target-interval-ms 16 --tag edf_A2
```

### Plotly — A1/A2 (HTML generators)
```bash
python -m scripts.web.bench_plotly_A1_throughput --data-dir data --edf your_file.edf --window-s 60 --n-channels 64 --tag edf_A1
python -m scripts.web.bench_plotly_A2_cadenced    --data-dir data --edf your_file.edf --window-s 60 --n-channels 64 --target-interval-ms 16 --tag edf_A2
```

Open the generated HTML and copy the `BENCH_JSON` line from the browser console.

Optional headless capture (requires Playwright):
```bash
python -m scripts.web.collect_plotly_console_playwright --html outputs/A2_CADENCED/VIS_PLOTLY/edf_A2/plotly_A2_interactions.html
```

## Output locations

All outputs are written under `./outputs/<BENCH_ID>/<TOOL_ID>/<TAG>/`.

Each run writes a `manifest.json` to record arguments and environment metadata.

## Lexicon and status

See:
- `docs/LEXICON.md`
- `docs/STATUS.md`
- `docs/ARCHITECTURE.md`
