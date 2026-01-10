# benchmark_pack

A minimal, reproducible folder structure for EEG benchmarking scripts with consistent outputs.

## Preconditions

- Put EDF files under `./data/` (this pack does not ship data).
- Run commands from the project root (so `benchkit` imports resolve).

## PyQtGraph A1/A2 hang repro + fix note

**What was happening:** PyQtGraph A1/A2 could appear to stall during pan/zoom sequences if the UI stopped emitting paint events; the job would keep running without output, and the orchestrator would wait indefinitely.  
**What changed:** A1/A2 now log each phase, emit a 1s heartbeat, enforce per-step and per-run timeouts, and `run_all.py` will terminate jobs that stop emitting output (with a tail snippet for debugging). A1/A2 treat zoom saturation as a NOOP step so ZOOM_IN/ZOOM_OUT runs complete without timing out when the range stops changing.

Minimal repro command (single job, ZOOM_IN). Replace `__DATA_FILE__` with your EDF path:

```powershell
python scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py --format EDF --file "__DATA_FILE__" --tag repro --window-s 60 --load-duration-s 60 --n-ch 8 --sequence ZOOM_IN --steps 60
```

```bat
python scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py --format EDF --file "__DATA_FILE__" --tag repro --window-s 60 --load-duration-s 60 --n-ch 8 --sequence ZOOM_IN --steps 60
```

## Quick commands

## Quick setup (Windows PowerShell copy/paste)

Replace the placeholders once (Ctrl+H):
- `__DATA_FILE__` → full path to your EDF/NWB file
- `__REPO__` → full path to the repo root (if you want to paste from anywhere)
- `__TAG__` → run label (keeps outputs separate)

> ⚠️ The block below must be pasted into **PowerShell**, not `cmd.exe`.

```powershell
cd "__REPO__"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r scripts\desktop\requirements-desktop.txt

$FILE = "__DATA_FILE__"
$WINDOWS = @(60, 600, 1800) # 1, 10, 30 minutes in seconds
$CHANNELS = @(8, 16, 32, 64)
$TAG = "__TAG__"

foreach ($w in $WINDOWS) {
  foreach ($ch in $CHANNELS) {
    # VisPy: TFFR, A1, A2
    python scripts\desktop\bench_vispy.py --bench-id TFFR --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\desktop\bench_vispy.py --bench-id A1_THROUGHPUT --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\desktop\bench_vispy.py --bench-id A2_CADENCED --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch

    # Datoviz: TFFR, A1, A2
    python scripts\desktop\bench_datoviz.py --bench-id TFFR --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\desktop\bench_datoviz.py --bench-id A1_THROUGHPUT --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\desktop\bench_datoviz.py --bench-id A2_CADENCED --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch

    # PyQtGraph: TFFR, A1, A2
    python scripts\desktop\bench_pyqtgraph_tffr_v2.py --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\desktop\bench_pyqtgraph_A2_cadenced_v2.py --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch

    # D3.js (HTML generator)
    python scripts\web\gen_d3_html.py --bench-id TFFR --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\web\gen_d3_html.py --bench-id A1_THROUGHPUT --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch
    python scripts\web\gen_d3_html.py --bench-id A2_CADENCED --format EDF --file "$FILE" --tag $TAG --window-s $w --load-duration-s $w --n-ch $ch --target-interval-ms 16
  }
}
```

## Quick setup (Windows cmd.exe copy/paste)

Replace the placeholders once (Ctrl+H):
- `__DATA_FILE__` → full path to your EDF/NWB file
- `__REPO__` → full path to the repo root (if you want to paste from anywhere)
- `__TAG__` → run label (keeps outputs separate)

```bat
cd "__REPO__"
python -m venv .venv
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r scripts\desktop\requirements-desktop.txt

set FILE=__DATA_FILE__
set TAG=__TAG__

for %%w in (60 600 1800) do (
  for %%c in (8 16 32 64) do (
    rem VisPy: TFFR, A1, A2
    python scripts\desktop\bench_vispy.py --bench-id TFFR --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\desktop\bench_vispy.py --bench-id A1_THROUGHPUT --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\desktop\bench_vispy.py --bench-id A2_CADENCED --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c

    rem Datoviz: TFFR, A1, A2
    python scripts\desktop\bench_datoviz.py --bench-id TFFR --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\desktop\bench_datoviz.py --bench-id A1_THROUGHPUT --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\desktop\bench_datoviz.py --bench-id A2_CADENCED --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c

    rem PyQtGraph: TFFR, A1, A2
    python scripts\desktop\bench_pyqtgraph_tffr_v2.py --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\desktop\bench_pyqtgraph_A2_cadenced_v2.py --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c

    rem D3.js (HTML generator)
    python scripts\web\gen_d3_html.py --bench-id TFFR --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\web\gen_d3_html.py --bench-id A1_THROUGHPUT --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c
    python scripts\web\gen_d3_html.py --bench-id A2_CADENCED --format EDF --file "%FILE%" --tag %TAG% --window-s %%w --load-duration-s %%w --n-ch %%c --target-interval-ms 16
  )
)
```

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

## Synthetic benchmark datasets

Some EDF/NWB files ship with fewer than 64 channels, which can make `--n-ch 64` misleading if the loader caps to the available channels. Use the manual generator to create deterministic synthetic datasets with ≥64 channels and a full 30-minute duration.

Generate datasets (manual only; nothing auto-generates):

```bash
python scripts/generate_synth_data.py --n-ch 64 --duration-s 1800 --fs-hz 250
```

This creates:

- `data/synth_64ch_1800s_<fs>hz.edf`
- `data/synth_64ch_1800s_<fs>hz.nwb`

Recommended `--fs-hz` defaults:
- 250 Hz keeps file sizes reasonable for 64 channels × 30 minutes.
- Higher sampling rates scale file size and generation time linearly.

## Lexicon and status

See:
- `docs/LEXICON.md`
- `docs/STATUS.md`
- `docs/ARCHITECTURE.md`
