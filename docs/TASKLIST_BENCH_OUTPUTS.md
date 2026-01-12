# Benchmark Output Contract Tasklist

Checklist of output-contract gaps to close for each tool and bench type. Each task is scoped to one tool × bench type and must emit the standard artifacts (manifest.json + steps.csv or tffr.csv + summary.json) and pass the smoke command.

## Desktop tools

- [ ] **VIS_PYQTGRAPH · TFFR** — write `tffr.csv`, `summary.json`, and `manifest.json` with required fields.  
  **Schema:** `tffr.csv` columns `run_id,tffr_ms`; summary includes `tffr_ms`.  
  **Smoke:** `python scripts/desktop/bench_pyqtgraph_tffr_v2.py --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_pg_tffr`

- [ ] **VIS_PYQTGRAPH · A1_THROUGHPUT** — write `steps.csv` (200 rows, `latency_ms>=0`, `noop` where needed), `summary.json`, `manifest.json`.  
  **Schema:** `steps.csv` columns include `step_id,latency_ms,noop,status`.  
  **Smoke:** `python scripts/desktop/bench_pyqtgraph_A1_throughput_v2.py --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_pg_a1`

- [ ] **VIS_PYQTGRAPH · A2_CADENCED** — write `steps.csv` (200 rows), `summary.json` (p50/p95/max + noop/failed counts), `manifest.json`.  
  **Schema:** `steps.csv` columns include `step_id,latency_ms,noop,status`.  
  **Smoke:** `python scripts/desktop/bench_pyqtgraph_A2_cadenced_v2.py --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_pg_a2`

- [ ] **VIS_VISPY · TFFR** — write `tffr.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_vispy.py --bench-id TFFR --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_vispy_tffr`

- [ ] **VIS_VISPY · A1_THROUGHPUT** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_vispy.py --bench-id A1_THROUGHPUT --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_vispy_a1`

- [ ] **VIS_VISPY · A2_CADENCED** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_vispy.py --bench-id A2_CADENCED --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_vispy_a2`

- [ ] **VIS_DATOVIZ · TFFR** — write `tffr.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_datoviz.py --bench-id TFFR --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_dvz_tffr`

- [ ] **VIS_DATOVIZ · A1_THROUGHPUT** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_datoviz.py --bench-id A1_THROUGHPUT --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_dvz_a1`

- [ ] **VIS_DATOVIZ · A2_CADENCED** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_datoviz.py --bench-id A2_CADENCED --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_dvz_a2`

- [ ] **VIS_FASTPLOTLIB · TFFR** — write `tffr.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_fastplotlib.py --bench-id TFFR --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_fpl_tffr`

- [ ] **VIS_FASTPLOTLIB · A1_THROUGHPUT** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_fastplotlib.py --bench-id A1_THROUGHPUT --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_fpl_a1`

- [ ] **VIS_FASTPLOTLIB · A2_CADENCED** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_fastplotlib.py --bench-id A2_CADENCED --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_fpl_a2`

- [ ] **VIS_MNE_RAWPLOT · TFFR** — write `tffr.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_mne_rawplot.py --bench-id TFFR --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_mne_tffr`

- [ ] **VIS_MNE_RAWPLOT · A1_THROUGHPUT** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_mne_rawplot.py --bench-id A1_THROUGHPUT --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_mne_a1`

- [ ] **VIS_MNE_RAWPLOT · A2_CADENCED** — write `steps.csv`, `summary.json`, `manifest.json`.  
  **Smoke:** `python scripts/desktop/bench_mne_rawplot.py --bench-id A2_CADENCED --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_mne_a2`

## Web tools

- [ ] **VIS_PLOTLY · TFFR** — write `tffr.csv`, `summary.json`, `manifest.json` via collector.  
  **Smoke:** `python scripts/web/bench_plotly_tffr.py --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_plotly_tffr`

- [ ] **VIS_PLOTLY · A1_THROUGHPUT** — write `steps.csv`, `summary.json`, `manifest.json` via collector.  
  **Smoke:** `python scripts/web/bench_plotly_A1_throughput.py --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_plotly_a1`

- [ ] **VIS_PLOTLY · A2_CADENCED** — write `steps.csv`, `summary.json`, `manifest.json` via collector.  
  **Smoke:** `python scripts/web/bench_plotly_A2_cadenced.py --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_plotly_a2`

- [ ] **VIS_D3_CANVAS · TFFR** — write `tffr.csv`, `summary.json`, `manifest.json` via collector.  
  **Smoke:** `python scripts/web/gen_d3_html.py --bench-id TFFR --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_d3_tffr`

- [ ] **VIS_D3_CANVAS · A1_THROUGHPUT** — write `steps.csv`, `summary.json`, `manifest.json` via collector.  
  **Smoke:** `python scripts/web/gen_d3_html.py --bench-id A1_THROUGHPUT --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --out-root outputs --tag smoke_d3_a1`

- [ ] **VIS_D3_CANVAS · A2_CADENCED** — write `steps.csv`, `summary.json`, `manifest.json` via collector.  
  **Smoke:** `python scripts/web/gen_d3_html.py --bench-id A2_CADENCED --format EDF --file data/_synth/synth_8ch_60s_250hz.edf --window-s 60 --n-ch 8 --sequence PAN --steps 200 --runs 1 --target-interval-ms 16.666 --out-root outputs --tag smoke_d3_a2`
