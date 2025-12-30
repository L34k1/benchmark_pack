# Project architecture (canonical)

The objective is to make reruns reproducible and outputs comparable.

## Folder layout

```
benchmark_pack/
  benchkit/                 # shared utilities and lexicon
  scripts/
    io/                     # IO benchmarks
    desktop/                # desktop visualization benchmarks
    web/                    # browser visualization benchmarks (+ collectors)
  configs/                  # (optional) scenario matrices / run presets
  outputs/                  # generated outputs (kept out of version control)
  data/                     # dataset folder (user-provided; not shipped)
  docs/                     # lexicon + status + methodology notes
```

## Output contract

All scripts write under:

`outputs/<BENCH_ID>/<TOOL_ID>/<TAG>/...`

This keeps benchmarks separated while preserving consistent discovery paths.

Examples:
- `outputs/IO/IO_STACKS/edf_io/io_raw.csv`
- `outputs/TFFR/VIS_PYQTGRAPH/edf_tffr/pyqtgraph_tffr.csv`
- `outputs/A2_CADENCED/VIS_PYQTGRAPH/edf_A2/pyqtgraph_A2_steps.csv`

Each output directory contains:
- `manifest.json`: arguments + environment info for reproducibility
- One or more CSV/HTML artifacts (bench-specific)

## Keyword discipline

- Use **tool IDs** and **bench IDs** as the *only* stable identifiers in filenames and tables.
- Use scenario keys (e.g., `EDF_W60_C16_PAN_OVL_OFF_CACHE_WARM`) in analysis/figures, not in code-level branching.

## Intended separation of concerns

- `benchkit/lexicon.py`: defines all names (A1/A2/IO/TFFR, tools, sequences).
- `benchkit/common.py`: output paths + run manifest helpers.
- `scripts/*`: contain only the per-tool logic and emit the contract outputs.
