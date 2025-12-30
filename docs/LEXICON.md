# Lexicon

This project uses short, stable keywords to reduce ambiguity and compress reporting.

## Benchmarks

- **IO**: Data access benchmark (open + slice reads). Focus: file I/O and decoding cost.
- **TFFR**: *Time To First Render* (initial plot displayed).
- **A1_THROUGHPUT** (**A1**): Throughput-only interaction benchmark (back-to-back viewport updates; no cadence).
- **A2_CADENCED** (**A2**): Cadenced interaction benchmark (target update interval; measure lateness/jank proxies).

## Formats

- **EDF**: European Data Format (EDF/EDF+).
- **NWB**: Neurodata Without Borders.

## Tools (canonical IDs)

- **IO_PYEDFLIB**: EDF access via *pyEDFlib*.
- **IO_MNE**: EDF access via *MNE-Python*.
- **IO_NEO**: EDF access via *Neo* (EDF IO).
- **VIS_PYQTGRAPH**: Desktop visualization via *PyQtGraph*.
- **VIS_PLOTLY**: Browser visualization via *Plotly* (ScatterGL).

## Interaction sequences

- **PAN**
- **ZOOM_IN**
- **ZOOM_OUT**
- **PAN_ZOOM**

## Conditions

- **OVL_OFF / OVL_ON**: Overlay disabled/enabled (annotations/predictions).
- **CACHE_COLD / CACHE_WARM**: Cold/warm cache condition (explicitly recorded when applicable).

## Scenario key (recommended notation)

A scenario can be referenced as:

`<FMT>_W<window_s>_C<n_channels>_<SEQ>_<OVL>_<CACHE>`

Example:
`EDF_W600_C32_PAN_OVL_OFF_CACHE_WARM`
