"""
Project lexicon constants.

A1 = Throughput-only (back-to-back updates, no cadence constraint)
A2 = Cadenced interaction (target interval, measure lateness/jank proxies)
"""

# Formats
FMT_EDF = "EDF"
FMT_NWB = "NWB"

# Overlay state
OVL_OFF = "OVL_OFF"
OVL_ON  = "OVL_ON"

# Cache state (operational; may be WARM only on some OS)
CACHE_WARM = "CACHE_WARM"
CACHE_COLD = "CACHE_COLD"

# Bench IDs
BENCH_IO   = "IO"
BENCH_TFFR = "TFFR"
BENCH_A1   = "A1_THROUGHPUT"
BENCH_A2   = "A2_CADENCED"

# Tools (canonical IDs)
# I/O stacks
TOOL_IO_PYEDFLIB = "IO_PYEDFLIB"
TOOL_IO_MNE      = "IO_MNE"
TOOL_IO_NEO      = "IO_NEO"
TOOL_IO_PYNWB    = "IO_PYNWB"
TOOL_IO_H5PY     = "IO_H5PY"

# Visualization stacks
TOOL_PG     = "VIS_PYQTGRAPH"
TOOL_PLOTLY = "VIS_PLOTLY"

# Additional visualization tools
TOOL_VISPY = "VIS_VISPY"
TOOL_D3 = "VIS_D3_CANVAS"
TOOL_FASTPLOTLIB = "VIS_FASTPLOTLIB"
TOOL_MNE_RAWPLOT = "VIS_MNE_RAWPLOT"

# Interaction sequences
SEQ_PAN      = "PAN"
SEQ_ZOOM_IN  = "ZOOM_IN"
SEQ_ZOOM_OUT = "ZOOM_OUT"
SEQ_PAN_ZOOM = "PAN_ZOOM"
