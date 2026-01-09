from __future__ import annotations

import importlib.util
import os
from typing import Dict, Tuple

from benchkit.lexicon import (
    TOOL_DATOVIZ,
    TOOL_D3,
    TOOL_IO_MNE,
    TOOL_IO_NEO,
    TOOL_IO_PYEDFLIB,
    TOOL_PG,
    TOOL_PLOTLY,
    TOOL_VISPY,
)


_TOOL_IMPORTS: Dict[str, Tuple[str, ...]] = {
    TOOL_PG: ("pyqtgraph", "PyQt5"),
    TOOL_VISPY: ("vispy",),
    TOOL_DATOVIZ: ("datoviz",),
    TOOL_IO_PYEDFLIB: ("pyedflib",),
    TOOL_IO_MNE: ("mne",),
    TOOL_IO_NEO: ("neo",),
    TOOL_PLOTLY: (),
    TOOL_D3: (),
}


def _is_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_tool(tool_id: str) -> Tuple[bool, str]:
    skip_list = os.environ.get("BENCHKIT_SKIP_TOOLS", "")
    if skip_list:
        tokens = {t.strip() for t in skip_list.replace(";", ",").split(",") if t.strip()}
        if tool_id in tokens:
            return False, f"Skipped via BENCHKIT_SKIP_TOOLS={skip_list}"

    for module in _TOOL_IMPORTS.get(tool_id, ()):  # pragma: no branch - simple lookup
        if not _is_available(module):
            return False, f"Missing dependency: {module}"

    return True, "ok"


def check_format_dependency(fmt: str) -> Tuple[bool, str]:
    if fmt == "EDF":
        if not _is_available("pyedflib"):
            return False, "Missing dependency: pyedflib"
    if fmt == "NWB":
        if not _is_available("pynwb"):
            return False, "Missing dependency: pynwb"
    return True, "ok"


def playwright_available() -> bool:
    return _is_available("playwright")
