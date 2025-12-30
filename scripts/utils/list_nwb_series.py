from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _iter_series(nwbfile: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for name, obj in getattr(nwbfile, "acquisition", {}).items():
        out.append((f"acquisition/{name}", obj))

    processing = getattr(nwbfile, "processing", {})
    for mod_name, mod in processing.items():
        try:
            data_ifaces = getattr(mod, "data_interfaces", {})
        except Exception:
            data_ifaces = {}
        for name, obj in data_ifaces.items():
            out.append((f"processing/{mod_name}/{name}", obj))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="List candidate 2D TimeSeries objects in an NWB file.")
    p.add_argument("nwb_path", type=Path)
    args = p.parse_args()

    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(args.nwb_path), mode="r") as io:
        nwbfile = io.read()
        rows: List[Dict[str, str]] = []
        for path, obj in _iter_series(nwbfile):
            if not hasattr(obj, "data"):
                continue
            try:
                shape = tuple(getattr(obj.data, "shape", ()))
            except Exception:
                shape = ()
            if len(shape) != 2:
                continue
            rate = getattr(obj, "rate", None) or getattr(obj, "sampling_rate", None)
            has_ts = getattr(obj, "timestamps", None) is not None

            rows.append({
                "series_path": path,
                "shape": str(shape),
                "rate": str(rate) if rate is not None else "",
                "timestamps": "yes" if has_ts else "no",
                "type": type(obj).__name__,
            })

    if not rows:
        print("No obvious 2D TimeSeries candidates found (acquisition/ or processing/).")
        return

    widths = {k: max(len(k), max(len(r[k]) for r in rows)) for k in rows[0].keys()}
    header = " | ".join(k.ljust(widths[k]) for k in rows[0].keys())
    sep = "-+-".join("-" * widths[k] for k in rows[0].keys())
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(r[k].ljust(widths[k]) for k in rows[0].keys()))


if __name__ == "__main__":
    main()
