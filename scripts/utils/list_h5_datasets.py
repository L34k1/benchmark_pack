from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import h5py


def main() -> None:
    p = argparse.ArgumentParser(description="List 2D datasets in an HDF5/NWB file (h5py).")
    p.add_argument("h5_path", type=Path)
    p.add_argument("--min-time", type=int, default=1000)
    args = p.parse_args()

    rows: List[Tuple[str, Tuple[int, ...], str]] = []
    with h5py.File(str(args.h5_path), "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                shape = tuple(int(x) for x in obj.shape)
                if max(shape) >= int(args.min_time):
                    rows.append((name, shape, str(obj.dtype)))
        f.visititems(visitor)

    if not rows:
        print("No matching 2D datasets found.")
        return

    rows.sort(key=lambda x: (max(x[1]), x[0]), reverse=True)
    for name, shape, dtype in rows[:200]:
        print(f"{name}\tshape={shape}\tdtype={dtype}")


if __name__ == "__main__":
    main()
