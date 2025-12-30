from __future__ import annotations

import sys

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import ensure_dir, env_info, out_dir, write_json, write_manifest
from benchkit.lexicon import (
    BENCH_IO,
    FMT_EDF,
    FMT_NWB,
    TOOL_IO_H5PY,
    TOOL_IO_MNE,
    TOOL_IO_NEO,
    TOOL_IO_PYEDFLIB,
    TOOL_IO_PYNWB,
)
from benchkit.loaders import load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.stats import summarize_latency_ms


def _list_files(data_dir: Path, fmt: str, n_files: int) -> List[Path]:
    exts = {FMT_EDF: (".edf",), FMT_NWB: (".nwb", ".h5", ".hdf5")}[fmt]
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(data_dir.glob(f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No {fmt} files found in {data_dir} (extensions: {exts})")
    return files[: max(1, int(n_files))]


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _load_edf_mne(path: Path, start_s: float, dur_s: float, n_ch: int) -> float:
    import mne
    t0 = _now_ms()
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    picks = list(range(min(n_ch, raw.info["nchan"])))
    start = int(math.floor(start_s * fs))
    stop = int(math.floor((start_s + dur_s) * fs))
    raw.get_data(picks=picks, start=start, stop=stop, return_times=False)
    return _now_ms() - t0


def _load_edf_neo(path: Path, start_s: float, dur_s: float, n_ch: int) -> float:
    import neo
    t0 = _now_ms()
    reader_cls = getattr(neo.io, "EdfIO", None) or getattr(neo.io, "EDFIO", None)
    if reader_cls is None:
        raise AttributeError("neo.io.EdfIO/EDFIO not available in this Neo version.")
    reader = reader_cls(str(path))
    seg = reader.read_segment(lazy=False)
    sigs = [asig for asig in seg.analogsignals]
    if not sigs:
        raise RuntimeError(f"No AnalogSignal found in EDF: {path}")
    asig = sigs[0]
    fs = float(asig.sampling_rate.rescale("Hz").magnitude)
    start = int(math.floor(start_s * fs))
    stop = int(math.floor((start_s + dur_s) * fs))
    arr = np.asarray(asig.magnitude, dtype=np.float32)  # (n_samp, n_ch_total)
    _ = arr[start:stop, : min(n_ch, arr.shape[1])]
    return _now_ms() - t0


def _load_nwb_h5py(path: Path, start_s: float, dur_s: float, n_ch: int, dataset_path: Optional[str]) -> float:
    import h5py

    def score_dataset(name: str, dset: h5py.Dataset) -> int:
        if not isinstance(dset, h5py.Dataset) or dset.ndim != 2:
            return -1
        if dset.shape[0] < 10 or dset.shape[1] < 2:
            return -1
        score = 0
        ln = name.lower()
        if "electrical" in ln or "eeg" in ln or ln.split("/")[-1] == "data":
            score += 5
        if dset.shape[0] >= dset.shape[1]:
            score += 1
        return score

    t0 = _now_ms()
    with h5py.File(str(path), "r") as f:
        if dataset_path:
            dset = f[dataset_path]
            sel_name = dataset_path
        else:
            best = None

            def visitor(name, obj):
                nonlocal best
                if isinstance(obj, h5py.Dataset):
                    s = score_dataset(name, obj)
                    if s >= 0 and (best is None or s > best[0]):
                        best = (s, name)

            f.visititems(visitor)
            if best is None:
                raise RuntimeError("No suitable 2D dataset found. Provide --nwb-dataset-path.")
            sel_name = best[1]
            dset = f[sel_name]

        fs = float("nan")
        base = "/".join(sel_name.split("/")[:-1])
        if base and base in f:
            grp = f[base]
            if "rate" in grp.attrs:
                try:
                    fs = float(grp.attrs["rate"])
                except Exception:
                    pass
        if not (fs == fs) or fs <= 0:
            raise RuntimeError(
                f"Could not infer sampling rate from HDF5 for dataset {sel_name}. Use PyNWB or provide a series with rate/timestamps."
            )

        time_first = (dset.shape[0] >= dset.shape[1])
        n_time = int(dset.shape[0] if time_first else dset.shape[1])
        start = int(math.floor(start_s * fs))
        stop = min(int(math.floor((start_s + dur_s) * fs)), n_time)
        if stop - start <= 1:
            raise ValueError("Requested slice is empty or too small.")
        if time_first:
            _ = np.asarray(dset[start:stop, : min(n_ch, dset.shape[1])], dtype=np.float32)
        else:
            _ = np.asarray(dset[: min(n_ch, dset.shape[0]), start:stop], dtype=np.float32)

    return _now_ms() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="I/O benchmark for EDF and NWB backends.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--tag", type=str, required=True)

    p.add_argument("--formats", nargs="+", choices=[FMT_EDF, FMT_NWB], default=[FMT_EDF])
    p.add_argument("--tools", nargs="+", default=["all"], help="Subset of tools; or 'all'")

    p.add_argument("--n-files", type=int, default=3)
    p.add_argument("--runs", type=int, default=3)

    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--load-start-s", type=float, default=0.0)
    p.add_argument("--n-ch", type=int, default=16)

    p.add_argument("--nwb-series-path", type=str, default=None)
    p.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    p.add_argument("--nwb-dataset-path", type=str, default=None)

    args = p.parse_args()

    fmt_tools = {
        FMT_EDF: [TOOL_IO_PYEDFLIB, TOOL_IO_MNE, TOOL_IO_NEO],
        FMT_NWB: [TOOL_IO_PYNWB, TOOL_IO_H5PY],
    }

    if args.tools == ["all"]:
        tool_sel = {fmt: fmt_tools[fmt] for fmt in args.formats}
    else:
        wanted = set(args.tools)
        tool_sel = {fmt: [t for t in fmt_tools[fmt] if t in wanted] for fmt in args.formats}

    meta_base = {
        "env": env_info(),
        "window_s": float(args.window_s),
        "load_start_s": float(args.load_start_s),
        "n_ch": int(args.n_ch),
        "n_files": int(args.n_files),
        "runs": int(args.runs),
    }

    for fmt in args.formats:
        files = _list_files(args.data_dir, fmt, args.n_files)

        for tool_id in tool_sel.get(fmt, []):
            lat_ms: List[float] = []
            per_file: List[Dict[str, Any]] = []

            out = out_dir(args.out_root, BENCH_IO, tool_id, args.tag + f"_{fmt.lower()}")
            ensure_dir(out)

            extra = dict(meta_base)
            extra.update({"format": fmt, "tool_id": tool_id, "files": [str(x) for x in files]})
            if fmt == FMT_NWB:
                extra.update({
                    "nwb_series_path": args.nwb_series_path,
                    "nwb_time_dim": args.nwb_time_dim,
                    "nwb_dataset_path": args.nwb_dataset_path,
                })

            write_manifest(out, BENCH_IO, tool_id, args=vars(args), extra=extra)

            for r in range(int(args.runs)):
                for path in files:
                    if fmt == FMT_EDF and tool_id == TOOL_IO_PYEDFLIB:
                        t0 = _now_ms()
                        seg = load_edf_segment_pyedflib(path, args.load_start_s, args.window_s, args.n_ch)
                        _ = seg.data
                        dt = _now_ms() - t0

                    elif fmt == FMT_EDF and tool_id == TOOL_IO_MNE:
                        dt = _load_edf_mne(path, args.load_start_s, args.window_s, args.n_ch)

                    elif fmt == FMT_EDF and tool_id == TOOL_IO_NEO:
                        dt = _load_edf_neo(path, args.load_start_s, args.window_s, args.n_ch)

                    elif fmt == FMT_NWB and tool_id == TOOL_IO_PYNWB:
                        t0 = _now_ms()
                        seg = load_nwb_segment_pynwb(
                            path, args.load_start_s, args.window_s, args.n_ch,
                            series_path=args.nwb_series_path, time_dim=args.nwb_time_dim
                        )
                        _ = seg.data
                        dt = _now_ms() - t0

                    elif fmt == FMT_NWB and tool_id == TOOL_IO_H5PY:
                        dt = _load_nwb_h5py(path, args.load_start_s, args.window_s, args.n_ch, args.nwb_dataset_path)

                    else:
                        continue

                    lat_ms.append(float(dt))
                    per_file.append({"run": int(r), "file": str(path), "load_ms": float(dt)})

            summary = summarize_latency_ms(lat_ms)
            summary.update({"format": fmt, "tool_id": tool_id, "bench_id": BENCH_IO})
            write_json(out / "summary.json", summary)
            write_json(out / "per_file.json", {"rows": per_file})


if __name__ == "__main__":
    main()
