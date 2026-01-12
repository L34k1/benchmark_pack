from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _synth_signal(rng: np.random.Generator, t: np.ndarray, ch_idx: int) -> np.ndarray:
    base_freqs = np.array([0.8, 3.0, 7.5, 12.0, 22.0], dtype=np.float32)
    freq_jitter = rng.uniform(-0.2, 0.2, size=base_freqs.shape).astype(np.float32)
    freqs = base_freqs + freq_jitter
    phases = rng.uniform(0, 2 * math.pi, size=base_freqs.shape).astype(np.float32)
    amps = rng.uniform(0.3, 1.2, size=base_freqs.shape).astype(np.float32)

    signal = np.zeros_like(t, dtype=np.float32)
    for amp, freq, phase in zip(amps, freqs, phases):
        signal += amp * np.sin(2 * math.pi * freq * t + phase).astype(np.float32)

    white = rng.normal(0.0, 0.3, size=t.shape).astype(np.float32)
    drift = rng.normal(0.0, 0.02, size=t.shape).astype(np.float32).cumsum()
    drift = drift - drift.mean()

    gain = rng.uniform(0.8, 1.3)
    offset = rng.uniform(-0.2, 0.2) + ch_idx * 0.005
    return (signal + white + drift) * gain + offset


def generate_signals(n_ch: int, n_samples: int, fs_hz: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = (np.arange(n_samples, dtype=np.float32) / float(fs_hz)).astype(np.float32)
    data = np.zeros((n_ch, n_samples), dtype=np.float32)
    for ch in range(n_ch):
        data[ch] = _synth_signal(rng, t, ch)
    return data


def write_edf(path: Path, data: np.ndarray, fs_hz: float) -> None:
    try:
        import pyedflib
    except ImportError:
        print("SKIP_UNSUPPORTED_FORMAT")
        raise SystemExit(2)

    n_ch, _ = data.shape
    labels = [f"ch{idx+1:02d}" for idx in range(n_ch)]
    signal_headers = []
    for idx, label in enumerate(labels):
        ch_data = data[idx]
        physical_min = float(np.min(ch_data))
        physical_max = float(np.max(ch_data))
        if physical_max - physical_min < 1e-6:
            physical_max = physical_min + 1.0
        signal_headers.append(
            {
                "label": label,
                "dimension": "uV",
                "sample_frequency": float(fs_hz),
                "physical_min": physical_min,
                "physical_max": physical_max,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "synth",
                "prefilter": "none",
            }
        )

    with pyedflib.EdfWriter(str(path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        writer.setSignalHeaders(signal_headers)
        writer.writeSamples([data[ch] for ch in range(n_ch)])


def write_nwb(path: Path, data: np.ndarray, fs_hz: float) -> None:
    try:
        from pynwb import NWBFile, NWBHDF5IO
        from pynwb.ecephys import ElectricalSeries
        from pynwb.file import Subject
        from hdmf.backends.hdf5.h5_utils import H5DataIO
    except ImportError:
        print("SKIP_UNSUPPORTED_FORMAT")
        raise SystemExit(2)

    n_ch, n_samples = data.shape
    nwbfile = NWBFile(
        session_description="synthetic eeg benchmark",
        identifier=path.stem,
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.subject = Subject(subject_id="synth")
    device = nwbfile.create_device(name="synth_device")
    group = nwbfile.create_electrode_group(
        name="synth_group",
        description="synthetic",
        location="synthetic",
        device=device,
    )

    for idx in range(n_ch):
        nwbfile.add_electrode(
            id=idx,
            x=float(idx),
            y=0.0,
            z=0.0,
            imp=float("nan"),
            location="synthetic",
            filtering="none",
            group=group,
        )

    region = nwbfile.create_electrode_table_region(list(range(n_ch)), "all electrodes")
    chunk_time = min(4096, n_samples)
    data_io = H5DataIO(
        data.T,
        compression="gzip",
        chunks=(chunk_time, n_ch),
    )
    series = ElectricalSeries(
        name="eeg",
        data=data_io,
        rate=float(fs_hz),
        electrodes=region,
    )
    nwbfile.add_acquisition(series)

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single synthetic EDF/NWB file.")
    parser.add_argument("--format", choices=["EDF", "NWB"], required=True)
    parser.add_argument("--n-ch", type=int, default=8)
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--fs-hz", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "_synth")
    parser.add_argument("--no-clobber", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "edf" if args.format == "EDF" else "nwb"
    out_path = out_dir / f"synth_{args.n_ch}ch_{args.duration_s}s_{int(args.fs_hz)}hz.{suffix}"

    if out_path.exists() and args.no_clobber:
        print(f"Exists (no-clobber): {out_path}")
        return

    n_samples = int(args.duration_s * args.fs_hz)
    data = generate_signals(int(args.n_ch), n_samples, float(args.fs_hz), int(args.seed))

    if args.format == "EDF":
        write_edf(out_path, data, float(args.fs_hz))
    elif args.format == "NWB":
        write_nwb(out_path, data, float(args.fs_hz))
    else:
        print("SKIP_UNSUPPORTED_FORMAT")
        raise SystemExit(2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
