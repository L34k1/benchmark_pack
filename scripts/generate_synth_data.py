from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    import pyedflib

    n_ch, n_samples = data.shape
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
                "sample_rate": float(fs_hz),
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
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject
    from hdmf.backends.hdf5.h5_utils import H5DataIO

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


def verify_edf(path: Path, fs_hz: float, min_ch: int, min_samples: int) -> Tuple[int, int]:
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        n_ch = f.signals_in_file
        n_samples = min(f.getNSamples()) if n_ch else 0
    if n_ch < min_ch:
        raise AssertionError(f"EDF has {n_ch} channels (<{min_ch})")
    if n_samples < min_samples:
        raise AssertionError(f"EDF has {n_samples} samples (<{min_samples})")
    return n_ch, n_samples


def verify_nwb(path: Path, fs_hz: float, min_ch: int, min_samples: int) -> Tuple[int, int]:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        series = nwbfile.acquisition.get("eeg")
        if series is None:
            raise AssertionError("NWB missing acquisition 'eeg'")
        shape = series.data.shape
        n_samples, n_ch = int(shape[0]), int(shape[1])
    if n_ch < min_ch:
        raise AssertionError(f"NWB has {n_ch} channels (<{min_ch})")
    if n_samples < min_samples:
        raise AssertionError(f"NWB has {n_samples} samples (<{min_samples})")
    return n_ch, n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EDF + NWB benchmark datasets.")
    parser.add_argument("--n-ch", type=int, default=64)
    parser.add_argument("--duration-s", type=int, default=1800)
    parser.add_argument("--fs-hz", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    n_ch = int(args.n_ch)
    duration_s = int(args.duration_s)
    fs_hz = float(args.fs_hz)
    n_samples = int(duration_s * fs_hz)

    data = generate_signals(n_ch, n_samples, fs_hz, args.seed)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    edf_path = data_dir / f"synth_{n_ch}ch_{duration_s}s_{int(fs_hz)}hz.edf"
    nwb_path = data_dir / f"synth_{n_ch}ch_{duration_s}s_{int(fs_hz)}hz.nwb"

    write_edf(edf_path, data, fs_hz)
    write_nwb(nwb_path, data, fs_hz)

    min_samples = int(fs_hz * duration_s)
    edf_ch, edf_samples = verify_edf(edf_path, fs_hz, n_ch, min_samples)
    nwb_ch, nwb_samples = verify_nwb(nwb_path, fs_hz, n_ch, min_samples)

    edf_size = edf_path.stat().st_size / (1024 * 1024)
    nwb_size = nwb_path.stat().st_size / (1024 * 1024)

    print(
        "EDF: channels=%d samples=%d fs=%.1fHz size=%.1fMB path=%s"
        % (edf_ch, edf_samples, fs_hz, edf_size, edf_path)
    )
    print(
        "NWB: channels=%d samples=%d fs=%.1fHz size=%.1fMB path=%s"
        % (nwb_ch, nwb_samples, fs_hz, nwb_size, nwb_path)
    )


if __name__ == "__main__":
    main()
