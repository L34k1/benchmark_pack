from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from benchkit.common import ensure_dir, utc_stamp, write_json
from benchkit.stats import summarize_latency_ms


def write_manifest_contract(
    out_dir: Path,
    *,
    bench_id: str,
    tool_id: str,
    fmt: str,
    file_path: Path,
    window_s: float,
    n_channels: int,
    sequence: Optional[str],
    overlay: Optional[str],
    run_id: int,
    steps_target: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest = {
        "tool_id": tool_id,
        "bench_id": bench_id,
        "format": fmt,
        "file": str(file_path),
        "window_s": float(window_s),
        "n_channels": int(n_channels),
        "sequence": sequence,
        "overlay": overlay,
        "run_id": int(run_id),
        "steps_target": int(steps_target),
        "timestamp_utc": utc_stamp(),
        "extra": extra or {},
    }
    path = Path(out_dir) / "manifest.json"
    write_json(path, manifest)
    return path


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_steps_csv(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    fieldnames: Optional[List[str]] = None,
) -> Path:
    if fieldnames is None:
        fieldnames = ["step_id", "latency_ms", "noop", "status"]
        extras = [k for k in rows[0].keys() if k not in fieldnames] if rows else []
        fieldnames = fieldnames + extras
    path = Path(out_dir) / "steps.csv"
    _write_csv(path, fieldnames, rows)
    return path


def summarize_steps(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    lat_ms: List[float] = []
    noop = 0
    failed = 0
    for row in rows_list:
        status = str(row.get("status", "OK"))
        if status.upper() in {"FAIL", "ERROR"}:
            failed += 1
        if bool(row.get("noop")):
            noop += 1
        lat = float(row.get("latency_ms", 0.0))
        if status.upper() not in {"FAIL", "ERROR"}:
            lat_ms.append(lat)
    summary = summarize_latency_ms(lat_ms)
    summary.update(
        {
            "noop_steps": int(noop),
            "failed_steps": int(failed),
            "steps_recorded": int(len(rows_list)),
        }
    )
    return summary


def write_steps_summary(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    summary = summarize_steps(rows)
    summary.update(extra or {})
    path = Path(out_dir) / "summary.json"
    write_json(path, summary)
    return path


def write_tffr_csv(out_dir: Path, *, run_id: int, tffr_ms: float) -> Path:
    path = Path(out_dir) / "tffr.csv"
    _write_csv(path, ["run_id", "tffr_ms"], [{"run_id": int(run_id), "tffr_ms": float(tffr_ms)}])
    return path


def write_tffr_summary(
    out_dir: Path,
    *,
    bench_id: str,
    tool_id: str,
    fmt: str,
    window_s: float,
    n_channels: int,
    tffr_ms: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    summary = {
        "bench_id": bench_id,
        "tool_id": tool_id,
        "format": fmt,
        "window_s": float(window_s),
        "n_ch": int(n_channels),
        "tffr_ms": float(tffr_ms),
    }
    if extra:
        summary.update(extra)
    path = Path(out_dir) / "summary.json"
    write_json(path, summary)
    return path


def steps_from_latencies(
    latencies_ms: Iterable[float],
    *,
    steps_target: int,
    status_ok: str = "OK",
    status_fail: str = "FAIL",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lat_list = list(latencies_ms)
    for idx in range(steps_target):
        if idx < len(lat_list):
            latency = float(lat_list[idx])
            rows.append(
                {
                    "step_id": idx,
                    "latency_ms": max(0.0, latency),
                    "noop": False,
                    "status": status_ok,
                }
            )
        else:
            rows.append(
                {
                    "step_id": idx,
                    "latency_ms": 0.0,
                    "noop": False,
                    "status": status_fail,
                }
            )
    return rows
