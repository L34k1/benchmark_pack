from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from collections import deque

from benchkit.capabilities import check_format_dependency, check_tool, playwright_available
from benchkit.common import ensure_dir
from benchkit.bench_defaults import (
    DEFAULT_LOAD_DURATION_MULTIPLIER,
    DEFAULT_TARGET_INTERVAL_MS,
    DEFAULT_WINDOW_S,
    default_load_duration_s,
)
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    BENCH_IO,
    BENCH_TFFR,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_DATOVIZ,
    TOOL_D3,
    TOOL_IO_MNE,
    TOOL_IO_NEO,
    TOOL_IO_PYEDFLIB,
    TOOL_PG,
    TOOL_PLOTLY,
    TOOL_VISPY,
)

REPO_ROOT = Path(__file__).resolve().parent

MODES = {
    "IO": BENCH_IO,
    "TFFR": BENCH_TFFR,
    "A1": BENCH_A1,
    "A2": BENCH_A2,
}


@dataclass(frozen=True)
class ToolConfig:
    tool_id: str
    benches: Tuple[str, ...]
    formats: Tuple[str, ...]
    scripts: Dict[str, Path]
    kind: str


@dataclass
class Job:
    bench_id: str
    tool_id: str
    fmt: str
    file_path: Path
    window_s: float
    load_duration_s: float
    n_channels: int
    sequence: str
    overlay: str
    run_id: int
    cadence_ms: Optional[float]
    tag: str
    out_dir: Path
    cmd: List[str]
    timeout_s: float
    heartbeat_timeout_s: float
    job_id: str
    skip_reason: Optional[str] = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("-")
    out = "".join(safe).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "na"


def job_hash(payload: Dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def build_tag(job: Dict[str, object]) -> str:
    base = slugify(str(job["file"]).replace(os.sep, "_"))
    parts = [
        base,
        f"fmt{job['format']}",
        f"win{job['window_s']}s",
        f"dur{job['load_duration_s']}s",
        f"ch{job['n_channels']}",
        f"seq{job['sequence']}",
        f"ovl{job['overlay']}",
        f"run{job['run_id']}",
    ]
    cadence = job.get("cadence_ms")
    if cadence is not None:
        parts.append(f"cad{cadence}ms")
    return slugify("_".join(str(p) for p in parts))


def timings_header() -> List[str]:
    return [
        "timestamp_start",
        "timestamp_end",
        "duration_s",
        "status",
        "bench_id",
        "tool_id",
        "format",
        "file",
        "window_s",
        "load_duration_s",
        "n_channels",
        "sequence",
        "overlay",
        "run_id",
        "cmdline",
        "out_dir",
        "error_type",
        "error_message",
    ]


def is_done(bench_id: str, out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    manifest_ok = (out_dir / "manifest.json").exists()
    summary_ok = (out_dir / "summary.json").exists() or (out_dir / "summary.csv").exists()
    if not summary_ok:
        summary_files = list(out_dir.glob("*summary*.csv"))
        summary_ok = bool(summary_files)
    if bench_id in (BENCH_A1, BENCH_A2):
        steps_ok = (out_dir / "steps.csv").exists()
        return manifest_ok and summary_ok and steps_ok
    if bench_id == BENCH_TFFR:
        tffr_csv = list(out_dir.glob("*tffr*.csv"))
        summary_ok = summary_ok or bool(tffr_csv)
    return manifest_ok and summary_ok


def write_steps_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def derive_summary(job: Job, bench_json: Dict[str, object]) -> Dict[str, object]:
    if job.tool_id in {TOOL_PLOTLY, TOOL_D3}:
        if job.bench_id == BENCH_TFFR:
            tffr_ms = bench_json.get("first_render_ms")
            if tffr_ms is None:
                tffr_ms = bench_json.get("tffr_ms")
            return {
                "bench_id": job.bench_id,
                "tool_id": job.tool_id,
                "format": job.fmt,
                "overlay": job.overlay,
                "window_s": job.window_s,
                "n_ch": job.n_channels,
                "tffr_ms": tffr_ms,
                "meta": bench_json.get("meta", {}),
            }
        if job.tool_id == TOOL_PLOTLY:
            return {
                "bench_id": job.bench_id,
                "tool_id": job.tool_id,
                "format": job.fmt,
                "overlay": job.overlay,
                "window_s": job.window_s,
                "sequence": job.sequence,
                "summary": bench_json.get("summary", []),
                "meta": bench_json.get("meta", {}),
            }
        return bench_json
    return bench_json


def collect_bench_json(html_path: Path, out_path: Path, log_path: Path, timeout_ms: int) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "web" / "collect_console_playwright.py"),
           "--html", str(html_path), "--out", str(out_path), "--timeout-ms", str(timeout_ms)]
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n[collector] " + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=log_file, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Collector failed with exit code {proc.returncode}")


def run_job(job: Job, log_path: Path, logger: logging.Logger, force: bool) -> Tuple[str, Optional[str], Optional[str], float]:
    if job.skip_reason:
        logger.info("SKIP %s (%s)", job.job_id, job.skip_reason)
        return "SKIP", "skip", job.skip_reason, 0.0

    if not force and is_done(job.bench_id, job.out_dir):
        logger.info("SKIP %s (already completed)", job.job_id)
        return "SKIP", "already_done", "Output exists", 0.0

    ensure_dir(job.out_dir)
    ensure_dir(log_path.parent)

    logger.info(
        "RUN %s %s %s fmt=%s file=%s win=%ss ch=%s seq=%s ovl=%s run=%s elapsed=0.0s last_activity=0.0s -> %s",
        job.job_id,
        job.tool_id,
        job.bench_id,
        job.fmt,
        job.file_path.name,
        job.window_s,
        job.n_channels,
        job.sequence,
        job.overlay,
        job.run_id,
        job.out_dir,
    )

    start = time.time()
    last_activity = {"ts": start}
    log_lock = threading.Lock()
    tail_lines: Deque[str] = deque(maxlen=50)

    def stream_reader(stream, label: str) -> None:
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in iter(stream.readline, ""):
                with log_lock:
                    log_file.write(line)
                    log_file.flush()
                    tail_lines.append(f"[{label}] {line.rstrip()}")
                last_activity["ts"] = time.time()
        stream.close()

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("[command] " + " ".join(job.cmd) + "\n")
        log_file.flush()

    proc = subprocess.Popen(job.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    threads = [
        threading.Thread(target=stream_reader, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=stream_reader, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for t in threads:
        t.start()

    warn_every_s = 60.0
    next_warn = start + max(10.0, job.timeout_s * 0.6)

    def terminate_proc(reason: str) -> None:
        logger.error("Terminating job %s: %s", job.job_id, reason)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def tail_snippet() -> str:
        if not tail_lines:
            return ""
        return "\n".join(tail_lines)

    while True:
        if proc.poll() is not None:
            break
        now = time.time()
        elapsed = now - start
        if elapsed > job.timeout_s:
            terminate_proc(f"timeout exceeded {job.timeout_s}s")
            for t in threads:
                t.join(timeout=1)
            snippet = tail_snippet()
            msg = f"Exceeded timeout {job.timeout_s}s"
            if snippet:
                msg += f"\nLast output:\n{snippet}"
            return "FAIL", "timeout", msg, elapsed
        idle_s = now - last_activity["ts"]
        if idle_s > job.heartbeat_timeout_s:
            terminate_proc(f"idle for {idle_s:.1f}s (heartbeat timeout {job.heartbeat_timeout_s}s)")
            for t in threads:
                t.join(timeout=1)
            snippet = tail_snippet()
            msg = f"No heartbeat for {idle_s:.1f}s (limit {job.heartbeat_timeout_s}s)"
            if snippet:
                msg += f"\nLast output:\n{snippet}"
            return "FAIL", "no_heartbeat", msg, elapsed
        if now >= next_warn:
            logger.warning(
                "Job %s still running (elapsed=%.1fs, last_activity=%.1fs)",
                job.job_id,
                elapsed,
                idle_s,
            )
            next_warn = now + warn_every_s
        time.sleep(0.5)

    for t in threads:
        t.join(timeout=1)

    duration = time.time() - start
    if proc.returncode != 0:
        return "FAIL", "exit_code", f"Exit code {proc.returncode}", duration

    if job.tool_id in {TOOL_PLOTLY, TOOL_D3}:
        try:
            html_name = {
                (TOOL_PLOTLY, BENCH_TFFR): "plotly_tffr.html",
                (TOOL_PLOTLY, BENCH_A1): "plotly_A1_interactions.html",
                (TOOL_PLOTLY, BENCH_A2): "plotly_A2_interactions.html",
                (TOOL_D3, BENCH_TFFR): "d3_bench.html",
                (TOOL_D3, BENCH_A1): "d3_bench.html",
                (TOOL_D3, BENCH_A2): "d3_bench.html",
            }[(job.tool_id, job.bench_id)]
            html_path = job.out_dir / html_name
            bench_json_path = job.out_dir / "bench.json"
            collect_bench_json(html_path, bench_json_path, log_path, timeout_ms=20000)
            bench_json = json.loads(bench_json_path.read_text(encoding="utf-8"))

            if job.bench_id in (BENCH_A1, BENCH_A2):
                rows: List[Dict[str, object]] = []
                if job.tool_id == TOOL_PLOTLY:
                    for result in bench_json.get("results", []):
                        seq = result.get("seq")
                        for row in result.get("rows", []):
                            out_row = dict(row)
                            out_row.setdefault("seq", seq)
                            rows.append(out_row)
                elif job.tool_id == TOOL_D3:
                    lat_ms = bench_json.get("lat_ms") or []
                    late_ms = bench_json.get("lateness_ms") or []
                    for idx, lat in enumerate(lat_ms):
                        row = {"step_id": idx, "latency_ms": lat}
                        if late_ms:
                            row["lateness_ms"] = late_ms[idx]
                        rows.append(row)
                if rows:
                    write_steps_csv(rows, job.out_dir / "steps.csv")

            summary = derive_summary(job, bench_json)
            (job.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - want to keep running
            return "FAIL", "exception", str(exc), duration

    if not is_done(job.bench_id, job.out_dir):
        return "FAIL", "missing_outputs", "Expected outputs not found", duration

    return "OK", None, None, duration


def build_tools() -> Dict[str, ToolConfig]:
    return {
        TOOL_PG: ToolConfig(
            tool_id=TOOL_PG,
            benches=(BENCH_TFFR, BENCH_A1, BENCH_A2),
            formats=(FMT_EDF, FMT_NWB),
            scripts={
                BENCH_TFFR: REPO_ROOT / "scripts" / "desktop" / "bench_pyqtgraph_tffr_v2.py",
                BENCH_A1: REPO_ROOT / "scripts" / "desktop" / "bench_pyqtgraph_A1_throughput_v2.py",
                BENCH_A2: REPO_ROOT / "scripts" / "desktop" / "bench_pyqtgraph_A2_cadenced_v2.py",
            },
            kind="desktop",
        ),
        TOOL_VISPY: ToolConfig(
            tool_id=TOOL_VISPY,
            benches=(BENCH_A1, BENCH_A2),
            formats=(FMT_EDF, FMT_NWB),
            scripts={
                BENCH_A1: REPO_ROOT / "scripts" / "desktop" / "bench_vispy_A1_throughput.py",
                BENCH_A2: REPO_ROOT / "scripts" / "desktop" / "bench_vispy_A2_cadenced.py",
            },
            kind="desktop",
        ),
        TOOL_DATOVIZ: ToolConfig(
            tool_id=TOOL_DATOVIZ,
            benches=(BENCH_TFFR, BENCH_A1, BENCH_A2),
            formats=(FMT_EDF, FMT_NWB),
            scripts={
                BENCH_TFFR: REPO_ROOT / "scripts" / "desktop" / "bench_datoviz.py",
                BENCH_A1: REPO_ROOT / "scripts" / "desktop" / "bench_datoviz.py",
                BENCH_A2: REPO_ROOT / "scripts" / "desktop" / "bench_datoviz.py",
            },
            kind="desktop",
        ),
        TOOL_PLOTLY: ToolConfig(
            tool_id=TOOL_PLOTLY,
            benches=(BENCH_TFFR, BENCH_A1, BENCH_A2),
            formats=(FMT_EDF, FMT_NWB),
            scripts={
                BENCH_TFFR: REPO_ROOT / "scripts" / "web" / "bench_plotly_tffr.py",
                BENCH_A1: REPO_ROOT / "scripts" / "web" / "bench_plotly_A1_throughput.py",
                BENCH_A2: REPO_ROOT / "scripts" / "web" / "bench_plotly_A2_cadenced.py",
            },
            kind="web",
        ),
        TOOL_D3: ToolConfig(
            tool_id=TOOL_D3,
            benches=(BENCH_TFFR, BENCH_A1, BENCH_A2),
            formats=(FMT_EDF, FMT_NWB),
            scripts={
                BENCH_TFFR: REPO_ROOT / "scripts" / "web" / "gen_d3_html.py",
                BENCH_A1: REPO_ROOT / "scripts" / "web" / "gen_d3_html.py",
                BENCH_A2: REPO_ROOT / "scripts" / "web" / "gen_d3_html.py",
            },
            kind="web",
        ),
        TOOL_IO_PYEDFLIB: ToolConfig(
            tool_id=TOOL_IO_PYEDFLIB,
            benches=(BENCH_IO,),
            formats=(FMT_EDF,),
            scripts={BENCH_IO: REPO_ROOT / "scripts" / "io" / "bench_io_v2.py"},
            kind="io",
        ),
        TOOL_IO_MNE: ToolConfig(
            tool_id=TOOL_IO_MNE,
            benches=(BENCH_IO,),
            formats=(FMT_EDF,),
            scripts={BENCH_IO: REPO_ROOT / "scripts" / "io" / "bench_io_v2.py"},
            kind="io",
        ),
        TOOL_IO_NEO: ToolConfig(
            tool_id=TOOL_IO_NEO,
            benches=(BENCH_IO,),
            formats=(FMT_EDF,),
            scripts={BENCH_IO: REPO_ROOT / "scripts" / "io" / "bench_io_v2.py"},
            kind="io",
        ),
    }


def build_command(job: Job, tool_cfg: ToolConfig) -> List[str]:
    script = tool_cfg.scripts[job.bench_id]
    base = [sys.executable, str(script)]

    if tool_cfg.tool_id in {TOOL_IO_PYEDFLIB, TOOL_IO_MNE, TOOL_IO_NEO}:
        return base + [
            "--format",
            job.fmt,
            "--file",
            str(job.file_path),
            "--tool",
            job.tool_id,
            "--out-root",
            str(job.out_dir.parents[2]),
            "--tag",
            job.tag,
            "--window-s",
            str(job.window_s),
            "--n-ch",
            str(job.n_channels),
            "--runs",
            "1",
            "--n-files",
            "1",
            "--no-append-format-tag",
        ]

    if tool_cfg.tool_id == TOOL_DATOVIZ:
        cmd = base + [
            "--bench-id",
            job.bench_id,
            "--format",
            job.fmt,
            "--file",
            str(job.file_path),
            "--out-root",
            str(job.out_dir.parents[2]),
            "--tag",
            job.tag,
            "--n-ch",
            str(job.n_channels),
            "--window-s",
            str(job.window_s),
            "--load-duration-s",
            str(job.load_duration_s),
            "--sequence",
            job.sequence,
            "--runs",
            "1",
            "--overlay",
            job.overlay,
        ]
        if job.cadence_ms is not None:
            cmd += ["--target-interval-ms", str(job.cadence_ms)]
        return cmd

    if tool_cfg.tool_id in {TOOL_PLOTLY, TOOL_D3}:
        cmd = base + [
            "--format",
            job.fmt,
            "--file",
            str(job.file_path),
            "--out-root",
            str(job.out_dir.parents[2]),
            "--tag",
            job.tag,
            "--window-s",
            str(job.window_s),
        ]
        if tool_cfg.tool_id == TOOL_D3 or job.bench_id != BENCH_TFFR:
            if tool_cfg.tool_id == TOOL_PLOTLY:
                cmd += ["--load-duration", str(job.load_duration_s)]
            else:
                cmd += ["--load-duration-s", str(job.load_duration_s)]
        cmd += [
            "--n-ch",
            str(job.n_channels),
            "--overlay",
            job.overlay,
        ]
        if tool_cfg.tool_id == TOOL_D3:
            cmd += ["--bench-id", job.bench_id]
        if job.bench_id != BENCH_TFFR:
            cmd += ["--sequence", job.sequence]
        if job.bench_id == BENCH_A2 and job.cadence_ms is not None:
            cmd += ["--target-interval-ms", str(job.cadence_ms)]
        return cmd

    cmd = base + [
        "--format",
        job.fmt,
        "--file",
        str(job.file_path),
        "--out-root",
        str(job.out_dir.parents[2]),
        "--tag",
        job.tag,
        "--window-s",
        str(job.window_s),
        "--load-duration-s",
        str(job.load_duration_s),
        "--n-ch",
        str(job.n_channels),
        "--overlay",
        job.overlay,
    ]
    if job.bench_id != BENCH_TFFR:
        cmd += ["--sequence", job.sequence]
    if job.bench_id == BENCH_A2 and job.cadence_ms is not None:
        cmd += ["--target-interval-ms", str(job.cadence_ms)]
    if job.tool_id == TOOL_PG and job.bench_id in (BENCH_A1, BENCH_A2):
        cmd += ["--run-timeout-s", str(job.timeout_s)]
    return cmd


def list_data_files(data_dir: Path) -> Dict[str, List[Path]]:
    edf_files = sorted([p for p in data_dir.rglob("*.edf") if p.is_file()])
    nwb_files = sorted([p for p in data_dir.rglob("*.nwb") if p.is_file()])
    return {FMT_EDF: edf_files, FMT_NWB: nwb_files}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all benchmarks end-to-end.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--runs", type=int, default=3)
    default_windows = [
        DEFAULT_WINDOW_S,
        DEFAULT_WINDOW_S * 10,
        DEFAULT_WINDOW_S * 30,
    ]
    p.add_argument("--windows", nargs="+", type=float, default=default_windows)
    p.add_argument("--channels", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument(
        "--sequences",
        nargs="+",
        default=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM],
    )
    p.add_argument("--overlays", nargs="+", default=[OVL_OFF, OVL_ON])
    p.add_argument("--modes", nargs="+", default=["IO", "TFFR", "A1", "A2"])
    p.add_argument("--tools", nargs="+", default=["all"])
    p.add_argument("--cadence-ms", nargs="+", type=float, default=[DEFAULT_TARGET_INTERVAL_MS])
    p.add_argument("--load-duration-multiplier", type=float, default=DEFAULT_LOAD_DURATION_MULTIPLIER)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("--heartbeat-timeout-s", type=float, default=15.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_root = args.out_root
    orch_root = ensure_dir(out_root / "_orchestration")
    jobs_root = ensure_dir(orch_root / "jobs")

    log_path = orch_root / "orchestrator.log"
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    tools = build_tools()
    selected_tools = list(tools.keys()) if args.tools == ["all"] else args.tools

    if args.max_parallel > 1:
        logger.warning("max-parallel=%s requested; running sequentially for safety.", args.max_parallel)

    for tool_id in selected_tools:
        if tool_id not in tools:
            raise SystemExit(f"Unknown tool: {tool_id}")

    modes = [MODES[m] for m in args.modes]

    data_files = list_data_files(args.data_dir)
    for fmt, files in data_files.items():
        if not files:
            logger.warning("No %s files found under %s", fmt, args.data_dir)

    tool_availability: Dict[str, Tuple[bool, str]] = {
        tool_id: check_tool(tool_id) for tool_id in selected_tools
    }

    if not playwright_available():
        for tool_id in selected_tools:
            if tools[tool_id].kind == "web":
                tool_availability[tool_id] = (False, "Missing dependency: playwright")

    default_timeouts = {
        BENCH_IO: 120.0,
        BENCH_TFFR: 60.0,
        BENCH_A1: 180.0,
        BENCH_A2: 180.0,
    }

    jobs: List[Job] = []

    for tool_id in selected_tools:
        tool_cfg = tools[tool_id]
        tool_ok, tool_reason = tool_availability[tool_id]

        for bench_id in tool_cfg.benches:
            if bench_id not in modes:
                continue
            for fmt in tool_cfg.formats:
                fmt_ok, fmt_reason = check_format_dependency(fmt)
                files = data_files.get(fmt, [])
                if not files:
                    continue
                for file_path in files:
                    for window_s in args.windows:
                        for n_channels in args.channels:
                            sequences = [SEQ_PAN_ZOOM]
                            overlays = [OVL_OFF]
                            if bench_id in (BENCH_A1, BENCH_A2):
                                sequences = args.sequences
                                overlays = args.overlays
                            elif bench_id == BENCH_TFFR:
                                overlays = args.overlays
                            for sequence in sequences:
                                for overlay in overlays:
                                    cadence_list = [None]
                                    if bench_id == BENCH_A2:
                                        cadence_list = args.cadence_ms
                                    for cadence_ms in cadence_list:
                                        for run_id in range(int(args.runs)):
                                            load_duration_s = default_load_duration_s(
                                                window_s, args.load_duration_multiplier
                                            )
                                            job_payload = {
                                                "bench_id": bench_id,
                                                "tool_id": tool_id,
                                                "format": fmt,
                                                "file": file_path.name,
                                                "window_s": window_s,
                                                "load_duration_s": load_duration_s,
                                                "n_channels": n_channels,
                                                "sequence": sequence,
                                                "overlay": overlay,
                                                "run_id": run_id,
                                                "cadence_ms": cadence_ms,
                                            }
                                            tag = build_tag(job_payload)
                                            out_dir = out_root / bench_id / tool_id / tag
                                            timeout_s = args.timeout_s or default_timeouts[bench_id]
                                            cmd = build_command(
                                                Job(
                                                    bench_id=bench_id,
                                                    tool_id=tool_id,
                                                    fmt=fmt,
                                                    file_path=file_path,
                                                    window_s=window_s,
                                                    load_duration_s=load_duration_s,
                                                    n_channels=n_channels,
                                                    sequence=sequence,
                                                    overlay=overlay,
                                                    run_id=run_id,
                                                    cadence_ms=cadence_ms,
                                                    tag=tag,
                                                    out_dir=out_dir,
                                                    cmd=[],
                                                    timeout_s=timeout_s,
                                                    heartbeat_timeout_s=float(args.heartbeat_timeout_s),
                                                    job_id="",
                                                ),
                                                tool_cfg,
                                            )
                                            job_id = job_hash({**job_payload, "cmd": cmd})
                                            job = Job(
                                                bench_id=bench_id,
                                                tool_id=tool_id,
                                                fmt=fmt,
                                                file_path=file_path,
                                                window_s=window_s,
                                                load_duration_s=load_duration_s,
                                                n_channels=n_channels,
                                                sequence=sequence,
                                                overlay=overlay,
                                                run_id=run_id,
                                                cadence_ms=cadence_ms,
                                                tag=tag,
                                                out_dir=out_dir,
                                                cmd=cmd,
                                                timeout_s=timeout_s,
                                                heartbeat_timeout_s=float(args.heartbeat_timeout_s),
                                                job_id=job_id,
                                            )
                                            if not tool_ok:
                                                job.skip_reason = tool_reason
                                            elif not fmt_ok:
                                                job.skip_reason = fmt_reason
                                            jobs.append(job)

    timings_path = orch_root / "timings.csv"
    if not timings_path.exists():
        with timings_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(timings_header())

    total = len(jobs)
    logger.info("Starting %d jobs", total)

    results: List[Tuple[Job, str, Optional[str], Optional[str], float, Path]] = []
    failures: List[Tuple[Job, Path]] = []

    for idx, job in enumerate(jobs, start=1):
        job_log = jobs_root / f"{job.job_id}.log"
        start_ts = utc_stamp()
        status, error_type, error_message, duration = run_job(job, job_log, logger, args.force)
        end_ts = utc_stamp()

        if status == "FAIL":
            failures.append((job, job_log))
        results.append((job, status, error_type, error_message, duration, job_log))

        with timings_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    start_ts,
                    end_ts,
                    f"{duration:.3f}",
                    status,
                    job.bench_id,
                    job.tool_id,
                    job.fmt,
                    str(job.file_path),
                    f"{job.window_s}",
                    f"{job.load_duration_s}",
                    str(job.n_channels),
                    job.sequence,
                    job.overlay,
                    str(job.run_id),
                    " ".join(job.cmd),
                    str(job.out_dir),
                    error_type or "",
                    error_message or "",
                ]
            )

        logger.info(
            "[%d/%d] %s %s %s %s %s %s run=%s elapsed=%.1fs",
            idx,
            total,
            status,
            job.tool_id,
            job.bench_id,
            job.fmt,
            job.file_path.name,
            job.sequence,
            job.run_id,
            duration,
        )

    ok_count = sum(1 for _, status, *_ in results if status == "OK")
    skip_count = sum(1 for _, status, *_ in results if status == "SKIP")
    fail_count = sum(1 for _, status, *_ in results if status == "FAIL")

    logger.info("Completed: OK=%d FAIL=%d SKIP=%d", ok_count, fail_count, skip_count)

    slowest = sorted(
        [r for r in results if r[1] == "OK"], key=lambda r: r[4], reverse=True
    )[:20]
    if slowest:
        logger.info("Top-20 slowest jobs:")
        for job, _, _, _, dur, log_path in slowest:
            logger.info("  %.1fs %s %s (%s)", dur, job.tool_id, job.tag, log_path)

    if failures:
        logger.error("Failing jobs:")
        for job, log_path in failures:
            logger.error("  %s %s %s -> %s", job.tool_id, job.bench_id, job.tag, log_path)

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
