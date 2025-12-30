from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(out_root: Path, bench_id: str, tool_id: str, tag: str) -> Path:
    return ensure_dir(out_root / bench_id / tool_id / tag)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def env_info() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_stamp(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }


def write_manifest(out_dir: Path, bench_id: str, tool_id: str, args: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
    manifest = {
        "bench_id": bench_id,
        "tool_id": tool_id,
        "timestamp_utc": utc_stamp(),
        "args": args,
        "extra": extra or {},
        "env": env_info(),
    }
    path = Path(out_dir) / "manifest.json"
    write_json(path, manifest)
    return path
