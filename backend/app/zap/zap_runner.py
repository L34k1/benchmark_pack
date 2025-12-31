from __future__ import annotations

import os
import subprocess
from pathlib import Path


class ZapRunnerError(RuntimeError):
    pass


def run_zap_baseline(
    target_url: str,
    json_path: Path,
    html_path: Path,
    max_duration_sec: int,
) -> None:
    max_minutes = max(1, max_duration_sec // 60)
    docker_cmd = os.environ.get("DOCKER_CMD", "docker")
    container_name = os.environ.get("ZAP_CONTAINER", "zap")
    cmd = [
        docker_cmd,
        "exec",
        container_name,
        "zap-baseline.py",
        "-t",
        target_url,
        "-J",
        str(json_path),
        "-r",
        str(html_path),
        "-m",
        str(max_minutes),
        "-u",
        "MVP-Web-Security-Scanner",
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=max_duration_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise ZapRunnerError("ZAP baseline timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise ZapRunnerError(f"ZAP baseline failed: {exc}") from exc
