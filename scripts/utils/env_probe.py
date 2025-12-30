from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

from benchkit.common import env_info, write_json


def _try(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def main() -> None:
    p = argparse.ArgumentParser(description="Write a richer environment snapshot to JSON.")
    p.add_argument("--out", type=Path, default=Path("outputs/env_probe.json"))
    args = p.parse_args()

    env: Dict[str, Any] = {}
    env.update(env_info())
    env["uname"] = platform.uname()._asdict()
    env["lscpu"] = _try(["bash", "-lc", "lscpu"])
    env["nvidia_smi"] = _try(["bash", "-lc", "nvidia-smi -L"])
    env["glxinfo_renderer"] = _try(["bash", "-lc", "glxinfo -B 2>/dev/null | grep -i 'OpenGL renderer' || true"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, env)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
