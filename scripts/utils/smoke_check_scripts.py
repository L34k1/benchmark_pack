from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def iter_scripts(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path


def run_help(path: Path, timeout_s: float) -> Tuple[Path, int, str]:
    cmd = [sys.executable, str(path), "--help"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        output = (proc.stdout or "") + (proc.stderr or "")
        return path, proc.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return path, 124, f"Timed out after {timeout_s:.1f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check all script entrypoints via --help.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "scripts")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--include", nargs="*", default=None, help="Only run scripts containing any of these tokens.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Skip scripts containing any of these tokens.")
    args = parser.parse_args()

    scripts = list(iter_scripts(args.root))
    if args.include:
        scripts = [p for p in scripts if any(tok in str(p) for tok in args.include)]
    if args.exclude:
        scripts = [p for p in scripts if not any(tok in str(p) for tok in args.exclude)]

    failures: List[Path] = []
    for path in scripts:
        rel = path.relative_to(args.root.parent)
        _, code, output = run_help(path, args.timeout_s)
        if code == 0:
            print(f"[OK] {rel}")
        else:
            print(f"[FAIL] {rel} (exit={code})")
            if output:
                print(output)
            failures.append(path)

    if failures:
        print(f"{len(failures)} script(s) failed.")
        return 1
    print(f"All {len(scripts)} script(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
