#!/usr/bin/env python3
"""scripts/desktop/bench_datoviz.py

Placeholder benchmark entrypoint for Datoviz.
This script currently only records manifests and an error message while the
full interaction benchmarks are implemented.
"""

from __future__ import annotations

import sys

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir, write_json, write_manifest
from benchkit.lexicon import BENCH_A1, BENCH_A2, BENCH_TFFR, FMT_EDF, FMT_NWB, TOOL_DATOVIZ


def main() -> None:
    ap = argparse.ArgumentParser(description="Datoviz benchmark (placeholder).")
    ap.add_argument("--bench-id", choices=[BENCH_TFFR, BENCH_A1, BENCH_A2], required=True)
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    out = out_dir(args.out_root, args.bench_id, TOOL_DATOVIZ, args.tag)
    write_manifest(out, args.bench_id, TOOL_DATOVIZ, args=vars(args), extra={"format": args.format})
    write_json(out / "summary.json", {"bench_id": args.bench_id, "tool_id": TOOL_DATOVIZ, "status": "not_implemented"})


if __name__ == "__main__":
    main()
