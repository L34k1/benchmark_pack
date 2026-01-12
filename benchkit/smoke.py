from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def validate_steps_csv(path: Path, expected_steps: int) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"Missing steps.csv: {path}"
    rows = _read_csv_rows(path)
    if len(rows) != expected_steps:
        return False, f"steps.csv rows={len(rows)} expected={expected_steps}"
    for idx, row in enumerate(rows):
        try:
            latency = float(row.get("latency_ms", "nan"))
        except ValueError:
            return False, f"steps.csv row {idx} latency_ms not numeric"
        if latency < 0:
            return False, f"steps.csv row {idx} latency_ms negative"
    return True, "OK"


def validate_tffr_csv(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"Missing tffr.csv: {path}"
    rows = _read_csv_rows(path)
    if len(rows) != 1:
        return False, f"tffr.csv rows={len(rows)} expected=1"
    try:
        float(rows[0].get("tffr_ms", "nan"))
    except ValueError:
        return False, "tffr.csv tffr_ms not numeric"
    return True, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark output contract artifacts.")
    parser.add_argument("--steps-csv", type=Path, default=None)
    parser.add_argument("--tffr-csv", type=Path, default=None)
    parser.add_argument("--expected-steps", type=int, default=200)
    args = parser.parse_args()

    if not args.steps_csv and not args.tffr_csv:
        raise SystemExit("Provide --steps-csv or --tffr-csv")

    if args.steps_csv:
        ok, msg = validate_steps_csv(args.steps_csv, args.expected_steps)
        if not ok:
            print(f"[FAIL] {msg}")
            return 1
        print(f"[OK] {args.steps_csv}")

    if args.tffr_csv:
        ok, msg = validate_tffr_csv(args.tffr_csv)
        if not ok:
            print(f"[FAIL] {msg}")
            return 1
        print(f"[OK] {args.tffr_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
