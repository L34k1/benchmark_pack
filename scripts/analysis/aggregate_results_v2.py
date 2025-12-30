from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_summary_files(out_root: Path) -> Iterable[Path]:
    for p in out_root.rglob("summary.json"):
        yield p


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        else:
            out[kk] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate benchmark summary.json files into a single CSV/JSONL.")
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--csv", type=Path, default=Path("outputs/_aggregate/summary.csv"))
    p.add_argument("--jsonl", type=Path, default=Path("outputs/_aggregate/summary.jsonl"))
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    for sf in iter_summary_files(args.out_root):
        try:
            obj = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        flat = flatten(obj)
        flat["_path"] = str(sf)
        parts = sf.parts
        try:
            idx = parts.index("outputs")
            flat.setdefault("bench_id", parts[idx + 1])
            flat.setdefault("tool_id", parts[idx + 2])
            flat.setdefault("tag", parts[idx + 3])
        except Exception:
            pass
        rows.append(flat)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with args.csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.jsonl}")


if __name__ == "__main__":
    main()
