from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from benchkit.common import ensure_dir, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Collect BENCH_JSON from Plotly benchmark HTML via Playwright.")
    p.add_argument("--html", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout-ms", type=int, default=20000)
    args = p.parse_args()

    ensure_dir(args.out.parent)

    from playwright.sync_api import sync_playwright

    bench_json: Optional[dict] = None
    errors = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        def on_console(msg):
            nonlocal bench_json
            txt = msg.text()
            if "BENCH_JSON:" in txt:
                try:
                    payload = txt.split("BENCH_JSON:", 1)[1]
                    bench_json = json.loads(payload)
                except Exception as e:
                    errors.append(f"Failed to parse BENCH_JSON: {e}")

        page.on("console", on_console)

        page.goto(args.html.resolve().as_uri())

        t0 = time.time()
        while bench_json is None and (time.time() - t0) * 1000 < args.timeout_ms:
            page.wait_for_timeout(50)

        browser.close()

    if bench_json is None:
        write_json(args.out, {"error": "BENCH_JSON not captured", "errors": errors, "html": str(args.html)})
        raise SystemExit("BENCH_JSON not captured. Open the HTML and copy from DevTools console.")

    write_json(args.out, bench_json)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
