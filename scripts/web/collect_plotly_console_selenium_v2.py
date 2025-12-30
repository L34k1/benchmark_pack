from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from benchkit.common import ensure_dir, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Collect BENCH_JSON from Plotly benchmark HTML via Selenium.")
    p.add_argument("--html", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout-ms", type=int, default=20000)
    p.add_argument("--browser", choices=["chrome", "firefox"], default="chrome")
    args = p.parse_args()

    ensure_dir(args.out.parent)
    bench_json: Optional[dict] = None

    if args.browser == "chrome":
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        drv = webdriver.Chrome(options=opts)
    else:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        opts = Options()
        opts.add_argument("--headless")
        drv = webdriver.Firefox(options=opts)

    try:
        drv.get(args.html.resolve().as_uri())

        t0 = time.time()
        while bench_json is None and (time.time() - t0) * 1000 < args.timeout_ms:
            for entry in drv.get_log("browser"):
                msg = entry.get("message", "")
                if "BENCH_JSON:" in msg:
                    payload = msg.split("BENCH_JSON:", 1)[1]
                    j0 = payload.find("{")
                    if j0 >= 0:
                        payload = payload[j0:]
                    try:
                        bench_json = json.loads(payload)
                        break
                    except Exception:
                        pass
            time.sleep(0.05)

    finally:
        drv.quit()

    if bench_json is None:
        write_json(args.out, {"error": "BENCH_JSON not captured", "html": str(args.html)})
        raise SystemExit(
            "BENCH_JSON not captured. Selenium console logs may be restricted. "
            "Use Playwright collector or copy from DevTools."
        )

    write_json(args.out, bench_json)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
