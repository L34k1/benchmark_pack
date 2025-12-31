from __future__ import annotations

from typing import Any, Dict, List

import requests

Finding = Dict[str, Any]

SECURITY_HEADERS = {
    "content-security-policy": "Set a Content-Security-Policy header.",
    "x-frame-options": "Set X-Frame-Options to prevent clickjacking.",
    "x-content-type-options": "Set X-Content-Type-Options to nosniff.",
    "referrer-policy": "Define a Referrer-Policy to control referrer leakage.",
    "permissions-policy": "Define a Permissions-Policy header.",
}


def _analyze_cookies(cookies: List[str]) -> List[str]:
    findings = []
    for cookie in cookies:
        lower = cookie.lower()
        missing = []
        if "secure" not in lower:
            missing.append("Secure")
        if "httponly" not in lower:
            missing.append("HttpOnly")
        if "samesite" not in lower:
            missing.append("SameSite")
        if missing:
            findings.append(f"Cookie missing flags: {', '.join(missing)}")
    return findings


def run_headers_probe(run_id: str, target_url: str) -> List[Finding]:
    findings: List[Finding] = []
    try:
        response = requests.get(
            target_url,
            timeout=15,
            allow_redirects=True,
            headers={"User-Agent": "MVP-Web-Security-Scanner"},
        )
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "asset": {"type": "url", "value": target_url},
                "category": "Headers",
                "check": "headers_probe",
                "severity": "Medium",
                "summary": "Failed to fetch headers.",
                "evidence": {
                    "notes": str(exc),
                    "headers": {},
                    "screenshot_path": None,
                    "artifact_path": None,
                },
                "remediation": [
                    "Ensure the target is reachable and supports HTTP/HTTPS.",
                ],
                "source": "probe",
                "run_id": run_id,
            }
        ]
    headers = {k.lower(): v for k, v in response.headers.items()}

    for header, remediation in SECURITY_HEADERS.items():
        if header not in headers:
            findings.append(
                {
                    "asset": {"type": "url", "value": target_url},
                    "category": "Headers",
                    "check": "headers_probe",
                    "severity": "Low",
                    "summary": f"Missing security header: {header}.",
                    "evidence": {
                        "notes": "Header not present in response.",
                        "headers": dict(response.headers),
                        "screenshot_path": None,
                        "artifact_path": None,
                    },
                    "remediation": [remediation],
                    "source": "probe",
                    "run_id": run_id,
                }
            )

    cookies = (
        response.headers.getlist("set-cookie")
        if hasattr(response.headers, "getlist")
        else response.headers.get("set-cookie", "").split("\n")
    )
    cookie_notes = _analyze_cookies([c for c in cookies if c])
    if cookie_notes:
        findings.append(
            {
                "asset": {"type": "url", "value": target_url},
                "category": "Cookies",
                "check": "headers_probe",
                "severity": "Low",
                "summary": "Cookie flags are incomplete.",
                "evidence": {
                    "notes": "; ".join(cookie_notes),
                    "headers": dict(response.headers),
                    "screenshot_path": None,
                    "artifact_path": None,
                },
                "remediation": [
                    "Set Secure, HttpOnly, and SameSite flags on cookies.",
                ],
                "source": "probe",
                "run_id": run_id,
            }
        )

    findings.append(
        {
            "asset": {"type": "url", "value": target_url},
            "category": "Headers",
            "check": "headers_probe",
            "severity": "Low",
            "summary": "Headers probe completed.",
            "evidence": {
                "notes": "Response headers collected.",
                "headers": dict(response.headers),
                "screenshot_path": None,
                "artifact_path": None,
            },
            "remediation": ["Review header configuration and enforce security headers."],
            "source": "probe",
            "run_id": run_id,
        }
    )
    return findings
