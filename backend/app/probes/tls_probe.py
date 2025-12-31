from __future__ import annotations

import datetime
import socket
import ssl
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

Finding = Dict[str, Any]


def _extract_certificate(hostname: str, port: int = 443) -> dict:
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port), timeout=10) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert = ssock.getpeercert()
    return cert


def _parse_sans(cert: dict) -> List[str]:
    sans = []
    for entry in cert.get("subjectAltName", []):
        if len(entry) == 2:
            sans.append(entry[1])
    return sans


def _cert_expiry_days(cert: dict) -> int:
    expires = cert.get("notAfter")
    if not expires:
        return -1
    dt = datetime.datetime.strptime(expires, "%b %d %H:%M:%S %Y %Z")
    return (dt - datetime.datetime.utcnow()).days


def run_tls_probe(run_id: str, target_url: str) -> List[Finding]:
    parsed = urlparse(target_url)
    hostname = parsed.hostname
    if not hostname:
        return []
    findings: List[Finding] = []
    try:
        cert = _extract_certificate(hostname)
        sans = _parse_sans(cert)
        days_left = _cert_expiry_days(cert)
        findings.append(
            {
                "asset": {"type": "url", "value": target_url},
                "category": "TLS",
                "check": "tls_probe",
                "severity": "Low" if days_left > 30 else "Medium",
                "summary": f"TLS certificate expires in {days_left} days.",
                "evidence": {
                    "notes": "Certificate metadata collected.",
                    "headers": {},
                    "screenshot_path": None,
                    "artifact_path": None,
                    "certificate": cert,
                    "sans": sans,
                },
                "remediation": [
                    "Monitor certificate expiration and renew before expiry.",
                    "Ensure SAN entries match expected domains.",
                ],
                "source": "probe",
                "run_id": run_id,
            }
        )
    except Exception as exc:  # noqa: BLE001
        findings.append(
            {
                "asset": {"type": "url", "value": target_url},
                "category": "TLS",
                "check": "tls_probe",
                "severity": "Medium",
                "summary": "Failed to retrieve TLS certificate.",
                "evidence": {
                    "notes": str(exc),
                    "headers": {},
                    "screenshot_path": None,
                    "artifact_path": None,
                },
                "remediation": [
                    "Ensure the target supports TLS on port 443.",
                    "Verify network connectivity and DNS resolution.",
                ],
                "source": "probe",
                "run_id": run_id,
            }
        )

    try:
        response = requests.head(target_url, timeout=10, allow_redirects=True)
        hsts = response.headers.get("strict-transport-security")
        if hsts:
            findings.append(
                {
                    "asset": {"type": "url", "value": target_url},
                    "category": "TLS",
                    "check": "tls_probe",
                    "severity": "Low",
                    "summary": "HSTS header is present.",
                    "evidence": {
                        "notes": hsts,
                        "headers": dict(response.headers),
                        "screenshot_path": None,
                        "artifact_path": None,
                    },
                    "remediation": [
                        "Review HSTS policy configuration.",
                    ],
                    "source": "probe",
                    "run_id": run_id,
                }
            )
        else:
            findings.append(
                {
                    "asset": {"type": "url", "value": target_url},
                    "category": "TLS",
                    "check": "tls_probe",
                    "severity": "Medium",
                    "summary": "HSTS header is missing.",
                    "evidence": {
                        "notes": "strict-transport-security header not found",
                        "headers": dict(response.headers),
                        "screenshot_path": None,
                        "artifact_path": None,
                    },
                    "remediation": [
                        "Enable HTTP Strict Transport Security (HSTS).",
                    ],
                    "source": "probe",
                    "run_id": run_id,
                }
            )
    except Exception as exc:  # noqa: BLE001
        findings.append(
            {
                "asset": {"type": "url", "value": target_url},
                "category": "TLS",
                "check": "tls_probe",
                "severity": "Low",
                "summary": "Unable to assess HSTS header.",
                "evidence": {
                    "notes": str(exc),
                    "headers": {},
                    "screenshot_path": None,
                    "artifact_path": None,
                },
                "remediation": [
                    "Ensure the URL is reachable over HTTPS.",
                ],
                "source": "probe",
                "run_id": run_id,
            }
        )
    return findings
