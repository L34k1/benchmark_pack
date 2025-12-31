from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

Finding = Dict[str, Any]

SEVERITY_MAP = {
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
    "Informational": "Low",
}


def parse_zap_findings(run_id: str, json_path: Path) -> List[Finding]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text())
    findings: List[Finding] = []
    alerts = data.get("site", [{}])[0].get("alerts", [])
    for alert in alerts:
        severity = SEVERITY_MAP.get(alert.get("risk"), "Low")
        findings.append(
            {
                "asset": {"type": "url", "value": alert.get("url", "")},
                "category": alert.get("cweid", "OWASP"),
                "check": "zap_baseline",
                "severity": severity,
                "summary": alert.get("name", "ZAP alert"),
                "evidence": {
                    "notes": alert.get("description", ""),
                    "headers": {},
                    "screenshot_path": None,
                    "artifact_path": str(json_path),
                },
                "remediation": [alert.get("solution", "Review ZAP recommendation.")],
                "source": "zap",
                "run_id": run_id,
            }
        )
    return findings
