from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Web Security Scan Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    th, td { border: 1px solid #ddd; padding: 8px; }
    th { background: #f4f4f4; text-align: left; }
    .badge { padding: 2px 6px; border-radius: 4px; color: white; }
    .Low { background: #2f855a; }
    .Medium { background: #dd6b20; }
    .High { background: #c53030; }
  </style>
</head>
<body>
  <h1>Web Security Scan Report</h1>
  <p><strong>Target:</strong> {target}</p>
  <p>This report is generated for authorized targets only. No destructive tests performed.</p>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Summary</th>
        <th>Severity</th>
        <th>Check</th>
        <th>Category</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <h2>Raw Findings (JSON)</h2>
  <pre>{raw_json}</pre>
</body>
</html>"""


def build_report(target: str, findings: List[Dict[str, Any]], output_path: Path) -> None:
    rows = []
    for finding in findings:
        rows.append(
            "<tr>"
            f"<td>{finding['id']}</td>"
            f"<td>{finding['summary']}</td>"
            f"<td><span class='badge {finding['severity']}'>{finding['severity']}</span></td>"
            f"<td>{finding['check']}</td>"
            f"<td>{finding['category']}</td>"
            "</tr>"
        )
    html = HTML_TEMPLATE.format(
        target=target,
        rows="\n".join(rows),
        raw_json=json.dumps(findings, indent=2),
    )
    output_path.write_text(html)
