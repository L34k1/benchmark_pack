from __future__ import annotations

import datetime
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

from sqlmodel import select

from .db import get_session
from .models import Finding, Run
from .probes.headers_probe import run_headers_probe
from .probes.tls_probe import run_tls_probe
from .reporting.report_builder import build_report
from .zap.zap_parser import parse_zap_findings
from .zap.zap_runner import run_zap_baseline

FindingDict = Dict[str, Any]


class RunManager:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def create_run(self, target_url: str, max_duration_sec: int) -> str:
        run_id = str(uuid.uuid4())
        with get_session() as session:
            run = Run(
                id=run_id,
                target_url=target_url,
                status="queued",
                max_duration_sec=max_duration_sec,
            )
            session.add(run)
            session.commit()
        thread = threading.Thread(target=self._execute_run, args=(run_id,), daemon=True)
        thread.start()
        return run_id

    def _execute_run(self, run_id: str) -> None:
        run_dir = self.data_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._update_run(run_id, status="running", progress=5)
            zap_dir = run_dir / "zap"
            zap_dir.mkdir(parents=True, exist_ok=True)
            zap_json_path = zap_dir / "zap_report.json"
            zap_html_path = zap_dir / "zap_report.html"

            with get_session() as session:
                run = session.get(Run, run_id)
                target_url = run.target_url if run else ""
                max_duration_sec = run.max_duration_sec if run else 300

            self._update_run(run_id, progress=15)
            run_zap_baseline(
                target_url=target_url,
                json_path=zap_json_path,
                html_path=zap_html_path,
                max_duration_sec=max_duration_sec,
            )
            self._update_run(run_id, progress=45)

            findings: List[FindingDict] = []
            findings.extend(parse_zap_findings(run_id, zap_json_path))
            self._update_run(run_id, progress=60)

            findings.extend(run_tls_probe(run_id, target_url))
            self._update_run(run_id, progress=75)

            findings.extend(run_headers_probe(run_id, target_url))
            self._update_run(run_id, progress=85)

            normalized = self._assign_ids(findings)
            merged_path = run_dir / "merged_findings.json"
            merged_path.write_text(json.dumps(normalized, indent=2))

            report_path = run_dir / "report.html"
            build_report(target_url, normalized, report_path)

            self._persist_findings(run_id, normalized)
            self._update_run(
                run_id,
                status="finished",
                progress=100,
                artifacts_path=str(run_dir),
            )
        except Exception as exc:  # noqa: BLE001
            self._update_run(
                run_id,
                status="failed",
                error_message=str(exc),
            )

    def _update_run(self, run_id: str, **updates: Any) -> None:
        with get_session() as session:
            run = session.get(Run, run_id)
            if not run:
                return
            for key, value in updates.items():
                setattr(run, key, value)
            run.updated_at = datetime.datetime.utcnow()
            session.add(run)
            session.commit()

    def _assign_ids(self, findings: List[FindingDict]) -> List[FindingDict]:
        for index, finding in enumerate(findings, start=1):
            finding["id"] = f"F-{index:04d}"
        return findings

    def _persist_findings(self, run_id: str, findings: List[FindingDict]) -> None:
        with get_session() as session:
            existing = session.exec(select(Finding).where(Finding.run_id == run_id)).all()
            for item in existing:
                session.delete(item)
            for finding in findings:
                record = Finding(
                    id=finding["id"],
                    run_id=run_id,
                    category=finding["category"],
                    check=finding["check"],
                    severity=finding["severity"],
                    summary=finding["summary"],
                    asset=json.dumps(finding["asset"]),
                    evidence=json.dumps(finding["evidence"]),
                    remediation=json.dumps(finding["remediation"]),
                    source=finding["source"],
                )
                session.add(record)
            session.commit()

    def get_run(self, run_id: str) -> Run | None:
        with get_session() as session:
            return session.get(Run, run_id)

    def get_findings(self, run_id: str) -> List[FindingDict]:
        with get_session() as session:
            findings = session.exec(select(Finding).where(Finding.run_id == run_id)).all()
        return [
            {
                "id": finding.id,
                "run_id": finding.run_id,
                "category": finding.category,
                "check": finding.check,
                "severity": finding.severity,
                "summary": finding.summary,
                "asset": json.loads(finding.asset),
                "evidence": json.loads(finding.evidence),
                "remediation": json.loads(finding.remediation),
                "source": finding.source,
            }
            for finding in findings
        ]


def validate_url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return value
