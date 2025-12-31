from __future__ import annotations

import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    target_url: str
    status: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    max_duration_sec: int
    progress: int = 0
    error_message: Optional[str] = None
    artifacts_path: Optional[str] = None


class Finding(SQLModel, table=True):
    id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    category: str
    check: str
    severity: str
    summary: str
    asset: str
    evidence: str
    remediation: str
    source: str
