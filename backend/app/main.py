from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .db import init_db
from .runner import RunManager, validate_url

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

app = FastAPI(title="Web Security Scan (Passive)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

manager = RunManager(DATA_DIR)


class RunCreateRequest(BaseModel):
    target_url: str = Field(..., examples=["https://example.com"])
    max_duration_sec: int = Field(300, ge=60, le=1800)

    @field_validator("target_url")
    @classmethod
    def validate_target_url(cls, value: str) -> str:
        return validate_url(value)


class RunCreateResponse(BaseModel):
    run_id: str


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.post("/runs", response_model=RunCreateResponse)
async def create_run(payload: RunCreateRequest) -> RunCreateResponse:
    run_id = manager.create_run(payload.target_url, payload.max_duration_sec)
    return RunCreateResponse(run_id=run_id)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, str | int | None]:
    run = manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run.id,
        "target_url": run.target_url,
        "status": run.status,
        "progress": run.progress,
        "error_message": run.error_message,
        "artifacts_path": run.artifacts_path,
    }


@app.get("/runs/{run_id}/findings")
async def get_findings(run_id: str) -> Dict[str, object]:
    run = manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "findings": manager.get_findings(run_id)}


@app.get("/runs/{run_id}/report.html")
async def get_report(run_id: str) -> FileResponse:
    run = manager.get_run(run_id)
    if not run or not run.artifacts_path:
        raise HTTPException(status_code=404, detail="Report not found")
    report_path = Path(run.artifacts_path) / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="text/html")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
