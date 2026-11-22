"""Data contracts at the controller's boundary.

IncidentReport is what arrives (alerts + evidence, nothing else). It is deliberately the
controller's own model, not the lab's: the lab can change without the controller noticing,
and the controller can never pick up a ground-truth field by accident.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class AlertIn(BaseModel):
    name: str
    detail: str
    fired_at: str | None = None


class JobRunIn(BaseModel):
    job: str
    status: str
    finished_at: str | None = None
    error: str | None = None


class IncidentReport(BaseModel):
    incident_id: str
    opened_at: str | None = None
    title: str
    alerts: list[AlertIn]
    logs: dict[str, str] = Field(default_factory=dict)  # service -> recent log lines
    recent_job_runs: list[JobRunIn] = Field(default_factory=list)

    @classmethod
    def from_lab_record(cls, record: dict) -> IncidentReport:
        """Adapt one line of lab/incidents.jsonl, keeping only alert and evidence fields."""
        evidence = record.get("evidence") or {}
        return cls(
            incident_id=record["incident_id"],
            opened_at=record.get("opened_at"),
            title=record["title"],
            alerts=[AlertIn(name=a["name"], detail=a["detail"], fired_at=a.get("fired_at"))
                    for a in record.get("alerts", [])],
            logs=evidence.get("logs") or {},
            recent_job_runs=[JobRunIn(**{k: r.get(k) for k in ("job", "status", "finished_at", "error")})
                             for r in evidence.get("recent_job_runs") or []],
        )


def load_incidents(path: Path) -> list[IncidentReport]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [IncidentReport.from_lab_record(json.loads(line)) for line in lines if line.strip()]


class IncidentFeatures(BaseModel):
    """Everything downstream blocks reason over."""

    incident_id: str
    alerts: list[str]  # alert names, in firing order
    signals: dict[str, int]  # observed evidence type -> number of distinct lines showing it
    components: list[str]  # parts of the platform that are alerting or logging errors
    error_lines: list[str]  # masked, de-duplicated "[service] message" lines
    text: str  # masked summary for embedding models and LLM prompts
