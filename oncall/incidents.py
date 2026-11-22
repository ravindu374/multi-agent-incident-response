"""Incident detection: turn a stream of health reports into opened / updated / resolved incidents.

An Incident is what the controller will receive in later phases: alerts and evidence only,
never the injected fault. Ground truth is linked separately in lab/ground_truth.jsonl.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .lab import Lab, new_id, utcnow_iso
from .oracle import CHECK_ORDER, HealthReport

EVIDENCE_SERVICES = ("ingest", "transform", "metrics-api")
RECENT_RUNS_SQL = """
SELECT job, status, finished_at, error FROM ops.job_runs
ORDER BY finished_at DESC LIMIT 8
"""


class Alert(BaseModel):
    name: str
    check: str
    detail: str
    fired_at: str


class Incident(BaseModel):
    incident_id: str
    opened_at: str
    resolved_at: str | None = None
    title: str
    alerts: list[Alert] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        return "; ".join(f"{a.name}: {a.detail}" for a in self.alerts)


class Event(BaseModel):
    kind: Literal["opened", "updated", "resolved"]
    incident: Incident
    new_alerts: list[Alert] = Field(default_factory=list)


def _rank(check_name: str) -> int:
    return CHECK_ORDER.index(check_name) if check_name in CHECK_ORDER else len(CHECK_ORDER)


class IncidentDetector:
    """One incident at a time. Opens on the first failing report and resolves after
    `resolve_after` consecutive healthy reports (so a flapping check does not close it)."""

    def __init__(self, resolve_after: int = 2):
        self.current: Incident | None = None
        self._healthy_streak = 0
        self._resolve_after = resolve_after

    def observe(self, report: HealthReport, now: str | None = None) -> list[Event]:
        now = now or utcnow_iso()
        failing = sorted(report.failing, key=lambda c: _rank(c.name))

        if self.current is None:
            if not failing:
                return []
            alerts = [Alert(name=c.alert, check=c.name, detail=c.detail, fired_at=now) for c in failing]
            self.current = Incident(incident_id=new_id("inc"), opened_at=now, title=alerts[0].name, alerts=alerts)
            self._healthy_streak = 0
            return [Event(kind="opened", incident=self.current, new_alerts=alerts)]

        if failing:
            self._healthy_streak = 0
            known = {a.check for a in self.current.alerts}
            new = [Alert(name=c.alert, check=c.name, detail=c.detail, fired_at=now)
                   for c in failing if c.name not in known]
            if not new:
                return []
            self.current.alerts.extend(new)
            return [Event(kind="updated", incident=self.current, new_alerts=new)]

        self._healthy_streak += 1
        if self._healthy_streak < self._resolve_after:
            return []
        incident, self.current, self._healthy_streak = self.current, None, 0
        incident.resolved_at = now
        return [Event(kind="resolved", incident=incident)]


def capture_evidence(lab: Lab, tail: int = 25) -> dict[str, Any]:
    """What an engineer would glance at first: recent job runs and the last log lines."""
    evidence: dict[str, Any] = {"logs": {}, "recent_job_runs": []}
    for service in EVIDENCE_SERVICES:
        evidence["logs"][service] = lab.compose.logs(service, tail=tail)
    try:
        evidence["recent_job_runs"] = [
            {"job": job, "status": status, "finished_at": str(finished), "error": error}
            for job, status, finished, error in lab.sql(RECENT_RUNS_SQL)
        ]
    except Exception as exc:
        evidence["recent_job_runs_error"] = f"{type(exc).__name__}: {str(exc).strip()[:200]}"
    return evidence
