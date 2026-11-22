"""Validation episodes: inject -> detect -> reference fix -> recover -> revert.

Phase 1's exit criterion is that every fault in the catalogue is (a) detected by the oracle,
(b) fixed by its reference actions from the allowlist, and (c) cleanly reverted. Each episode
also writes one labelled incident (incidents.jsonl + ground_truth.jsonl), which becomes
the routing dataset in Phase 2.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .compose import DockerError
from .faults import FAULTS, apply_reference_fix, inject_fault, revert_fault
from .incidents import Incident, IncidentDetector, capture_evidence
from .lab import Lab, utcnow_iso
from .oracle import HealthReport, observe

POLL_S = 5.0


class LabNotReady(RuntimeError):
    pass


class EpisodeResult(BaseModel):
    fault_id: str
    owner: str
    injection_id: str | None = None
    detected: bool = False
    detect_s: float | None = None
    incident_id: str | None = None
    title: str | None = None
    alerts: list[str] = Field(default_factory=list)
    fix_actions: list[str] = Field(default_factory=list)
    fix_error: str | None = None
    recovered: bool = False
    recover_s: float | None = None
    error: str | None = None


REQUIRED_SERVICES = ("warehouse", "orders-api", "toxiproxy", "ingest", "transform", "metrics-api")


def preflight(lab: Lab) -> list[str]:
    """Reasons the lab cannot run right now. An empty list means: go ahead."""
    problems = []
    try:
        import psycopg  # noqa: F401
    except ImportError:
        problems.append('psycopg is not installed. Run:  pip install -e ".[dev]"')
    try:
        running = lab.compose.compose("ps", "--services", "--filter", "status=running", timeout=30).split()
    except FileNotFoundError:
        return problems + ["the docker CLI is not installed or not on PATH"]
    except (DockerError, subprocess.TimeoutExpired) as exc:
        reason = str(exc).split("): ", 1)[-1].strip().splitlines()[-1][:200]
        return problems + [f"Docker is not reachable ({reason}). Start Docker, then:  docker compose up -d"]
    missing = [s for s in REQUIRED_SERVICES if s not in running]
    if missing:
        problems.append(f"sandbox services not running: {', '.join(missing)}. Start them with:  "
                        "docker compose --profile lab build  then  docker compose up -d")
    return problems


def wait_healthy(lab: Lab, timeout_s: float = 300, streak: int = 2,
                 on_report: Callable[[HealthReport], None] | None = None,
                 log: Callable[[str], None] | None = None, label: str = "waiting for a healthy sandbox",
                 progress_every_s: float = 30) -> bool:
    started = time.monotonic()
    last_progress = started
    consecutive = 0
    while time.monotonic() - started < timeout_s:
        report = observe(lab)
        if on_report:
            on_report(report)
        consecutive = consecutive + 1 if report.healthy else 0
        if consecutive >= streak:
            return True
        if log and time.monotonic() - last_progress >= progress_every_s:
            failing = ", ".join(c.name for c in report.failing) or "none (confirming)"
            log(f"{label}: {time.monotonic() - started:.0f}s, failing checks: {failing}")
            last_progress = time.monotonic()
        time.sleep(POLL_S)
    return False


def run_episode(lab: Lab, fault_id: str, overrides: dict[str, Any] | None = None, *,
                detect_timeout_s: float = 300, settle_s: float = 30, recover_timeout_s: float = 300,
                log: Callable[[str], None] = print) -> EpisodeResult:
    fault = FAULTS[fault_id]
    result = EpisodeResult(fault_id=fault_id, owner=fault.owner)
    log(f"[{fault_id}] waiting for a healthy baseline...")
    if not wait_healthy(lab, timeout_s=300, log=log, label=f"[{fault_id}] baseline"):
        raise LabNotReady("sandbox is not healthy before injection; check `python -m oncall health`")

    detector = IncidentDetector()
    incident: Incident | None = None
    try:
        record = inject_fault(lab, fault_id, overrides)
        result.injection_id = record["injection_id"]
        log(f"[{fault_id}] injected ({fault.title})")
        t_inject = time.monotonic()

        while incident is None and time.monotonic() - t_inject < detect_timeout_s:
            for event in detector.observe(observe(lab)):
                if event.kind == "opened":
                    incident = event.incident
                    result.detected, result.detect_s = True, round(time.monotonic() - t_inject, 1)
                    log(f"[{fault_id}] incident opened after {result.detect_s}s: {incident.summary}")
            if incident is None:
                time.sleep(POLL_S)

        if incident is not None:
            settle_end = time.monotonic() + settle_s  # let the follow-up alerts fire
            while time.monotonic() < settle_end:
                for event in detector.observe(observe(lab)):
                    if event.kind == "updated":
                        log(f"[{fault_id}] + {', '.join(a.name for a in event.new_alerts)}")
                time.sleep(POLL_S)
            incident.evidence = capture_evidence(lab)
            result.incident_id, result.title = incident.incident_id, incident.title
            result.alerts = [a.name for a in incident.alerts]
        else:
            log(f"[{fault_id}] NOT detected within {detect_timeout_s:.0f}s")

        t_fix = time.monotonic()
        try:
            result.fix_actions = apply_reference_fix(lab, fault_id, record["details"])
            log(f"[{fault_id}] reference fix applied: {result.fix_actions}")
        except Exception as exc:
            result.fix_error = f"{type(exc).__name__}: {exc}"
            log(f"[{fault_id}] reference fix FAILED: {result.fix_error}")

        result.recovered = wait_healthy(lab, timeout_s=recover_timeout_s, log=log, label=f"[{fault_id}] recovery")
        if result.recovered:
            result.recover_s = round(time.monotonic() - t_fix, 1)
        log(f"[{fault_id}] recovered: {result.recovered} ({result.recover_s}s)")

        if incident is not None:
            incident.resolved_at = utcnow_iso() if result.recovered else None
            lab.ledger.append("incidents", incident.model_dump())
            lab.ledger.append("ground_truth", {
                "incident_id": incident.incident_id,
                "injection_id": result.injection_id,
                "source": "validate",
                **fault.ground_truth(),
            })
    except LabNotReady:
        raise
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        log(f"[{fault_id}] episode error: {result.error}")
    finally:
        try:
            revert_fault(lab, fault_id)
        except Exception as exc:
            result.error = (result.error or "") + f" | revert failed: {type(exc).__name__}: {exc}"
        log(f"[{fault_id}] reverted; waiting for the baseline to return...")
        wait_healthy(lab, timeout_s=300, log=log, label=f"[{fault_id}] post-revert")

    lab.ledger.append("episodes", {"at": utcnow_iso(), **result.model_dump()})
    return result
