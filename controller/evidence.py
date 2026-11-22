"""Evidence bundle: what each expert agent's read-only tools return for an incident.

The controller routes on alerts + logs only (cheap and shared). Each domain's expert gets its
own slice of deeper evidence, the way its tools would answer at investigation time. This is
what lets a data expert and an infra expert tell apart two incidents whose logs are identical
(a heavy analyst query vs a rogue process): the query list and the process list differ.

Synthetic incidents carry a pre-computed bundle under evidence.domain. On the live sandbox the
same shapes will come from real tool calls. The router never reads this (Block 1 ignores it).
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class PathProbe(BaseModel):
    path: str  # e.g. "source_primary"
    target: str  # service the path leads to
    p50_ms: float | None = None
    error: str | None = None


class NetworkEvidence(BaseModel):
    probes: list[PathProbe] = Field(default_factory=list)


class ActiveQuery(BaseModel):
    pid: int
    usename: str
    application_name: str
    client_addr: str | None = None  # None = local socket
    state: str
    wait_event_type: str | None = None
    duration_s: float
    query: str


class LockInfo(BaseModel):
    pid: int
    relation: str
    mode: str
    granted: bool
    blocked_by: list[int] = Field(default_factory=list)


class DbtFailure(BaseModel):
    node: str
    status: str
    message: str


class DataEvidence(BaseModel):
    active_queries: list[ActiveQuery] = Field(default_factory=list)
    locks: list[LockInfo] = Field(default_factory=list)
    connections_by_app: dict[str, int] = Field(default_factory=dict)
    last_dbt_failures: list[DbtFailure] = Field(default_factory=list)
    source_sample_fields: list[str] = Field(default_factory=list)


class ContainerStat(BaseModel):
    service: str
    status: str = "running"
    cpu_pct: float
    mem_mb: float
    mem_limit_mb: float | None = None
    restarts: int = 0
    oom_kills: int = 0


class ProcessInfo(BaseModel):
    pid: int
    cpu_pct: float
    command: str


class DiskUsage(BaseModel):
    service: str
    mount: str
    used_pct: float
    largest_files: list[str] = Field(default_factory=list)


class ChangeEvent(BaseModel):
    at: str
    component: str
    change: str


class InfraEvidence(BaseModel):
    containers: list[ContainerStat] = Field(default_factory=list)
    top_processes: dict[str, list[ProcessInfo]] = Field(default_factory=dict)
    disks: list[DiskUsage] = Field(default_factory=list)
    recent_changes: list[ChangeEvent] = Field(default_factory=list)


class ClientOrigin(BaseModel):
    usename: str
    application_name: str
    client_addr: str
    sessions: int


class SecurityEvidence(BaseModel):
    db_client_origins: list[ClientOrigin] = Field(default_factory=list)
    failed_logins_by_ip: dict[str, int] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    network: NetworkEvidence
    data: DataEvidence
    infra: InfraEvidence
    security: SecurityEvidence

    def for_domain(self, domain: str) -> BaseModel:
        return getattr(self, domain)


def load_evidence_bundles(path: Path) -> dict[str, EvidenceBundle]:
    bundles = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        domain = (record.get("evidence") or {}).get("domain")
        if domain:
            bundles[record["incident_id"]] = EvidenceBundle.model_validate(domain)
    return bundles
