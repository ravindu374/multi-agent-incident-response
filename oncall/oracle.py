"""Health oracle: collect a snapshot of the platform, then judge it against the SLOs.

`collect` does I/O; `evaluate` is a pure function of (snapshot, settings) so it can be
unit-tested. A run counts as "resolved" only when every check passes, which is the success
signal the routing experiments learn from (never an LLM's opinion).
"""
from __future__ import annotations

import time
from collections import Counter as _Counter

import httpx
from pydantic import BaseModel, Field

from .lab import Lab
from .settings import Settings

ALERT_NAMES = {
    "freshness": "OrdersMartStale",
    "ingest_job": "IngestJobFailing",
    "transform_job": "TransformJobFailing",
    "api_up": "DashboardDown",
    "api_latency": "DashboardSlow",
    "db_connections": "WarehouseConnectionsSaturated",
    "unexpected_db_clients": "UnexpectedWarehouseClient",
    "login_failures": "LoginFailureSpike",
    "monitoring": "MonitoringGap",
}
CHECK_ORDER = list(ALERT_NAMES)


class JobStatus(BaseModel):
    failures_since_success: int = 0
    last_success_age_s: float | None = None
    last_error: str | None = None


class Snapshot(BaseModel):
    taken_at: float = Field(default_factory=time.time)
    freshness_lag_s: float | None = None
    jobs: dict[str, JobStatus] = Field(default_factory=dict)
    api_up: bool = False
    api_detail: str = ""
    api_latency_p95_s: float | None = None
    api_errors: int = 0
    api_probes: int = 0
    db_connections: int | None = None
    db_max_connections: int | None = None
    # application_name -> sessions using the etl role
    unexpected_clients: dict[str, int] = Field(default_factory=dict)
    login_failures_60s: int | None = None
    collection_errors: list[str] = Field(default_factory=list)


class Check(BaseModel):
    name: str
    ok: bool
    detail: str

    @property
    def alert(self) -> str:
        return ALERT_NAMES[self.name]


class HealthReport(BaseModel):
    taken_at: float
    checks: list[Check]

    @property
    def healthy(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failing(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


# -- evaluation (pure) -----------------------------------------------------------------

def _job_check(name: str, job: str, status: JobStatus | None, stale_after_s: float, streak: int) -> Check:
    if status is None or status.last_success_age_s is None:
        return Check(name=name, ok=False, detail=f"no successful {job} run recorded")
    if status.failures_since_success >= streak:
        return Check(name=name, ok=False, detail=(
            f"{job} failed {status.failures_since_success} runs in a row; last error: {status.last_error}"))
    if status.last_success_age_s > stale_after_s:
        return Check(name=name, ok=False, detail=(
            f"no successful {job} run for {status.last_success_age_s:.0f}s (expected every <{stale_after_s:.0f}s)"))
    return Check(name=name, ok=True, detail=f"last {job} success {status.last_success_age_s:.0f}s ago")


def evaluate(snap: Snapshot, s: Settings) -> HealthReport:
    checks: list[Check] = []

    if snap.freshness_lag_s is None:
        checks.append(Check(name="freshness", ok=False, detail="orders_mart has no data (not built or unreadable)"))
    else:
        checks.append(Check(
            name="freshness", ok=snap.freshness_lag_s <= s.freshness_slo_s,
            detail=f"orders_mart is {snap.freshness_lag_s:.0f}s behind the source (SLO {s.freshness_slo_s:.0f}s)",
        ))

    checks.append(_job_check("ingest_job", "ingest", snap.jobs.get("ingest"), s.ingest_stale_after_s,
                             s.job_failure_streak))
    checks.append(_job_check("transform_job", "transform", snap.jobs.get("transform"), s.transform_stale_after_s,
                             s.job_failure_streak))

    checks.append(Check(name="api_up", ok=snap.api_up,
                        detail="dashboard API healthy" if snap.api_up else f"dashboard API down: {snap.api_detail}"))

    if not snap.api_up or snap.api_latency_p95_s is None:
        checks.append(Check(name="api_latency", ok=True, detail="skipped (dashboard API down)"))
    elif snap.api_errors:
        checks.append(Check(name="api_latency", ok=False, detail=(
            f"{snap.api_errors}/{snap.api_probes} dashboard revenue requests failed; {snap.api_detail}".rstrip("; "))))
    else:
        checks.append(Check(
            name="api_latency", ok=snap.api_latency_p95_s <= s.api_latency_slo_s,
            detail=f"dashboard p95 latency {snap.api_latency_p95_s:.2f}s over {snap.api_probes} probes "
                   f"(SLO {s.api_latency_slo_s:.1f}s)",
        ))

    if snap.db_connections is not None and snap.db_max_connections:
        share = snap.db_connections / snap.db_max_connections
        checks.append(Check(
            name="db_connections", ok=share <= s.db_connection_limit,
            detail=f"warehouse connections at {snap.db_connections}/{snap.db_max_connections} ({share:.0%})",
        ))

    if snap.unexpected_clients:
        apps = ", ".join(f"{app or '<none>'} x{n}" for app, n in sorted(snap.unexpected_clients.items()))
        checks.append(Check(name="unexpected_db_clients", ok=False,
                            detail=f"service role etl used by unexpected application(s): {apps}"))
    else:
        checks.append(Check(name="unexpected_db_clients", ok=True, detail="etl role used only by ingest"))

    if snap.login_failures_60s is not None:
        checks.append(Check(
            name="login_failures", ok=snap.login_failures_60s <= s.login_failure_limit,
            detail=f"{snap.login_failures_60s} failed dashboard logins in the last 60s "
                   f"(limit {s.login_failure_limit})",
        ))

    if snap.collection_errors:
        checks.append(Check(name="monitoring", ok=False, detail="; ".join(snap.collection_errors)))

    return HealthReport(taken_at=snap.taken_at, checks=checks)


# -- collection (I/O) --------------------------------------------------------------------

FRESHNESS_SQL = "SELECT EXTRACT(EPOCH FROM now() - max(_source_max_created_at)) FROM marts.orders_mart"
JOBS_SQL = """
WITH last_ok AS (
    SELECT job, max(finished_at) AS t FROM ops.job_runs WHERE status = 'success' GROUP BY job
), latest AS (
    SELECT DISTINCT ON (job) job, error FROM ops.job_runs ORDER BY job, finished_at DESC
)
SELECT j.job,
       count(*) FILTER (WHERE j.status = 'failed' AND j.finished_at > coalesce(l.t, '-infinity')),
       EXTRACT(EPOCH FROM now() - l.t),
       max(latest.error)
FROM ops.job_runs j
LEFT JOIN last_ok l USING (job)
LEFT JOIN latest USING (job)
WHERE j.finished_at > now() - interval '1 hour'
GROUP BY j.job, l.t
"""
CONNECTIONS_SQL = (
    "SELECT count(*), current_setting('max_connections')::int FROM pg_stat_activity "
    "WHERE backend_type = 'client backend'"
)
ETL_CLIENTS_SQL = (
    "SELECT application_name FROM pg_stat_activity "
    "WHERE usename = 'etl' AND backend_type = 'client backend' AND application_name <> 'ingest'"
)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text.splitlines()[0] if text else ''}"


def _collect_warehouse(lab: Lab, snap: Snapshot) -> None:
    import psycopg

    with lab.connect() as conn:
        try:
            lag = conn.execute(FRESHNESS_SQL).fetchone()[0]
            snap.freshness_lag_s = float(lag) if lag is not None else None
        except psycopg.errors.UndefinedTable:
            snap.freshness_lag_s = None
        for job, fails, ok_age, error in conn.execute(JOBS_SQL).fetchall():
            snap.jobs[job] = JobStatus(
                failures_since_success=int(fails),
                last_success_age_s=float(ok_age) if ok_age is not None else None,
                last_error=error,
            )
        snap.db_connections, snap.db_max_connections = conn.execute(CONNECTIONS_SQL).fetchone()
        apps = [row[0] for row in conn.execute(ETL_CLIENTS_SQL).fetchall()]
        snap.unexpected_clients = dict(_Counter(apps))


def _collect_api(lab: Lab, snap: Snapshot, probes: int = 5) -> None:
    base = lab.settings.metrics_api_url
    with httpx.Client(timeout=5.0) as client:
        try:
            resp = client.get(f"{base}/healthz")
            snap.api_up = resp.status_code == 200
            snap.api_detail = "" if snap.api_up else f"/healthz returned {resp.status_code}"
        except httpx.HTTPError as exc:
            snap.api_up, snap.api_detail = False, _first_line(exc)
        if not snap.api_up:
            return

        try:
            snap.login_failures_60s = int(client.get(f"{base}/internal/stats").json()["login_failures_60s"])
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            snap.collection_errors.append(f"login stats: {_first_line(exc)}")

        latencies: list[float] = []
        errors, last_error = 0, ""
        for _ in range(probes):
            started = time.perf_counter()
            try:
                token = lab.dashboard_token(client)
                resp = client.get(f"{base}/api/revenue", headers={"Authorization": f"Bearer {token}"})
                if resp.status_code == 401:  # API restarted and issued a new token
                    token = lab.dashboard_token(client, refresh=True)
                    resp = client.get(f"{base}/api/revenue", headers={"Authorization": f"Bearer {token}"})
                if resp.status_code != 200:
                    errors += 1
                    last_error = f"HTTP {resp.status_code}"
            except httpx.HTTPError as exc:
                errors += 1
                last_error = _first_line(exc)
            latencies.append(time.perf_counter() - started)
        latencies.sort()
        snap.api_probes = probes
        snap.api_errors = errors
        snap.api_latency_p95_s = latencies[int(0.95 * (len(latencies) - 1))]
        if errors:
            snap.api_detail = f"last error: {last_error}"


def collect(lab: Lab) -> Snapshot:
    snap = Snapshot()
    try:
        _collect_warehouse(lab, snap)
    except Exception as exc:
        snap.collection_errors.append(f"warehouse: {_first_line(exc)}")
    _collect_api(lab, snap)
    return snap


def observe(lab: Lab) -> HealthReport:
    return evaluate(collect(lab), lab.settings)
