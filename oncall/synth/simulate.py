"""Simulate one incident: job runs and platform state over time, judged by the real oracle and
incident detector, then captured the way `oncall validate` captures a real incident."""
from __future__ import annotations

from dataclasses import dataclass

from ..incidents import IncidentDetector
from ..oracle import JobStatus, Snapshot, evaluate
from ..settings import Settings
from .evidence import baseline_bundle
from .noise import Noise, apply_evidence_noise
from .scenarios import READ_TIMEOUT, Effects
from .world import GATEWAY, World

TICK_S = 5.0
T_BEGIN = -90.0
DETECT_LIMIT_S = 300.0
SETTLE_S = 30.0
LOG_TAIL = 25
ORDERS_PER_S = 5.0
UVICORN_START = (
    "INFO:     Started server process [1]",
    "INFO:     Waiting for application startup.",
    "INFO:     Application startup complete.",
    "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
)
DBT_FAIL_DURATION = {"test_fail": (6.0, 9.0), "statement_timeout": (15.1, 16.8), "lock_timeout": (5.6, 9.0),
                     "slots": (0.4, 1.5), "oom": (1.8, 4.5)}


@dataclass
class Run:
    job: str
    start: float
    end: float
    status: str
    error: str | None
    recorded: bool


@dataclass
class SimResult:
    record: dict  # lab/incidents.jsonl shape, without the final incident_id
    t_open: float


class _Log:
    def __init__(self):
        self.events: dict[str, list[tuple[float, int, str | tuple[str, str]]]] = {
            "ingest": [], "transform": [], "metrics-api": []}
        self._seq = 0

    def add(self, service: str, t: float, entry: str | tuple[str, str]) -> None:
        self._seq += 1
        self.events[service].append((t, self._seq, entry))

    def tail(self, world: World, service: str, until: float, n: int = LOG_TAIL) -> str:
        lines: list[str] = []
        for t, _, entry in sorted(e for e in self.events[service] if e[0] <= until):
            text = entry if isinstance(entry, str) else f"{world.stamp(t)} {service} {entry[0]} {entry[1]}"
            lines.extend(text.split("\n"))
        return "\n".join(lines[-n:]) + ("\n" if lines else "")


def dbt_console(world: World, t: float, kind: str, params: dict) -> list[str]:
    c = world.clock(t)
    head = [f"{c}  Running with dbt=1.9.1", f"{c}  Registered adapter: postgres=1.9.0",
            f"{c}  Found 2 models, 4 data tests, 1 source, 433 macros", f"{c}  Concurrency: 1 threads (target='sandbox')"]
    if kind == "oom":
        return head[:2]
    if kind == "test_fail":
        nulls, dups = params["nulls"], params["dups"]
        errors = 1 + bool(dups)
        unique = (f"{c}  4 of 6 FAIL {dups} unique_stg_orders_order_id ....................... [FAIL {dups} in 0.04s]"
                  if dups else f"{c}  4 of 6 PASS unique_stg_orders_order_id ......................... [PASS in 0.04s]")
        return head + [
            f"{c}  1 of 6 OK created sql view model staging.stg_orders ................. [CREATE VIEW in 0.06s]",
            f"{c}  3 of 6 FAIL {nulls} not_null_stg_orders_amount ........................ [FAIL {nulls} in 0.03s]",
            unique,
            f"{c}  6 of 6 SKIP relation marts.orders_mart ............................... [SKIP]",
            f"{c}  Failure in test not_null_stg_orders_amount (models/staging/_stg_models.yml)",
            f"{c}    Got {nulls} results, configured to fail if != 0",
            f"{c}  Done. PASS={4 - errors} WARN=0 ERROR={errors} SKIP=1 TOTAL=6",
        ]
    if kind == "statement_timeout":
        return head + [
            f"{c}  1 of 6 OK created sql view model staging.stg_orders ................. [CREATE VIEW in 0.31s]",
            f"{c}  6 of 6 ERROR creating sql table model marts.orders_mart ................ [ERROR in 15.02s]",
            f"{c}  Database Error in model orders_mart (models/marts/orders_mart.sql)",
            f"{c}    canceling statement due to statement timeout",
            f"{c}  Done. PASS=5 WARN=0 ERROR=1 SKIP=0 TOTAL=6",
        ]
    if kind == "lock_timeout":
        return head + [
            f"{c}  1 of 6 ERROR creating sql view model staging.stg_orders ................ [ERROR in 5.01s]",
            f"{c}  Database Error in model stg_orders (models/staging/stg_orders.sql)",
            f"{c}    canceling statement due to lock timeout",
            f"{c}  Done. PASS=0 WARN=0 ERROR=1 SKIP=5 TOTAL=6",
        ]
    if kind == "slots":
        return head + [
            f"{c}  1 of 6 ERROR creating sql view model staging.stg_orders ................ [ERROR in 0.01s]",
            f"{c}  Database Error in model stg_orders (models/staging/stg_orders.sql)",
            f"{c}    {params['conn']}",
            f"{c}  Done. PASS=0 WARN=0 ERROR=1 SKIP=5 TOTAL=6",
        ]
    raise ValueError(f"unknown dbt console block {kind!r}")


def _simulate_ingest(world, e: Effects, s: Settings, noise: Noise, log: _Log, until: float):
    r = world.rng
    runs: list[Run] = []
    data_until: list[tuple[float, float]] = []  # (run end, data available up to)
    last_end = T_BEGIN - r.uniform(0, 15)
    runs.append(Run("ingest", last_end - 0.5, last_end, "success", None, True))
    data_until.append((last_end, last_end))
    batch = r.randint(50, 900)
    transient = noise.transient_ingest_failure_at
    t = last_end + s.ingest_interval_s
    while t < until:
        active = t >= e.onset_s
        if transient is not None and not active and t >= transient:
            fail, error, run_s, recorded, transient = True, READ_TIMEOUT, 5.0, True, None
        elif active and e.ingest_error and r.random() < e.ingest_fail_prob:
            fail, error, run_s, recorded = True, e.ingest_error, e.ingest_run_s, e.ingest_recorded
        else:
            fail, error, run_s, recorded = False, None, r.uniform(0.2, 0.9), True
        end = t + run_s * r.uniform(0.95, 1.1)
        if fail:
            runs.append(Run("ingest", t, end, "failed", error, recorded))
            log.add("ingest", end, ("ERROR", f"run failed: {error}"))
            if not recorded:
                log.add("ingest", end + 0.01, ("WARNING", f"could not record failed run in ops.job_runs: {error}"))
        else:
            rows = max(1, min(2000, int(ORDERS_PER_S * (end - data_until[-1][1]) * r.uniform(0.92, 1.08))))
            batch += 1
            runs.append(Run("ingest", t, end, "success", None, True))
            data_until.append((end, end))
            log.add("ingest", end, ("INFO", f"run ok: {rows} rows (batch {batch})"))
        t = end + s.ingest_interval_s
    return runs, data_until


def _simulate_transform(world, e: Effects, s: Settings, data_until, log: _Log, until: float):
    r = world.rng

    def available(t: float) -> float:
        return max(d for end, d in data_until if end <= t) if any(end <= t for end, _ in data_until) else data_until[0][1]

    runs: list[Run] = []
    marts: list[tuple[float, float]] = []  # (run end, mart has data up to)
    last_end = T_BEGIN - r.uniform(0, 30)
    runs.append(Run("transform", last_end - 6, last_end, "success", None, True))
    marts.append((last_end, last_end - r.uniform(10, 25)))
    t = last_end + s.transform_interval_s
    while t < until:
        active = t >= e.onset_s
        if active and e.transform_error and r.random() < e.transform_fail_prob:
            duration = r.uniform(*DBT_FAIL_DURATION.get(e.dbt or "", (2.0, 6.0)))
            end = t + duration
            runs.append(Run("transform", t, end, "failed", e.transform_error, e.transform_recorded))
            if e.dbt:
                for line in dbt_console(world, end - 0.05, e.dbt, e.dbt_params):
                    log.add("transform", end - 0.05, line)
            log.add("transform", end, ("ERROR", f"dbt build failed in {duration:.1f}s: {e.transform_error}"))
            if not e.transform_recorded:
                log.add("transform", end + 0.01, ("WARNING", f"could not record run in ops.job_runs: "
                                                             f"{e.transform_record_error}"))
        else:
            duration = r.uniform(*(e.transform_duration if active and e.transform_duration else (5.0, 9.0)))
            end = t + duration
            runs.append(Run("transform", t, end, "success", None, True))
            marts.append((end, available(t)))
            log.add("transform", end, ("INFO", f"dbt build ok in {duration:.1f}s"))
        t = end + s.transform_interval_s
    return runs, marts


def _job_status(runs: list[Run], t: float) -> JobStatus | None:
    recorded = [x for x in runs if x.recorded and x.end <= t]
    if not recorded:
        return None
    successes = [x.end for x in recorded if x.status == "success"]
    last_ok = max(successes) if successes else None
    fails = sum(1 for x in recorded if x.status == "failed" and (last_ok is None or x.end > last_ok)
                and x.end > t - 3600)
    latest = max(recorded, key=lambda x: x.end)
    return JobStatus(failures_since_success=fails, last_success_age_s=None if last_ok is None else t - last_ok,
                     last_error=latest.error)


def simulate(world: World, e: Effects, settings: Settings, noise: Noise) -> SimResult | None:
    """Return the captured incident, or None if nothing was detected in time (or a false alarm
    fired before the fault started)."""
    r = world.rng
    horizon = DETECT_LIMIT_S + SETTLE_S + 30
    log = _Log()
    for line in UVICORN_START:
        log.add("metrics-api", T_BEGIN - 3600, line)
    ingest_runs, data_until = _simulate_ingest(world, e, settings, noise, log, horizon)
    transform_runs, marts = _simulate_transform(world, e, settings, data_until, log, horizon)
    for t in noise.benign_logins:
        log.add("metrics-api", t, ("WARNING", f"failed login user=analyst client={GATEWAY}"))

    detector = IncidentDetector()
    incident, t_open = None, None
    t = T_BEGIN
    while t <= horizon:
        active = t >= e.onset_s
        mart_until = max(m for end, m in marts if end <= t)

        api_up, api_detail, p95, errors = True, "", r.uniform(0.03, 0.25), 0
        if active and e.api_down:
            api_up, api_detail = False, e.api_down
        elif active:
            if e.api_p95:
                p95 = r.uniform(*e.api_p95)
            if e.api_errors:
                errors = r.randint(*e.api_errors)
                api_detail = "last error: HTTP 503"
                for _ in range(errors if e.api_error_log else 0):
                    log.add("metrics-api", t + r.uniform(0, TICK_S), ("ERROR", e.api_error_log))
        if active and e.metrics_api_lines:
            for level, message in e.metrics_api_lines(world, t):
                log.add("metrics-api", t + r.uniform(0, TICK_S), (level, message))

        logins = None
        if api_up:
            logins = sum(1 for x in noise.benign_logins if t - 60 < x <= t)
            if active and e.login_failures:
                logins += r.randint(*e.login_failures)

        snap = Snapshot(
            taken_at=t,
            freshness_lag_s=t - mart_until,
            jobs={job: st for job, st in (("ingest", _job_status(ingest_runs, t)),
                                          ("transform", _job_status(transform_runs, t))) if st},
            api_up=api_up, api_detail=api_detail,
            api_latency_p95_s=p95 if api_up else None, api_errors=errors, api_probes=5 if api_up else 0,
            db_connections=r.randint(*e.db_connections) if active and e.db_connections else r.randint(4, 9),
            db_max_connections=40,
            unexpected_clients=dict(e.unexpected_clients) if active else {},
            login_failures_60s=logins,
        )
        for event in detector.observe(evaluate(snap, settings), now=world.iso(t)):
            if event.kind == "opened":
                if t < e.onset_s:
                    return None  # noise alone tripped an alert: discard
                incident, t_open = event.incident, t
        if incident is None and t > DETECT_LIMIT_S:
            return None
        if t_open is not None and t >= t_open + SETTLE_S:
            break
        t += TICK_S

    if incident is None:
        return None
    t_capture = t
    bundle = baseline_bundle(world)
    if e.evidence:
        e.evidence(bundle, world, t_capture)
    apply_evidence_noise(bundle, world, noise, t_capture)

    recorded = sorted((x for x in ingest_runs + transform_runs if x.recorded and x.end <= t_capture),
                      key=lambda x: x.end, reverse=True)[:8]
    record = {
        "incident_id": "",
        "opened_at": incident.opened_at,
        "resolved_at": None,
        "title": incident.title,
        "alerts": [a.model_dump() for a in incident.alerts],
        "evidence": {
            "logs": {svc: log.tail(world, svc, t_capture) for svc in ("ingest", "transform", "metrics-api")},
            "recent_job_runs": [{"job": x.job, "status": x.status, "finished_at": world.pg_time(x.end),
                                 "error": x.error} for x in recorded],
            "domain": bundle.model_dump(),
        },
    }
    return SimResult(record=record, t_open=t_open)
