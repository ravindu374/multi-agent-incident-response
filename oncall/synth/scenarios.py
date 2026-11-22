"""One scenario per fault: how the platform's state changes after onset, plus what each domain's
evidence shows. Each scenario draws its own random variant, so incidents of the same fault
differ in severity, timing and which symptoms appear.

Messages copy what the real services print (see sandbox/*/ and samples/build_samples.py).
They are best guesses until checked against real sandbox captures.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from controller.evidence import (
    ActiveQuery,
    ChangeEvent,
    ClientOrigin,
    DbtFailure,
    DiskUsage,
    EvidenceBundle,
    LockInfo,
    ProcessInfo,
)

from .evidence import add_container, container, probe
from .world import World

EvidenceFn = Callable[[EvidenceBundle, World, float], None]  # (bundle, world, capture time)
LinesFn = Callable[[World, float], list[tuple[str, str]]]  # (world, tick) -> [(level, message)]
# What a correct fix must act on, per intent: a list of alternatives, each {param: allowed values}
# (pid/pids are one key "pid"; "target" is the component). Read only AFTER the evidence was built,
# so it may only look up pids the evidence already created (generate() checks this).
FixTargetsFn = Callable[[World], dict[str, list[dict[str, list]]]]


@dataclass
class Effects:
    variant: str = "default"
    onset_s: float = 0.0
    # ingest job
    ingest_error: str | None = None
    ingest_fail_prob: float = 1.0
    ingest_recorded: bool = True  # False: the failure cannot even be written to ops.job_runs
    ingest_run_s: float = 0.5
    # transform job (dbt)
    transform_error: str | None = None
    transform_fail_prob: float = 1.0
    transform_recorded: bool = True
    transform_record_error: str | None = None
    transform_duration: tuple[float, float] | None = None  # successful runs under this fault
    dbt: str | None = None  # which dbt console block precedes the error line
    dbt_params: dict = field(default_factory=dict)
    # dashboard
    api_down: str | None = None
    api_p95: tuple[float, float] | None = None
    api_errors: tuple[int, int] | None = None
    api_error_log: str | None = None  # metrics-api ERROR line written for each failed request
    metrics_api_lines: LinesFn | None = None
    # warehouse / security signals
    db_connections: tuple[int, int] | None = None
    unexpected_clients: dict[str, int] = field(default_factory=dict)
    login_failures: tuple[int, int] | None = None
    evidence: EvidenceFn | None = None
    fix_targets: FixTargetsFn | None = None  # None: any well-formed fix of a valid intent counts


SCENARIOS: dict[str, Callable[[World], Effects]] = {}


def scenario(fault_id: str):
    def register(fn: Callable[[World], Effects]) -> Callable[[World], Effects]:
        SCENARIOS[fault_id] = fn
        return fn

    return register


READ_TIMEOUT = "ReadTimeout: timed out"
SLOTS = "FATAL:  remaining connection slots are reserved for roles with the SUPERUSER attribute"
INGEST_STATEMENT_TIMEOUT = "QueryCanceled: canceling statement due to statement timeout"
MART_TIMEOUT = ("model.orders_platform.orders_mart error: Database Error in model orders_mart "
                "(models/marts/orders_mart.sql)\n  canceling statement due to statement timeout")
STG_LOCK_TIMEOUT = ("model.orders_platform.stg_orders error: Database Error in model stg_orders "
                    "(models/staging/stg_orders.sql)\n  canceling statement due to lock timeout")
OOM = "dbt build was killed (SIGKILL, exit 137); the container may be out of memory"
NOTEBOOK_QUERY = ("SELECT sum(length(md5(o.order_id::text || g::text))) FROM raw.orders o "
                  "CROSS JOIN generate_series(1, 300) g")


REROUTE = {route: {"reroute_connection": [{"route": [route], "endpoint": ["secondary"]}]}
           for route in ("source", "warehouse")}


def pg_connect_error(w: World, via_proxy: bool, detail: str) -> str:
    host, port = (w.ips["toxiproxy"], 15432) if via_proxy else (w.ips["warehouse"], 5432)
    return f'connection failed: connection to server at "{host}", port {port} failed: {detail}'


def _cpu_pressure(w: World, e: Effects) -> None:
    """Shared by faults that starve the warehouse of CPU (heavy query, rogue process, bulk export)."""
    r = w.rng
    e.api_p95 = (r.uniform(1.05, 1.6), r.uniform(1.8, 3.2))
    e.transform_error, e.dbt = MART_TIMEOUT, "statement_timeout"
    e.transform_fail_prob = r.uniform(0.3, 0.9)
    e.transform_duration = (9.0, 14.5)
    if r.random() < 0.3:
        e.ingest_error, e.ingest_fail_prob, e.ingest_run_s = INGEST_STATEMENT_TIMEOUT, r.uniform(0.4, 0.8), 10.0
        e.variant += "+ingest_timeouts"


def _dbt_query_red_herring(w: World, b: EvidenceBundle, t: float) -> None:
    """dbt's own long-running build query, visible whenever the warehouse is slow."""
    b.data.active_queries.append(ActiveQuery(
        pid=w.pid("dbt-build"), usename="transformer", application_name="dbt", client_addr=w.ips["transform"],
        state="active", duration_s=round(w.rng.uniform(6, 14), 1),
        query='create table "warehouse"."marts"."orders_mart__dbt_tmp" as (with orders as (select * from ...'))


# -- network --------------------------------------------------------------------------------------

@scenario("N1")
def _source_latency(w: World) -> Effects:
    r = w.rng
    hard = r.random() < 0.7
    latency_ms = r.uniform(6000, 12000) if hard else r.uniform(4300, 6500)

    def evidence(b, w, t):
        probe(b, "source_primary").p50_ms = round(latency_ms)

    return Effects(variant="hard" if hard else "intermittent", onset_s=r.uniform(0, 10),
                   ingest_error=READ_TIMEOUT, ingest_fail_prob=1.0 if hard else r.uniform(0.55, 0.85),
                   ingest_run_s=5.0, evidence=evidence, fix_targets=lambda w: REROUTE["source"])


@scenario("N2")
def _warehouse_resets(w: World) -> Effects:
    error = "OperationalError: " + pg_connect_error(w, True, "server closed the connection unexpectedly")

    def evidence(b, w, t):
        p = probe(b, "warehouse_primary")
        p.p50_ms, p.error = None, "connection reset by peer"

    return Effects(onset_s=w.rng.uniform(0, 10), ingest_error=error, ingest_recorded=False, ingest_run_s=0.1,
                   evidence=evidence, fix_targets=lambda w: REROUTE["warehouse"])


@scenario("N3")
def _source_unresolvable(w: World) -> Effects:
    r = w.rng
    host = r.choice(["orders-source.internal", "orders-api.prod.internal", "src-orders.lan"])
    logged = r.random() < 0.7

    def evidence(b, w, t):
        p = probe(b, "source_primary")
        p.p50_ms, p.error = None, f"name resolution failed for {host}"
        if logged:
            b.infra.recent_changes.append(ChangeEvent(at=w.iso(-r.uniform(5, 180)), component="ingest",
                                                      change="config updated: source.endpoints.primary"))

    return Effects(variant="change_logged" if logged else "change_not_logged", onset_s=r.uniform(0, 10),
                   ingest_error="ConnectError: [Errno -2] Name or service not known", ingest_run_s=0.05,
                   evidence=evidence,
                   fix_targets=lambda w: {**REROUTE["source"], "restore_config": [{"target": ["ingest"]}]})


# -- data ---------------------------------------------------------------------------------------

@scenario("D1")
def _schema_drift(w: World) -> Effects:
    renamed = w.rng.choice(["total_amount", "order_amount", "amount_lkr"])
    fields = sorted(["created_at", "customer_id", "order_id", "status", "store_id", renamed])
    error = f"SchemaMismatch: source payload missing required field(s) ['amount']; got {fields}"

    def evidence(b, w, t):
        b.data.source_sample_fields = fields

    return Effects(variant=f"renamed_to_{renamed}", onset_s=w.rng.uniform(0, 10), ingest_error=error,
                   ingest_run_s=0.3, evidence=evidence,
                   fix_targets=lambda w: {"map_schema_fields": [{"mapping": [renamed]}]})


@scenario("D2")
def _bad_rows(w: World) -> Effects:
    r = w.rng
    nulls, dups = r.randint(1, 6), r.choice([0, 2, 2, 3])
    node = f"test.orders_platform.not_null_stg_orders_amount.{w.hex(10)}"
    error = f"{node} fail: Got {nulls} results, configured to fail if != 0" + (" (+1 more)" if dups else "")

    def evidence(b, w, t):
        b.data.last_dbt_failures.append(DbtFailure(node=node, status="fail",
                                                   message=f"Got {nulls} results, configured to fail if != 0"))
        if dups:
            b.data.last_dbt_failures.append(DbtFailure(
                node=f"test.orders_platform.unique_stg_orders_order_id.{w.hex(10)}", status="fail",
                message=f"Got {dups} results, configured to fail if != 0"))

    return Effects(variant="nulls+duplicates" if dups else "nulls_only", onset_s=r.uniform(0, 20),
                   transform_error=error, dbt="test_fail", dbt_params={"nulls": nulls, "dups": dups},
                   evidence=evidence)


@scenario("D3")
def _table_lock(w: World) -> Effects:
    r = w.rng
    blocks_dbt = r.random() < 0.85
    maint, ingest_pid = w.pid("maint"), w.pid("ingest-session")

    def evidence(b, w, t):
        b.data.locks += [LockInfo(pid=maint, relation="raw.orders", mode="AccessExclusiveLock", granted=True),
                         LockInfo(pid=ingest_pid, relation="raw.orders", mode="RowExclusiveLock", granted=False,
                                  blocked_by=[maint])]
        b.data.active_queries.append(ActiveQuery(
            pid=maint, usename="maint", application_name="adhoc-maintenance", client_addr=None, state="active",
            wait_event_type="Timeout", duration_s=round(t + r.uniform(1, 20), 1),
            query="BEGIN; LOCK TABLE raw.orders IN ACCESS EXCLUSIVE MODE; SELECT pg_sleep(86400);"))

    e = Effects(variant="blocks_ingest_and_dbt" if blocks_dbt else "blocks_ingest", onset_s=r.uniform(0, 10),
                ingest_error="LockNotAvailable: canceling statement due to lock timeout", ingest_run_s=3.1,
                evidence=evidence, fix_targets=lambda w: {"stop_session": [  # the lock holder, never the waiter
                    {"pid": [maint]}, {"application_name": ["adhoc-maintenance"]}, {"usename": ["maint"]}]})
    if blocks_dbt:
        e.transform_error, e.dbt = STG_LOCK_TIMEOUT, "lock_timeout"
    return e


@scenario("D4")
def _connection_leak(w: World) -> Effects:
    r = w.rng
    leaked = r.randint(33, 36)
    worker_ip = w.ips["reporting-worker"]

    def evidence(b, w, t):
        b.data.connections_by_app["reporting-worker"] = leaked
        for i in range(5):  # the tool returns the first rows of a long list
            b.data.active_queries.append(ActiveQuery(
                pid=w.pid(f"leak-{i}"), usename="app_reporting", application_name="reporting-worker",
                client_addr=worker_ip, state="idle", duration_s=round(t + r.uniform(5, 30), 1), query=""))
        b.security.db_client_origins.append(ClientOrigin(usename="app_reporting", application_name="reporting-worker",
                                                         client_addr=worker_ip, sessions=leaked))
        add_container(b, w, "reporting-worker", r.uniform(0, 1), r.uniform(40, 60))

    return Effects(
        onset_s=r.uniform(0, 5),
        ingest_error="OperationalError: " + pg_connect_error(w, True, SLOTS), ingest_recorded=False, ingest_run_s=0.05,
        transform_error=("model.orders_platform.stg_orders error: Database Error in model stg_orders "
                         f"(models/staging/stg_orders.sql)\n  {pg_connect_error(w, False, SLOTS)}"),
        dbt="slots", dbt_params={"conn": pg_connect_error(w, False, SLOTS)}, transform_recorded=False,
        transform_record_error="OperationalError: " + pg_connect_error(w, False, SLOTS),
        api_errors=(3, 5), api_error_log="revenue query failed: OperationalError: " + pg_connect_error(w, False, SLOTS),
        db_connections=(37, 39), evidence=evidence,
        fix_targets=lambda w: {"stop_session": [  # all leaked sessions: a few pids would not free the slots
            {"application_name": ["reporting-worker"]}, {"usename": ["app_reporting"]}, {"client_addr": [worker_ip]}]},
    )


@scenario("D5")
def _runaway_query(w: World) -> Effects:
    r = w.rng
    threads = r.choice([1, 2, 2, 3])
    notebook_ip = w.ips["analytics-notebook"]
    e = Effects(variant=f"{threads}_queries", onset_s=r.uniform(0, 10))
    _cpu_pressure(w, e)

    def evidence(b, w, t):
        top = b.infra.top_processes["warehouse"]
        for i in range(threads):
            pid = w.pid(f"notebook-{i}")
            b.data.active_queries.append(ActiveQuery(
                pid=pid, usename="analyst", application_name="jupyter-notebook", client_addr=notebook_ip,
                state="active", duration_s=round(r.uniform(15, 240), 1), query=NOTEBOOK_QUERY))
            top.insert(0, ProcessInfo(pid=pid, cpu_pct=round(r.uniform(90, 99) / threads, 1),
                                      command=f"postgres: analyst warehouse {notebook_ip}({r.randint(40000, 60000)}) SELECT"))
        container(b, "warehouse").cpu_pct = round(r.uniform(92, 100), 1)
        add_container(b, w, "analytics-notebook", r.uniform(0, 2), r.uniform(50, 80))
        if r.random() < 0.5:
            _dbt_query_red_herring(w, b, t)

    e.evidence = evidence
    e.fix_targets = lambda w: {"stop_session": [  # the notebook queries, not dbt's own build query
        {"pid": [w.pid(f"notebook-{i}") for i in range(threads)]}, {"application_name": ["jupyter-notebook"]},
        {"usename": ["analyst"]}, {"client_addr": [notebook_ip]}]}
    return e


# -- infrastructure -----------------------------------------------------------------------------

@scenario("I1")
def _cpu_hog(w: World) -> Effects:
    r = w.rng
    procs = r.randint(4, 12)
    e = Effects(variant=f"{procs}_processes", onset_s=r.uniform(0, 10))
    _cpu_pressure(w, e)

    def evidence(b, w, t):
        top = b.infra.top_processes["warehouse"]
        command = f"sh /usr/local/lib/maint/backup-compress.sh {procs}"
        burners = [ProcessInfo(pid=w.pid(f"burn-{i}"), cpu_pct=round(r.uniform(88, 97) / procs, 1), command=command)
                   for i in range(procs)]
        b.infra.top_processes["warehouse"] = burners + [ProcessInfo(pid=w.pid("burn-parent"), cpu_pct=0.0,
                                                                     command=command)] + top
        container(b, "warehouse").cpu_pct = round(r.uniform(97, 100), 1)
        if r.random() < 0.5:
            _dbt_query_red_herring(w, b, t)

    e.evidence = evidence
    e.fix_targets = lambda w: {"stop_process": [
        {"target": ["warehouse"], "pid": [w.pid(f"burn-{i}") for i in range(procs)] + [w.pid("burn-parent")]}]}
    return e


@scenario("I2")
def _transform_oom(w: World) -> Effects:
    r = w.rng
    limit = r.choice([48, 64, 80, 96])
    logged = r.random() < 0.6

    def evidence(b, w, t):
        c = container(b, "transform")
        c.mem_limit_mb, c.mem_mb, c.oom_kills = limit, round(limit * r.uniform(0.9, 0.99), 1), r.randint(1, 6)
        if logged:
            b.infra.recent_changes.append(ChangeEvent(at=w.iso(-r.uniform(5, 300)), component="transform",
                                                      change=f"memory limit changed from 512 MB to {limit} MB"))

    return Effects(variant="change_logged" if logged else "change_not_logged", onset_s=r.uniform(0, 20),
                   transform_error=OOM, dbt="oom", evidence=evidence,
                   fix_targets=lambda w: {"resize_compute": [{"target": ["transform"]}]})


@scenario("I3")
def _spool_full(w: World) -> Effects:
    r = w.rng
    filename = r.choice(["stale-backfill.tmp", "batch-backfill-2026.json.part", "core.dump"])

    def evidence(b, w, t):
        b.infra.disks[0] = DiskUsage(service="ingest", mount="/spool", used_pct=100.0,
                                     largest_files=[f"{filename} ({r.uniform(7.6, 8.0):.1f} MB)"])

    return Effects(variant=filename, onset_s=r.uniform(0, 10),
                   ingest_error="OSError: [Errno 28] No space left on device", ingest_run_s=0.1, evidence=evidence,
                   fix_targets=lambda w: {"free_storage": [{"target": ["ingest"]}]})


@scenario("I4")
def _bad_config(w: World) -> Effects:
    r = w.rng
    field_name, problem = r.choice([
        ("cache_ttl_s", "cache_ttl_s: Input should be a valid number, unable to parse string as a number"),
        ("warehouse.port", "warehouse.port: Input should be a valid integer, unable to parse string as an integer"),
        ("users", "users: Field required"),
    ])
    logged = r.random() < 0.7
    restart_every = r.uniform(4, 15)

    def lines(w, t):
        if (t % restart_every) < 5:  # one crash per restart, as Docker backs off
            return [("CRITICAL", f"invalid configuration in metrics_api.json, refusing to start: {problem}")]
        return []

    def evidence(b, w, t):
        c = container(b, "metrics-api")
        c.status, c.cpu_pct, c.restarts = "restarting", 0.0, r.randint(3, 20)
        if logged:
            b.infra.recent_changes.append(ChangeEvent(at=w.iso(-r.uniform(5, 120)), component="metrics-api",
                                                      change=f"config updated: {field_name}"))

    return Effects(variant=f"{field_name}+{'change_logged' if logged else 'change_not_logged'}",
                   onset_s=r.uniform(0, 5), api_down="ConnectError: [Errno 111] Connection refused",
                   metrics_api_lines=lines, evidence=evidence,
                   fix_targets=lambda w: {"restore_config": [{"target": ["metrics-api"]}]})


# -- security -----------------------------------------------------------------------------------

@scenario("S1")
def _credential_misuse(w: World) -> Effects:
    r = w.rng
    heavy = r.random() < 0.3
    threads = r.randint(25, 33) if heavy else r.randint(3, 10)
    client_ip = w.ips["bi-sync-7f3a"]
    e = Effects(variant="heavy" if heavy else "light", onset_s=r.uniform(0, 10), unexpected_clients={"psql": threads})
    if threads >= 6:
        e.api_p95 = (r.uniform(1.02, 1.3), r.uniform(1.4, 2.2))
    if heavy:
        e.db_connections = (36, 39)
        e.ingest_error, e.ingest_recorded, e.ingest_run_s = "OperationalError: " + pg_connect_error(w, True, SLOTS), False, 0.05
        e.api_errors, e.api_error_log = (1, 4), "revenue query failed: OperationalError: " + pg_connect_error(w, False, SLOTS)

    def evidence(b, w, t):
        b.security.db_client_origins.append(ClientOrigin(usename="etl", application_name="psql",
                                                         client_addr=client_ip, sessions=threads))
        for i in range(min(threads, 6)):
            b.data.active_queries.append(ActiveQuery(
                pid=w.pid(f"copy-{i}"), usename="etl", application_name="psql", client_addr=client_ip,
                state="active", duration_s=round(r.uniform(0.2, 4), 1), query="COPY (SELECT * FROM raw.orders) TO STDOUT"))
        b.data.connections_by_app["psql"] = threads
        container(b, "warehouse").cpu_pct = round(min(100.0, 30 + threads * r.uniform(4, 7)), 1)
        add_container(b, w, "bi-sync-7f3a", r.uniform(5, 30), r.uniform(40, 90))

    e.evidence = evidence
    e.fix_targets = lambda w: {"rotate_credential": [{"role": ["etl"]}], "stop_session": [
        {"usename": ["etl"]}, {"application_name": ["psql"]}, {"client_addr": [client_ip]}]}
    return e


@scenario("S2")
def _login_brute_force(w: World) -> Effects:
    r = w.rng
    unresponsive = r.random() < 0.3
    attempts = r.randint(80, 400)
    client_ip = w.ips["web-client-2c91"]

    def lines(w, t):
        return [("WARNING", f"failed login user={r.choice(['admin', 'analyst', 'root', 'ops', 'finance'])} "
                            f"client={client_ip}") for _ in range(r.randint(3, 8))]

    def evidence(b, w, t):
        b.security.failed_logins_by_ip[client_ip] = attempts
        c = container(b, "metrics-api")
        c.cpu_pct = round(r.uniform(47, 50), 1)  # its limit is half a core
        b.infra.top_processes["metrics-api"] = [ProcessInfo(pid=1, cpu_pct=c.cpu_pct,
                                                            command="/usr/local/bin/python /usr/local/bin/uvicorn app:app")]
        add_container(b, w, "web-client-2c91", r.uniform(10, 40), r.uniform(30, 60))

    e = Effects(variant="unresponsive" if unresponsive else "slow", onset_s=r.uniform(0, 5),
                metrics_api_lines=lines, evidence=evidence,
                fix_targets=lambda w: {"block_client": [{"ip": [client_ip]}]})
    if unresponsive:  # health checks time out; login stats cannot be collected either
        e.api_down = "ReadTimeout: timed out"
    else:
        e.api_p95 = (r.uniform(1.2, 2.0), r.uniform(2.2, 3.5))
        e.login_failures = (int(attempts * 0.8), int(attempts * 1.2))
    return e
