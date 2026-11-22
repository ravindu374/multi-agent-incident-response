"""Build the Block 1 sample set: one hand-written incident per fault, in lab/incidents.jsonl format.

These imitate what the sandbox prints (our services' log format, Postgres/psycopg, httpx and
dbt messages). They exist to check intake (Block 1) only. Do NOT train or evaluate the
classifier on them: they were written from the labels, which is exactly the circularity the
project avoids. Real incidents come from `python -m oncall validate`.

    python samples/build_samples.py   ->  samples/incidents.jsonl + samples/expected_intake.json
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)
DB = 'connection failed: connection to server at "172.18.0.5", port 15432 failed:'
DB_DIRECT = 'connection failed: connection to server at "172.18.0.4", port 5432 failed:'
SLOTS = "FATAL:  remaining connection slots are reserved for roles with the SUPERUSER attribute"


def stamp(sec: int) -> str:
    return (BASE + timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S,") + f"{(sec * 137) % 1000:03d}"


def iso(sec: int) -> str:
    return (BASE + timedelta(seconds=sec)).isoformat(timespec="seconds")


def lines(service: str, *entries) -> str:
    """entries: (second, level, message) tuples, or raw strings (e.g. dbt's own output)."""
    out = []
    for entry in entries:
        out.append(entry if isinstance(entry, str) else f"{stamp(entry[0])} {service} {entry[1]} {entry[2]}")
    return "\n".join(out) + "\n"


def every(start: int, step: int, n: int, level: str, message: str) -> list[tuple]:
    return [(start + i * step, level, message) for i in range(n)]


def dbt_failure(clock: str, failing: list[str], message_lines: list[str], skipped: int = 1) -> list[str]:
    out = [f"{clock}  Running with dbt=1.9.1", f"{clock}  Registered adapter: postgres=1.9.0",
           f"{clock}  Found 2 models, 4 data tests, 1 source, 433 macros"]
    out += [f"{clock}  {i} of 6 FAIL {node} ........................ [FAIL]" for i, node in enumerate(failing, 3)]
    out += [f"{clock}  {m}" for m in message_lines]
    out.append(f"{clock}  Done. PASS=3 WARN=0 ERROR={len(failing)} SKIP={skipped} TOTAL=6")
    return out


INGEST_OK = every(0, 15, 2, "INFO", "run ok: 75 rows (batch 41)")
TRANSFORM_OK = every(5, 30, 2, "INFO", "dbt build ok in 6.2s")
STMT_TIMEOUT = ("model.orders_platform.orders_mart error: Database Error in model orders_mart "
                "(models/marts/orders_mart.sql)\n  canceling statement due to statement timeout")

SAMPLES = [
    dict(
        imitates="N1", note="latency on the source path: ingest read timeouts, data goes stale",
        alerts=[("IngestJobFailing", "ingest failed 2 runs in a row; last error: ReadTimeout: timed out", 50),
                ("OrdersMartStale", "orders_mart is 151s behind the source (SLO 120s)", 140)],
        logs={"ingest": lines("ingest", *INGEST_OK, *every(30, 20, 4, "ERROR", "run failed: ReadTimeout: timed out")),
              "transform": lines("transform", *TRANSFORM_OK)},
        runs=[("ingest", "failed", 90, "ReadTimeout: timed out"), ("ingest", "failed", 70, "ReadTimeout: timed out"),
              ("transform", "success", 65, None)],
        must=["network_timeout", "stale_data"],
        must_not=["lock_timeout", "statement_timeout", "connection_reset", "name_resolution", "disk_full"],
        components=["ingest", "marts"],
    ),
    dict(
        imitates="N2", note="resets on the warehouse path: ingest cannot even record its failures",
        alerts=[("IngestJobFailing", "no successful ingest run for 75s (expected every <60s)", 75),
                ("OrdersMartStale", "orders_mart is 133s behind the source (SLO 120s)", 135)],
        logs={"ingest": lines("ingest", *INGEST_OK,
                              *every(30, 15, 3, "ERROR", f"run failed: OperationalError: {DB} server closed the "
                                                         "connection unexpectedly"),
                              *every(31, 15, 3, "WARNING", "could not record failed run in ops.job_runs: "
                                                           f"OperationalError: {DB} server closed the connection "
                                                           "unexpectedly"))},
        runs=[("transform", "success", 65, None), ("ingest", "success", 15, None)],
        must=["connection_reset", "connect_failed", "stale_data"],
        must_not=["network_timeout", "name_resolution", "too_many_clients"],
        components=["ingest", "marts"],
    ),
    dict(
        imitates="N3", note="source hostname does not resolve",
        alerts=[("IngestJobFailing", "ingest failed 2 runs in a row; last error: ConnectError: [Errno -2] "
                                     "Name or service not known", 45),
                ("OrdersMartStale", "orders_mart is 128s behind the source (SLO 120s)", 130)],
        logs={"ingest": lines("ingest", *INGEST_OK,
                              *every(30, 15, 4, "ERROR", "run failed: ConnectError: [Errno -2] Name or service "
                                                         "not known"))},
        runs=[("ingest", "failed", 75, "ConnectError: [Errno -2] Name or service not known")],
        must=["name_resolution", "connect_failed", "stale_data"],
        must_not=["network_timeout", "connection_reset"],
        components=["ingest", "marts"],
    ),
    dict(
        imitates="D1", note="schema drift in the source payload",
        alerts=[("IngestJobFailing", "ingest failed 2 runs in a row; last error: SchemaMismatch: source payload "
                                     "missing required field(s) ['amount']; got ['created_at', 'customer_id', "
                                     "'order_id', 'status', 'store_id', 'total_amount']", 45),
                ("OrdersMartStale", "orders_mart is 140s behind the source (SLO 120s)", 140)],
        logs={"ingest": lines("ingest", *INGEST_OK,
                              *every(30, 15, 3, "ERROR", "run failed: SchemaMismatch: source payload missing "
                                                         "required field(s) ['amount']; got ['created_at', "
                                                         "'customer_id', 'order_id', 'status', 'store_id', "
                                                         "'total_amount']"))},
        runs=[("ingest", "failed", 60, "SchemaMismatch: source payload missing required field(s) ['amount']")],
        must=["schema_mismatch", "stale_data"],
        must_not=["network_timeout", "data_test_failed", "connect_failed"],
        components=["ingest", "marts"],
    ),
    dict(
        imitates="D2", note="bad batch: dbt not_null and unique tests fail, mart is not rebuilt",
        alerts=[("TransformJobFailing", "transform failed 2 runs in a row; last error: "
                                        "test.orders_platform.not_null_stg_orders_amount.4b6a2c1f3e fail: Got 3 "
                                        "results, configured to fail if != 0 (+1 more)", 70),
                ("OrdersMartStale", "orders_mart is 126s behind the source (SLO 120s)", 130)],
        logs={"transform": lines("transform", *TRANSFORM_OK,
                                 *dbt_failure("10:01:05", ["not_null_stg_orders_amount", "unique_stg_orders_order_id"],
                                              ["Failure in test not_null_stg_orders_amount "
                                               "(models/staging/_stg_models.yml)",
                                               "  Got 3 results, configured to fail if != 0"]),
                                 (66, "ERROR", "dbt build failed in 7.4s: test.orders_platform."
                                               "not_null_stg_orders_amount.4b6a2c1f3e fail: Got 3 results, "
                                               "configured to fail if != 0 (+1 more)"))},
        runs=[("transform", "failed", 66, "test.orders_platform.not_null_stg_orders_amount.4b6a2c1f3e fail: Got 3 "
                                          "results, configured to fail if != 0 (+1 more)")],
        must=["data_test_failed", "stale_data"],
        must_not=["oom_killed", "schema_mismatch", "statement_timeout"],
        components=["marts", "transform"],
    ),
    dict(
        imitates="D3", note="exclusive table lock: ingest and dbt both hit lock timeouts",
        alerts=[("IngestJobFailing", "ingest failed 2 runs in a row; last error: LockNotAvailable: canceling "
                                     "statement due to lock timeout", 40),
                ("TransformJobFailing", "transform failed 2 runs in a row; last error: model.orders_platform."
                                        "stg_orders error: Database Error in model stg_orders "
                                        "(models/staging/stg_orders.sql)\n  canceling statement due to lock "
                                        "timeout", 70),
                ("OrdersMartStale", "orders_mart is 139s behind the source (SLO 120s)", 140)],
        logs={"ingest": lines("ingest", *every(20, 18, 3, "ERROR", "run failed: LockNotAvailable: canceling "
                                                                    "statement due to lock timeout")),
              "transform": lines("transform", (64, "ERROR", "dbt build failed in 8.9s: model.orders_platform."
                                                            "stg_orders error: Database Error in model stg_orders "
                                                            "(models/staging/stg_orders.sql)"),
                                 "10:01:04    canceling statement due to lock timeout")},
        runs=[("ingest", "failed", 56, "LockNotAvailable: canceling statement due to lock timeout"),
              ("transform", "failed", 64, "model.orders_platform.stg_orders error: Database Error in model "
                                          "stg_orders (models/staging/stg_orders.sql)")],
        must=["lock_timeout", "stale_data"],
        must_not=["network_timeout", "statement_timeout", "too_many_clients"],
        components=["ingest", "marts", "transform"],
    ),
    dict(
        imitates="D4", note="leaked connections exhaust the warehouse: everyone fails to connect",
        alerts=[("DashboardSlow", "5/5 dashboard revenue requests failed; last error: HTTP 503", 10),
                ("WarehouseConnectionsSaturated", "warehouse connections at 38/40 (95%)", 10),
                ("IngestJobFailing", "no successful ingest run for 70s (expected every <60s)", 70),
                ("OrdersMartStale", "orders_mart is 131s behind the source (SLO 120s)", 130)],
        logs={"ingest": lines("ingest", *every(15, 15, 3, "ERROR", f"run failed: OperationalError: {DB} {SLOTS}")),
              "metrics-api": lines("metrics-api", *every(10, 5, 5, "ERROR", "revenue query failed: OperationalError: "
                                                                            f"{DB_DIRECT} {SLOTS}")),
              "transform": lines("transform", (35, "WARNING", "could not record run in ops.job_runs: "
                                                              f"OperationalError: {DB_DIRECT} {SLOTS}"))},
        runs=[("transform", "success", 5, None), ("ingest", "success", 0, None)],
        must=["too_many_clients", "db_saturation", "http_5xx", "connect_failed", "stale_data"],
        must_not=["lock_timeout", "login_failures", "unexpected_db_client"],
        components=["ingest", "marts", "metrics-api", "transform", "warehouse"],
    ),
    dict(
        imitates="D5", note="runaway analyst query; on purpose almost identical to I1 at this level",
        alerts=[("DashboardSlow", "dashboard p95 latency 1.84s over 5 probes (SLO 1.0s)", 20),
                ("TransformJobFailing", f"transform failed 2 runs in a row; last error: {STMT_TIMEOUT}", 75)],
        logs={"ingest": lines("ingest", *INGEST_OK),
              "transform": lines("transform", (45, "ERROR", "dbt build failed in 16.3s: model.orders_platform."
                                                            "orders_mart error: Database Error in model orders_mart "
                                                            "(models/marts/orders_mart.sql)"),
                                 "10:00:45    canceling statement due to statement timeout")},
        runs=[("transform", "failed", 75, STMT_TIMEOUT), ("transform", "failed", 45, STMT_TIMEOUT)],
        must=["high_latency", "statement_timeout"],
        must_not=["stale_data", "lock_timeout", "unexpected_db_client"],
        components=["metrics-api", "transform"],
    ),
    dict(
        imitates="I1", note="CPU hog in the warehouse container; same signals as D5",
        alerts=[("DashboardSlow", "dashboard p95 latency 2.12s over 5 probes (SLO 1.0s)", 15),
                ("TransformJobFailing", f"transform failed 2 runs in a row; last error: {STMT_TIMEOUT}", 80)],
        logs={"ingest": lines("ingest", *INGEST_OK),
              "transform": lines("transform", (50, "ERROR", "dbt build failed in 15.9s: model.orders_platform."
                                                            "orders_mart error: Database Error in model orders_mart "
                                                            "(models/marts/orders_mart.sql)"),
                                 "10:00:50    canceling statement due to statement timeout")},
        runs=[("transform", "failed", 80, STMT_TIMEOUT), ("transform", "failed", 50, STMT_TIMEOUT)],
        must=["high_latency", "statement_timeout"],
        must_not=["stale_data", "lock_timeout", "unexpected_db_client"],
        components=["metrics-api", "transform"],
    ),
    dict(
        imitates="I2", note="dbt killed by the out-of-memory killer",
        alerts=[("TransformJobFailing", "transform failed 2 runs in a row; last error: dbt build was killed "
                                        "(SIGKILL, exit 137); the container may be out of memory", 70),
                ("OrdersMartStale", "orders_mart is 137s behind the source (SLO 120s)", 140)],
        logs={"transform": lines("transform", *TRANSFORM_OK,
                                 *every(66, 33, 2, "ERROR", "dbt build failed in 2.7s: dbt build was killed "
                                                            "(SIGKILL, exit 137); the container may be out of "
                                                            "memory"))},
        runs=[("transform", "failed", 99, "dbt build was killed (SIGKILL, exit 137); the container may be out of "
                                          "memory")],
        must=["oom_killed", "stale_data"],
        must_not=["data_test_failed", "statement_timeout", "disk_full"],
        components=["marts", "transform"],
    ),
    dict(
        imitates="I3", note="spool disk full on the ingest worker",
        alerts=[("IngestJobFailing", "ingest failed 3 runs in a row; last error: OSError: [Errno 28] No space "
                                     "left on device", 55),
                ("OrdersMartStale", "orders_mart is 129s behind the source (SLO 120s)", 130)],
        logs={"ingest": lines("ingest", *INGEST_OK,
                              *every(30, 15, 3, "ERROR", "run failed: OSError: [Errno 28] No space left on device"))},
        runs=[("ingest", "failed", 60, "OSError: [Errno 28] No space left on device")],
        must=["disk_full", "stale_data"],
        must_not=["oom_killed", "network_timeout", "schema_mismatch"],
        components=["ingest", "marts"],
    ),
    dict(
        imitates="I4", note="bad config deploy: dashboard API crash-loops",
        alerts=[("DashboardDown", "dashboard API down: ConnectError: [Errno 111] Connection refused", 10)],
        logs={"metrics-api": lines("metrics-api", *every(2, 4, 3, "CRITICAL",
                                                         "invalid configuration in metrics_api.json, refusing to "
                                                         "start: cache_ttl_s: Input should be a valid number, unable "
                                                         "to parse string as a number"))},
        runs=[("ingest", "success", 15, None), ("transform", "success", 5, None)],
        must=["service_down", "connect_failed", "config_invalid"],
        must_not=["stale_data", "login_failures", "high_latency"],
        components=["metrics-api"],
    ),
    dict(
        imitates="S1", note="ETL credential used by an unknown client; no error lines at all",
        alerts=[("UnexpectedWarehouseClient", "service role etl used by unexpected application(s): psql x6", 10),
                ("DashboardSlow", "dashboard p95 latency 1.31s over 5 probes (SLO 1.0s)", 40)],
        logs={"ingest": lines("ingest", *INGEST_OK), "transform": lines("transform", *TRANSFORM_OK)},
        runs=[("ingest", "success", 15, None), ("transform", "success", 35, None)],
        must=["unexpected_db_client", "high_latency"],
        must_not=["login_failures", "too_many_clients", "stale_data"],
        components=["metrics-api", "warehouse"],
    ),
    dict(
        imitates="S2", note="password guessing against the dashboard login",
        alerts=[("DashboardSlow", "dashboard p95 latency 2.40s over 5 probes (SLO 1.0s)", 15),
                ("LoginFailureSpike", "212 failed dashboard logins in the last 60s (limit 30)", 15)],
        logs={"metrics-api": lines("metrics-api",
                                   *every(1, 1, 3, "WARNING", "failed login user=admin client=172.18.0.12"),
                                   *every(4, 1, 2, "WARNING", "failed login user=root client=172.18.0.12"),
                                   (6, "WARNING", "failed login user=finance client=172.18.0.12"))},
        runs=[("ingest", "success", 15, None), ("transform", "success", 5, None)],
        must=["login_failures", "high_latency"],
        must_not=["unexpected_db_client", "too_many_clients"],
        components=["metrics-api"],
    ),
]


OWNERS = {"N": "network", "D": "data", "I": "infra", "S": "security"}


def build() -> tuple[list[dict], dict]:
    incidents, expected = [], {}
    for n, s in enumerate(SAMPLES, 1):
        incident_id = f"sample-{n:02d}"
        alerts = [{"name": name, "check": "", "detail": detail, "fired_at": iso(t)} for name, detail, t in s["alerts"]]
        incidents.append({
            "incident_id": incident_id,
            "opened_at": alerts[0]["fired_at"],
            "resolved_at": None,
            "title": alerts[0]["name"],
            "alerts": alerts,
            "evidence": {
                "logs": s["logs"],
                "recent_job_runs": [{"job": j, "status": st, "finished_at": iso(t), "error": e}
                                    for j, st, t, e in sorted(s["runs"], key=lambda r: -r[2])],
            },
        })
        expected[incident_id] = {
            "imitates": s["imitates"], "owner": OWNERS[s["imitates"][0]], "note": s["note"],
            "alerts": [a["name"] for a in alerts], "components": s["components"],
            "must_have_signals": s["must"], "must_not_have_signals": s["must_not"],
        }
    return incidents, expected


if __name__ == "__main__":
    incidents, expected = build()
    (HERE / "incidents.jsonl").write_text("".join(json.dumps(i) + "\n" for i in incidents), encoding="utf-8")
    (HERE / "expected_intake.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(incidents)} incidents to {HERE / 'incidents.jsonl'}")
