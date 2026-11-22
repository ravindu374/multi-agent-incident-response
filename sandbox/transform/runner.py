"""Transform job: run `dbt build` every TRANSFORM_INTERVAL_S and record the outcome in ops.job_runs.

`dbt build` runs models and tests in DAG order, so a failing staging test stops the mart
from being rebuilt and the dashboard goes stale, as in a real dbt deployment.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from prometheus_client import Counter, Gauge, start_http_server
from psycopg.types.json import Jsonb

INTERVAL_S = float(os.getenv("TRANSFORM_INTERVAL_S", "30"))
PROJECT_DIR = os.getenv("DBT_PROJECT_DIR", "/dbt")
TARGET_DIR = Path(os.getenv("DBT_TARGET_PATH", "/tmp/dbt-target"))
HOST = os.getenv("DBT_HOST", "warehouse")
PASSWORD = os.getenv("TRANSFORMER_PASSWORD", "transformer_pw")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9102"))

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s transform %(levelname)s %(message)s"
)
log = logging.getLogger("transform")

RUNS = Counter("transform_runs_total", "dbt build runs by outcome", ["status"])
LAST_SUCCESS = Gauge("transform_last_success_unixtime", "Unix time of the last successful dbt build")
LAST_DURATION = Gauge("transform_last_duration_seconds", "Duration of the last dbt build")

DBT_CMD = [
    "dbt", "build",
    "--project-dir", PROJECT_DIR,
    "--profiles-dir", PROJECT_DIR,
    "--target-path", str(TARGET_DIR),
    "--log-path", "/tmp/dbt-logs",
    "--no-use-colors",
]
RECORD_SQL = (
    "INSERT INTO ops.job_runs (job, started_at, status, rows_processed, error, details) "
    "VALUES ('transform', to_timestamp(%s), %s, NULL, %s, %s)"
)


def summarise(returncode: int, stdout: str) -> tuple[str, str | None, dict]:
    """Turn a dbt exit code + run_results.json into (status, error message, details)."""
    if returncode in (-9, 137):
        return "failed", "dbt build was killed (SIGKILL, exit 137); the container may be out of memory", {
            "returncode": returncode
        }
    results: list[dict] = []
    try:
        results = json.loads((TARGET_DIR / "run_results.json").read_text())["results"]
    except (OSError, ValueError, KeyError):
        pass
    failures = [
        {
            "node": r.get("unique_id"),
            "status": r.get("status"),
            "message": (r.get("message") or "")[:300],
            "failures": r.get("failures"),
        }
        for r in results
        if r.get("status") in ("error", "fail")
    ]
    if returncode == 0 and not failures:
        return "success", None, {"nodes": len(results)}
    if failures:
        first = failures[0]
        more = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
        return "failed", f"{first['node']} {first['status']}: {first['message']}{more}", {"failures": failures}
    tail = [line for line in stdout.strip().splitlines() if line.strip()][-3:]
    return "failed", f"dbt exited {returncode}: " + " | ".join(tail), {"returncode": returncode}


def record(started: float, status: str, error: str | None, details: dict) -> None:
    try:
        with psycopg.connect(
            host=HOST, dbname="warehouse", user="transformer", password=PASSWORD,
            connect_timeout=5, autocommit=True, application_name="transform-runner",
        ) as conn:
            conn.execute(RECORD_SQL, (started, status, error, Jsonb(details)))
    except Exception as exc:
        log.warning("could not record run in ops.job_runs: %s: %s", type(exc).__name__, str(exc).strip()[:200])


def run_once() -> None:
    (TARGET_DIR / "run_results.json").unlink(missing_ok=True)
    started = time.time()
    try:
        proc = subprocess.run(DBT_CMD, capture_output=True, text=True, timeout=600)
        status, error, details = summarise(proc.returncode, proc.stdout)
        if status == "failed":
            print(proc.stdout[-4000:], flush=True)  # full dbt output for whoever investigates
    except subprocess.TimeoutExpired:
        status, error, details = "failed", "dbt build timed out after 600s", {}
    except OSError as exc:
        status, error, details = "failed", f"could not start dbt: {exc}", {}

    duration = time.time() - started
    LAST_DURATION.set(duration)
    RUNS.labels(status).inc()
    if status == "success":
        LAST_SUCCESS.set(time.time())
        log.info("dbt build ok in %.1fs", duration)
    else:
        log.error("dbt build failed in %.1fs: %s", duration, error)
    record(started, status, error, details)


def main() -> None:
    start_http_server(METRICS_PORT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log.info("transform started, interval %.0fs", INTERVAL_S)
    while True:
        run_once()
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
