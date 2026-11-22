"""Ingest job: every INGEST_INTERVAL_S, pull new orders from the source API into raw.orders.

Each run: read the cursor -> fetch a batch -> spool it to local disk -> insert it and move
the cursor in one transaction -> record the run in ops.job_runs. Configuration is re-read
from /config/ingest.json on every run, so remediation actions take effect without a restart.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import httpx
import psycopg
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from psycopg.types.json import Jsonb

from schema import normalise

CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/config"))
SPOOL_DIR = Path(os.getenv("SPOOL_DIR", "/spool"))
INTERVAL_S = float(os.getenv("INGEST_INTERVAL_S", "15"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9101"))

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s ingest %(levelname)s %(message)s"
)
log = logging.getLogger("ingest")

RUNS = Counter("ingest_runs_total", "Ingest runs by outcome", ["status"])
ROWS = Counter("ingest_rows_total", "Rows loaded into raw.orders")
LAST_SUCCESS = Gauge("ingest_last_success_unixtime", "Unix time of the last successful run")
DURATION = Histogram("ingest_run_seconds", "Ingest run duration in seconds")

INSERT_SQL = (
    "INSERT INTO raw.orders (order_id, created_at, customer_id, store_id, amount, status, batch_id) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
)
CURSOR_SQL = "SELECT last_order_id FROM raw.ingest_state WHERE source = 'orders_api'"
ADVANCE_SQL = (
    "UPDATE raw.ingest_state SET last_order_id = GREATEST(last_order_id, %s), updated_at = now() "
    "WHERE source = 'orders_api'"
)
RECORD_SQL = (
    "INSERT INTO ops.job_runs (job, started_at, status, rows_processed, error, details) "
    "VALUES ('ingest', to_timestamp(%s), %s, %s, %s, %s)"
)


def load_config() -> dict:
    for path in (CONFIG_DIR / "ingest.json", CONFIG_DIR / "defaults" / "ingest.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no ingest config under {CONFIG_DIR}")


def connect(cfg: dict) -> psycopg.Connection:
    wh = cfg["warehouse"]
    host, port = wh["endpoints"][wh["active"]].rsplit(":", 1)
    return psycopg.connect(
        host=host,
        port=int(port),
        dbname=wh["dbname"],
        user=wh["user"],
        password=wh["password"],
        connect_timeout=5,
        autocommit=True,
        application_name="ingest",
    )


def describe(exc: BaseException) -> str:
    text = str(exc).strip()
    first_line = text.splitlines()[0] if text else ""
    return f"{type(exc).__name__}: {first_line}"[:500]


def record(conn: psycopg.Connection, started: float, status: str, rows: int, error: str | None, batch_id) -> None:
    conn.execute(RECORD_SQL, (started, status, rows, error, Jsonb({"batch_id": batch_id})))


def record_failure(cfg: dict | None, started: float, error: str, batch_id) -> None:
    if cfg is None:
        return
    try:
        with connect(cfg) as conn:
            record(conn, started, "failed", 0, error, batch_id)
    except Exception as exc:  # the warehouse itself may be the problem
        log.warning("could not record failed run in ops.job_runs: %s", describe(exc))


def run_once() -> None:
    started = time.time()
    cfg: dict | None = None
    spool_file: Path | None = None
    batch_id = None
    try:
        cfg = load_config()
        with connect(cfg) as conn:
            cursor_id = conn.execute(CURSOR_SQL).fetchone()[0]
            source = cfg["source"]
            base_url = source["endpoints"][source["active"]]
            resp = httpx.get(
                f"{base_url}/orders",
                params={"after_id": cursor_id, "limit": cfg.get("batch_limit", 2000)},
                timeout=cfg.get("source_timeout_s", 5),
            )
            resp.raise_for_status()
            rows = normalise(resp.json()["orders"], cfg.get("field_map", {}))

            if rows:
                batch_id = conn.execute("SELECT nextval('raw.batch_id_seq')").fetchone()[0]
                spool_file = SPOOL_DIR / f"batch-{batch_id}.json"
                spool_file.write_text(json.dumps(rows), encoding="utf-8")
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.executemany(
                            INSERT_SQL,
                            [
                                (r["order_id"], r["created_at"], r["customer_id"], r["store_id"],
                                 r["amount"], r["status"], batch_id)
                                for r in rows
                            ],
                        )
                        cur.execute(ADVANCE_SQL, (max(int(r["order_id"]) for r in rows),))
            record(conn, started, "success", len(rows), None, batch_id)

        RUNS.labels("success").inc()
        ROWS.inc(len(rows))
        LAST_SUCCESS.set(time.time())
        log.info("run ok: %d rows (batch %s)", len(rows), batch_id)
    except Exception as exc:
        error = describe(exc)
        RUNS.labels("failed").inc()
        log.error("run failed: %s", error)
        record_failure(cfg, started, error, batch_id)
    finally:
        DURATION.observe(time.time() - started)
        if spool_file is not None:
            try:
                spool_file.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> None:
    start_http_server(METRICS_PORT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log.info("ingest started, interval %.0fs", INTERVAL_S)
    while True:
        run_once()
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
