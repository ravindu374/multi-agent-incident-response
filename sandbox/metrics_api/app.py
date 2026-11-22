"""Dashboard backend: analyst login + revenue figures from marts.orders_mart.

The full config is validated once at start-up (a bad deploy crash-loops the container).
`blocked_ips` is re-read whenever the config file changes, so an IP block applies without a restart.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
from collections import deque
from pathlib import Path

import psycopg
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, ValidationError

CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/config"))

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s metrics-api %(levelname)s %(message)s"
)
log = logging.getLogger("metrics-api")


class WarehouseConfig(BaseModel):
    host: str
    port: int = 5432
    dbname: str
    user: str
    password: str


class Config(BaseModel):
    warehouse: WarehouseConfig
    cache_ttl_s: float = Field(ge=0)
    users: dict[str, str]  # username -> "pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>"
    blocked_ips: list[str] = []


def _config_path() -> Path:
    live = CONFIG_DIR / "metrics_api.json"
    return live if live.exists() else CONFIG_DIR / "defaults" / "metrics_api.json"


def _load_config() -> Config:
    path = _config_path()
    try:
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        log.critical("invalid configuration in %s, refusing to start: %s", path.name, problems)
    except (OSError, ValueError) as exc:
        log.critical("cannot read configuration %s: %s", path, exc)
    sys.exit(1)


CONFIG = _load_config()
TOKEN = secrets.token_hex(16)
_DUMMY_HASH = "pbkdf2_sha256$200000$" + "00" * 16 + "$" + "00" * 32

REQUESTS = Counter("http_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("http_request_seconds", "HTTP request latency in seconds", ["path"])
LOGIN_FAILURES = Counter("login_failures_total", "Failed dashboard logins")
BLOCKED = Counter("blocked_requests_total", "Requests rejected by the IP blocklist")


class _Blocklist:
    def __init__(self, initial: list[str]):
        self._ips = set(initial)
        self._mtime: float | None = None
        self._lock = threading.Lock()

    def current(self) -> set[str]:
        path = CONFIG_DIR / "metrics_api.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self._ips
        if mtime != self._mtime:
            try:
                ips = set(json.loads(path.read_text(encoding="utf-8")).get("blocked_ips") or [])
            except (OSError, ValueError, AttributeError):
                return self._ips
            with self._lock:
                self._ips, self._mtime = ips, mtime
        return self._ips


blocklist = _Blocklist(CONFIG.blocked_ips)
_failures: deque[float] = deque()
_failures_lock = threading.Lock()


def _trim(now: float) -> None:
    while _failures and now - _failures[0] > 60:
        _failures.popleft()


def _record_failure() -> None:
    now = time.monotonic()
    with _failures_lock:
        _failures.append(now)
        _trim(now)


def _failures_last_60s() -> int:
    with _failures_lock:
        _trim(time.monotonic())
        return len(_failures)


def _verify(password: str, encoded: str) -> bool:
    _, iterations, salt, expected = encoded.split("$")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(digest.hex(), expected)


app = FastAPI(title="metrics-api")


@app.middleware("http")
async def guard_and_measure(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    path = request.url.path
    if client in blocklist.current():
        BLOCKED.inc()
        return JSONResponse({"detail": "client blocked"}, status_code=403)
    started = time.perf_counter()
    response = await call_next(request)
    LATENCY.labels(path).observe(time.perf_counter() - started)
    REQUESTS.labels(path, str(response.status_code)).inc()
    return response


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/internal/stats")
def stats() -> dict:
    return {"login_failures_60s": _failures_last_60s()}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginRequest, request: Request) -> dict:
    encoded = CONFIG.users.get(body.username)
    valid = _verify(body.password, encoded or _DUMMY_HASH) and encoded is not None
    if not valid:
        LOGIN_FAILURES.inc()
        _record_failure()
        client = request.client.host if request.client else "unknown"
        log.warning("failed login user=%s client=%s", body.username, client)
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {"token": TOKEN}


_cache: dict[int, tuple[float, dict]] = {}

REVENUE_SQL = """
    SELECT coalesce(sum(revenue), 0), coalesce(sum(order_count), 0), max(_source_max_created_at)
    FROM marts.orders_mart
    WHERE minute >= now() - make_interval(mins => %s)
"""


@app.get("/api/revenue")
def revenue(minutes: int = 15, authorization: str = Header(default="")) -> dict:
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="missing or invalid token")
    minutes = max(1, min(minutes, 1440))
    if CONFIG.cache_ttl_s > 0:
        hit = _cache.get(minutes)
        if hit and time.monotonic() - hit[0] < CONFIG.cache_ttl_s:
            return hit[1]

    wh = CONFIG.warehouse
    try:
        with psycopg.connect(
            host=wh.host, port=wh.port, dbname=wh.dbname, user=wh.user, password=wh.password,
            connect_timeout=3, application_name="metrics-api",
        ) as conn:
            total, orders, as_of = conn.execute(REVENUE_SQL, (minutes,)).fetchone()
    except psycopg.Error as exc:
        text = str(exc).strip()
        log.error("revenue query failed: %s: %s", type(exc).__name__, text.splitlines()[0] if text else "")
        raise HTTPException(status_code=503, detail="warehouse unavailable") from None

    payload = {
        "minutes": minutes,
        "revenue": float(total),
        "orders": int(orders),
        "data_as_of": as_of.isoformat() if as_of else None,
    }
    _cache[minutes] = (time.monotonic(), payload)
    return payload
