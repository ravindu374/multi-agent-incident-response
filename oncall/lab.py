"""The Lab context (everything the lab can touch) and the ground-truth ledger.

Information barrier: ./lab (ledger, ground truth) and the admin endpoints used here
(Toxiproxy API, orders-api /admin, warehouse superuser) are never exposed to agents.
"""
from __future__ import annotations

import json
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .compose import Compose
from .runtime_config import RuntimeConfig
from .settings import Settings
from .toxiproxy import Toxiproxy

LAB_APPLICATION_NAME = "oncall-lab"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class Ledger:
    """Append-only JSONL streams under ./lab, plus the set of currently active injections."""

    def __init__(self, lab_dir: Path):
        self.dir = Path(lab_dir)

    def append(self, stream: str, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / f"{stream}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read(self, stream: str) -> list[dict]:
        path = self.dir / f"{stream}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_active(self) -> dict[str, dict]:
        path = self.dir / "active.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_active(self, active: dict[str, dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "active.json").write_text(json.dumps(active, indent=2, default=str), encoding="utf-8")


class Lab:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.compose = Compose(self.settings)
        self.toxiproxy = Toxiproxy(self.settings.toxiproxy_url)
        self.runtime = RuntimeConfig(self.settings.runtime_dir)
        self.ledger = Ledger(self.settings.lab_dir)
        self._dashboard_token: str | None = None

    # -- warehouse (superuser; lab only) -------------------------------------
    @contextmanager
    def connect(self):
        import psycopg  # imported lazily so the rest of the lab works without it

        conn = psycopg.connect(
            host="127.0.0.1",
            port=self.settings.warehouse_port,
            dbname="warehouse",
            user="postgres",
            password=self.settings.warehouse_admin_password,
            autocommit=True,
            connect_timeout=5,
            application_name=LAB_APPLICATION_NAME,
        )
        try:
            yield conn
        finally:
            conn.close()

    def sql(self, query: Any, params: Any = None) -> list[tuple]:
        with self.connect() as conn:
            cur = conn.execute(query, params)
            return cur.fetchall() if cur.description else []

    def execute(self, query: Any, params: Any = None) -> int:
        with self.connect() as conn:
            return conn.execute(query, params).rowcount

    # -- source system fault switches -----------------------------------------
    def set_source_faults(self, **flags: Any) -> dict:
        resp = httpx.post(f"{self.settings.orders_api_url}/admin/faults", json=flags, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def clear_source_faults(self) -> None:
        httpx.delete(f"{self.settings.orders_api_url}/admin/faults", timeout=5).raise_for_status()

    # -- dashboard API (as the analyst user) ----------------------------------
    def dashboard_token(self, client: httpx.Client, refresh: bool = False) -> str:
        if self._dashboard_token is None or refresh:
            resp = client.post(
                f"{self.settings.metrics_api_url}/api/login",
                json={"username": self.settings.dashboard_user, "password": self.settings.dashboard_password},
            )
            resp.raise_for_status()
            self._dashboard_token = resp.json()["token"]
        return self._dashboard_token
