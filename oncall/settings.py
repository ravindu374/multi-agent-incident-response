"""Where the sandbox lives and the service-level objectives it is judged by."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    raw = os.getenv(name)
    return default if raw is None else type(default)(raw)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    project: str = "oncall"
    runtime_dir: Path = field(default=ROOT / "runtime")
    lab_dir: Path = field(default=ROOT / "lab")

    warehouse_port: int = 55432
    warehouse_admin_password: str = "admin_pw"
    orders_api_port: int = 18000
    metrics_api_port: int = 18001
    toxiproxy_port: int = 8474

    ingest_interval_s: float = 15.0
    transform_interval_s: float = 30.0

    # SLOs: breaching any of these opens an incident
    freshness_slo_s: float = 120.0
    api_latency_slo_s: float = 1.0
    login_failure_limit: int = 30  # failed logins per 60 s
    db_connection_limit: float = 0.85  # share of max_connections in use
    job_failure_streak: int = 2  # consecutive failed runs before a job alert

    dashboard_user: str = "analyst"
    dashboard_password: str = "analyst-pass"

    @classmethod
    def from_env(cls) -> Settings:
        base = cls()
        return cls(
            warehouse_port=_env("WAREHOUSE_PORT", base.warehouse_port),
            warehouse_admin_password=_env("WAREHOUSE_ADMIN_PASSWORD", base.warehouse_admin_password),
            orders_api_port=_env("ORDERS_API_PORT", base.orders_api_port),
            metrics_api_port=_env("METRICS_API_PORT", base.metrics_api_port),
            toxiproxy_port=_env("TOXIPROXY_PORT", base.toxiproxy_port),
            ingest_interval_s=_env("INGEST_INTERVAL_S", base.ingest_interval_s),
            transform_interval_s=_env("TRANSFORM_INTERVAL_S", base.transform_interval_s),
            freshness_slo_s=_env("FRESHNESS_SLO_S", base.freshness_slo_s),
        )

    @property
    def compose_file(self) -> Path:
        return self.root / "docker-compose.yml"

    @property
    def orders_api_url(self) -> str:
        return f"http://127.0.0.1:{self.orders_api_port}"

    @property
    def metrics_api_url(self) -> str:
        return f"http://127.0.0.1:{self.metrics_api_port}"

    @property
    def toxiproxy_url(self) -> str:
        return f"http://127.0.0.1:{self.toxiproxy_port}"

    @property
    def ingest_stale_after_s(self) -> float:
        return max(4 * self.ingest_interval_s, 60.0)

    @property
    def transform_stale_after_s(self) -> float:
        return 4 * self.transform_interval_s + 60.0
