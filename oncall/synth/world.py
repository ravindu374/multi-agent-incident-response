"""A randomised environment for one synthetic incident: wall clock, addresses, process ids."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

BASE = datetime(2026, 11, 1, tzinfo=timezone.utc)
SERVICES = ("warehouse", "orders-api", "toxiproxy", "ingest", "transform", "metrics-api",
            "postgres-exporter", "prometheus")
WORKERS = ("reporting-worker", "analytics-notebook", "bi-sync-7f3a", "web-client-2c91")
GATEWAY = "172.18.0.1"  # traffic from the host (people, the oracle) arrives from the bridge gateway


@dataclass
class World:
    rng: random.Random
    t0: datetime  # wall-clock time of t = 0 (fault onset reference)
    ips: dict[str, str]
    _pids: dict[str, int] = field(default_factory=dict)

    def at(self, t: float) -> datetime:
        return self.t0 + timedelta(seconds=t)

    def stamp(self, t: float) -> str:  # Python logging's default asctime
        dt = self.at(t)
        return dt.strftime("%Y-%m-%d %H:%M:%S,") + f"{dt.microsecond // 1000:03d}"

    def iso(self, t: float) -> str:
        return self.at(t).isoformat(timespec="seconds")

    def clock(self, t: float) -> str:  # dbt's console timestamps
        return self.at(t).strftime("%H:%M:%S")

    def pg_time(self, t: float) -> str:  # str() of a timestamptz from psycopg
        return str(self.at(t))

    def hex(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789abcdef") for _ in range(n))

    def pid(self, name: str) -> int:
        if name not in self._pids:
            used = set(self._pids.values())
            pid = self.rng.randint(60, 9000)
            while pid in used:
                pid = self.rng.randint(60, 9000)
            self._pids[name] = pid
        return self._pids[name]


def make_world(rng: random.Random) -> World:
    t0 = BASE + timedelta(days=rng.randint(0, 27), seconds=rng.uniform(0, 86_400))
    hosts = rng.sample(range(2, 60), len(SERVICES) + len(WORKERS))
    ips = {name: f"172.18.0.{host}" for name, host in zip(SERVICES + WORKERS, hosts)}
    return World(rng=rng, t0=t0, ips=ips)
