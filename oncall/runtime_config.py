"""Live service configuration in ./runtime (mounted read-only into containers at /config).

Faults and remediation actions edit these files from the host; ingest re-reads its file every
run, metrics-api reads its file at start-up (blocklist on change). Writes are atomic.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

SERVICES = ("ingest", "metrics_api")


class RuntimeConfig:
    def __init__(self, runtime_dir: Path):
        self.dir = Path(runtime_dir)

    def _check(self, service: str) -> None:
        if service not in SERVICES:
            raise ValueError(f"unknown service config {service!r}; expected one of {SERVICES}")

    def live_path(self, service: str) -> Path:
        self._check(service)
        return self.dir / f"{service}.json"

    def default_path(self, service: str) -> Path:
        self._check(service)
        return self.dir / "defaults" / f"{service}.json"

    def default(self, service: str) -> dict:
        return json.loads(self.default_path(service).read_text(encoding="utf-8"))

    def load(self, service: str) -> dict:
        path = self.live_path(service)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return self.default(service)

    def save(self, service: str, data: dict) -> None:
        path = self.live_path(service)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def update(self, service: str, mutate: Callable[[dict], None]) -> dict:
        data = self.load(service)
        mutate(data)
        self.save(service, data)
        return data

    def restore_default(self, service: str) -> None:
        self.save(service, self.default(service))

    def init(self, overwrite: bool = False) -> list[str]:
        written = []
        for service in SERVICES:
            if overwrite or not self.live_path(service).exists():
                self.restore_default(service)
                written.append(service)
        return written
