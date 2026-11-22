"""Thin wrapper over the docker / docker compose CLI: the lab's only way to touch containers."""
from __future__ import annotations

import subprocess
from collections.abc import Sequence

from .settings import Settings


class DockerError(RuntimeError):
    pass


class Compose:
    def __init__(self, settings: Settings):
        self._base = ["docker", "compose", "-f", str(settings.compose_file), "-p", settings.project]

    @staticmethod
    def _run(cmd: Sequence[str], check: bool, timeout: float) -> str:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip()
            raise DockerError(f"`{' '.join(cmd)}` failed ({proc.returncode}): {message}")
        return proc.stdout

    def compose(self, *args: str, check: bool = True, timeout: float = 120) -> str:
        return self._run([*self._base, *args], check, timeout)

    def docker(self, *args: str, check: bool = True, timeout: float = 60) -> str:
        return self._run(["docker", *args], check, timeout)

    def exec(self, service: str, *cmd: str, detach: bool = False,
             env: dict[str, str] | None = None, check: bool = True) -> str:
        args = ["exec", "-T"]
        if detach:
            args.append("-d")
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        return self.compose(*args, service, *cmd, check=check)

    def logs(self, service: str, tail: int = 30) -> str:
        return self.compose("logs", "--no-color", "--no-log-prefix", "--tail", str(tail), service, check=False)

    def container_id(self, service: str) -> str:
        cid = self.compose("ps", "-q", service).strip()
        if not cid:
            raise DockerError(f"service {service!r} is not running")
        return cid.splitlines()[0]

    def set_memory_limit(self, service: str, memory_mb: int) -> None:
        cid = self.container_id(service)
        self.docker("update", "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m", cid)

    def restart(self, service: str) -> None:
        self.compose("restart", service, timeout=180)

    def start_worker(self, name: str, *cmd: str) -> None:
        """Run a toolbox script as a named, detached container on the sandbox network."""
        self.remove_container(name)
        self.compose("--profile", "lab", "run", "-d", "--no-deps", "--name", name, "toolbox", *cmd)

    def remove_container(self, name: str) -> None:
        self.docker("rm", "-f", name, check=False)

    def container_ip(self, name: str) -> str:
        out = self.docker("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", name)
        ips = out.split()
        if not ips:
            raise DockerError(f"container {name!r} has no network address")
        return ips[0]
