"""Allowlisted remediation actions.

Agents will only ever *propose* one of these by name with typed parameters; the validator
checks risk and permissions, and the executor runs it. In Phase 1 the reference fixes use
the same functions, so every fault is proven fixable through the allowlist.
"""
from __future__ import annotations

import ipaddress
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .lab import LAB_APPLICATION_NAME, Lab, utcnow_iso
from .runtime_config import SERVICES

DOMAINS = ("network", "data", "infra", "security")
Risk = Literal["low", "medium", "high"]

REQUIRED_ORDER_FIELDS = ("order_id", "created_at", "customer_id", "store_id", "amount", "status")
RESTARTABLE_SERVICES = ("ingest", "transform", "metrics-api", "orders-api", "toxiproxy")
MEMORY_LIMITED_SERVICES = ("ingest", "transform", "metrics-api")
MANAGED_DB_ROLES = ("etl",)


class ActionError(ValueError):
    """The action was refused (bad parameters or outside its guard rails)."""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    risk: Risk
    domains: tuple[str, ...]  # specialist domains allowed to propose it
    reversible: bool
    summary: str
    fn: Callable[..., dict]


ACTIONS: dict[str, ActionSpec] = {}


def action(name: str, *, risk: Risk, domains: tuple[str, ...], reversible: bool, summary: str):
    def register(fn: Callable[..., dict]) -> Callable[..., dict]:
        ACTIONS[name] = ActionSpec(name, risk, domains, reversible, summary, fn)
        return fn

    return register


def execute(lab: Lab, name: str, params: dict[str, Any], *, actor: str) -> dict:
    spec = ACTIONS.get(name)
    if spec is None:
        raise ActionError(f"unknown action {name!r}")
    result = spec.fn(lab, **params)
    lab.ledger.append(
        "actions",
        {"at": utcnow_iso(), "actor": actor, "action": name, "risk": spec.risk, "params": params, "result": result},
    )
    return result


# -- network ------------------------------------------------------------------

def _switch_endpoint(lab: Lab, section: str, endpoint: str) -> dict:
    if endpoint not in ("primary", "secondary"):
        raise ActionError("endpoint must be 'primary' or 'secondary'")
    lab.runtime.update("ingest", lambda cfg: cfg[section].__setitem__("active", endpoint))
    return {f"ingest.{section}.active": endpoint}


@action("switch_source_endpoint", risk="low", domains=("network",), reversible=True,
        summary="Route ingest's calls to the orders API through the primary or secondary network path.")
def switch_source_endpoint(lab: Lab, endpoint: str) -> dict:
    return _switch_endpoint(lab, "source", endpoint)


@action("switch_warehouse_endpoint", risk="low", domains=("network",), reversible=True,
        summary="Route ingest's warehouse connections through the primary or secondary network path.")
def switch_warehouse_endpoint(lab: Lab, endpoint: str) -> dict:
    return _switch_endpoint(lab, "warehouse", endpoint)


# -- configuration ------------------------------------------------------------

@action("rollback_service_config", risk="medium", domains=("infra", "network", "data"), reversible=True,
        summary="Restore a service's configuration to the last known-good (deployed) version; "
                "restarts services that only read configuration at start-up.")
def rollback_service_config(lab: Lab, service: str) -> dict:
    key = service.replace("-", "_")
    if key not in SERVICES:
        raise ActionError(f"no managed configuration for {service!r}")
    lab.runtime.restore_default(key)
    restarted = key == "metrics_api"
    if restarted:
        lab.compose.restart("metrics-api")
    return {"service": key, "restarted": restarted}


# -- data ---------------------------------------------------------------------

@action("set_field_mapping", risk="high", domains=("data",), reversible=True,
        summary="Map renamed source fields onto the warehouse schema (changes data semantics).")
def set_field_mapping(lab: Lab, mapping: dict[str, str]) -> dict:
    if not mapping:
        raise ActionError("mapping is empty")
    unknown = sorted(set(mapping.values()) - set(REQUIRED_ORDER_FIELDS))
    if unknown:
        raise ActionError(f"unknown target field(s) {unknown}")
    lab.runtime.update("ingest", lambda cfg: cfg.setdefault("field_map", {}).update(mapping))
    return {"field_map": dict(mapping)}


QUARANTINE_SQL = """
WITH bad AS (
    SELECT o.ctid FROM raw.orders o
    WHERE o.batch_id = %(batch_id)s
      AND (o.amount IS NULL
           OR EXISTS (SELECT 1 FROM raw.orders p
                      WHERE p.order_id = o.order_id AND p.batch_id < o.batch_id))
), moved AS (
    DELETE FROM raw.orders o USING bad WHERE o.ctid = bad.ctid RETURNING o.*
)
INSERT INTO raw.orders_quarantine
    (order_id, created_at, customer_id, store_id, amount, status, batch_id, loaded_at, reason)
SELECT order_id, created_at, customer_id, store_id, amount, status, batch_id, loaded_at, %(reason)s
FROM moved
"""

BAD_BATCHES_SQL = """
SELECT DISTINCT o.batch_id FROM raw.orders o
WHERE o.amount IS NULL
   OR EXISTS (SELECT 1 FROM raw.orders p WHERE p.order_id = o.order_id AND p.batch_id < o.batch_id)
ORDER BY 1
"""


@action("quarantine_rows", risk="low", domains=("data",), reversible=True,
        summary="Move the rows of one ingest batch that break data tests (null amount, re-sent order_id) "
                "into raw.orders_quarantine.")
def quarantine_rows(lab: Lab, batch_id: int) -> dict:
    moved = lab.execute(QUARANTINE_SQL, {"batch_id": int(batch_id), "reason": "failed data tests"})
    return {"batch_id": int(batch_id), "rows_quarantined": moved}


@action("terminate_backend", risk="medium", domains=("data", "security"), reversible=False,
        summary="Terminate one warehouse session (and its running query) by pid.")
def terminate_backend(lab: Lab, pid: int) -> dict:
    rows = lab.sql(
        "SELECT application_name FROM pg_stat_activity WHERE pid = %s AND backend_type = 'client backend'",
        (int(pid),),
    )
    if not rows:
        raise ActionError(f"no client session with pid {pid}")
    if rows[0][0] == LAB_APPLICATION_NAME:
        raise ActionError("refusing to terminate the lab's own session")
    terminated = lab.sql("SELECT pg_terminate_backend(%s)", (int(pid),))[0][0]
    return {"pid": int(pid), "terminated": bool(terminated)}


@action("terminate_sessions", risk="medium", domains=("data", "security"), reversible=False,
        summary="Terminate warehouse sessions matching an application name, role and/or client address.")
def terminate_sessions(lab: Lab, application_name: str | None = None, usename: str | None = None,
                       client_addr: str | None = None, exclude_application_name: str | None = None) -> dict:
    if not (application_name or usename or client_addr):
        raise ActionError("give at least one of application_name, usename, client_addr")
    where = ["backend_type = 'client backend'", "pid <> pg_backend_pid()", "application_name <> %(lab)s"]
    params: dict[str, Any] = {"lab": LAB_APPLICATION_NAME}
    if application_name:
        where.append("application_name = %(app)s")
        params["app"] = application_name
    if usename:
        where.append("usename = %(user)s")
        params["user"] = usename
    if client_addr:
        where.append("client_addr = %(addr)s::inet")
        params["addr"] = client_addr
    if exclude_application_name:
        where.append("application_name <> %(exclude)s")
        params["exclude"] = exclude_application_name
    rows = lab.sql(
        "SELECT pid, pg_terminate_backend(pid) FROM pg_stat_activity WHERE " + " AND ".join(where), params
    )
    return {"terminated": [pid for pid, ok in rows if ok]}


# -- infrastructure -----------------------------------------------------------

@action("kill_container_process", risk="medium", domains=("infra",), reversible=False,
        summary="Kill processes (container-namespace pids) inside a service container.")
def kill_container_process(lab: Lab, service: str, pids: list[int]) -> dict:
    pids = [int(p) for p in pids]
    if not pids or any(p <= 1 for p in pids):
        raise ActionError("pids must be a non-empty list of values > 1")
    lab.compose.exec(service, "kill", "-9", *map(str, pids), check=False)
    return {"service": service, "killed": pids}


@action("set_memory_limit", risk="medium", domains=("infra",), reversible=True,
        summary="Change a service container's memory limit (64-2048 MB).")
def set_memory_limit(lab: Lab, service: str, memory_mb: int) -> dict:
    if service not in MEMORY_LIMITED_SERVICES:
        raise ActionError(f"memory limit of {service!r} is not managed")
    if not 64 <= int(memory_mb) <= 2048:
        raise ActionError("memory_mb must be between 64 and 2048")
    lab.compose.set_memory_limit(service, int(memory_mb))
    return {"service": service, "memory_mb": int(memory_mb)}


@action("purge_spool", risk="medium", domains=("infra",), reversible=False,
        summary="Delete every file in the ingest worker's local spool directory.")
def purge_spool(lab: Lab) -> dict:
    out = lab.compose.exec("ingest", "sh", "-c", "du -sh /spool | cut -f1; find /spool -mindepth 1 -delete")
    return {"freed": out.strip()}


@action("restart_service", risk="medium", domains=("infra",), reversible=True,
        summary="Restart one pipeline service container.")
def restart_service(lab: Lab, service: str) -> dict:
    if service not in RESTARTABLE_SERVICES:
        raise ActionError(f"restarting {service!r} is not allowed")
    lab.compose.restart(service)
    return {"service": service, "restarted": True}


# -- security -----------------------------------------------------------------

@action("rotate_db_credential", risk="high", domains=("security",), reversible=True,
        summary="Give a service's warehouse role a new random password and update the pipeline config to match.")
def rotate_db_credential(lab: Lab, role: str = "etl") -> dict:
    if role not in MANAGED_DB_ROLES:
        raise ActionError(f"credential of role {role!r} is not managed")
    from psycopg import sql

    password = secrets.token_urlsafe(18)
    lab.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
    lab.runtime.update("ingest", lambda cfg: cfg["warehouse"].__setitem__("password", password))
    return {"role": role, "rotated": True}


@action("block_ip", risk="medium", domains=("security",), reversible=True,
        summary="Block a client IP address at the dashboard API.")
def block_ip(lab: Lab, ip: str) -> dict:
    try:
        address = str(ipaddress.ip_address(ip))
    except ValueError as exc:
        raise ActionError(str(exc)) from None

    def mutate(cfg: dict) -> None:
        blocked = cfg.setdefault("blocked_ips", [])
        if address not in blocked:
            blocked.append(address)

    lab.runtime.update("metrics_api", mutate)
    return {"blocked": address}
