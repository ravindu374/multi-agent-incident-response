"""Fault catalogue: what the lab can break, who truly owns each fault, and how to undo it.

Ownership rule: the owner is the domain whose fix resolves the root cause, not the domain
where the symptom shows up. `symptom_tags` record which faults look alike from the outside;
every fault shares a non-generic tag with another owner's fault, so symptoms alone never
identify the owner (tests/test_faults_catalog.py enforces this).

This module is ground truth. Nothing in it is ever shown to agents.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from . import actions
from .compose import DockerError
from .lab import Lab, new_id, utcnow_iso

ActionCall = tuple[str, dict[str, Any]]

FAULTS: dict[str, Fault] = {}


def register(cls: type[Fault]) -> type[Fault]:
    FAULTS[cls.id] = cls()
    return cls


class Fault(ABC):
    id: ClassVar[str]
    title: ClassVar[str]
    owner: ClassVar[str]
    involved: ClassVar[tuple[str, ...]]  # domains whose evidence matters for diagnosis
    symptom_tags: ClassVar[tuple[str, ...]]
    valid_fixes: ClassVar[tuple[str, ...]]  # action names that resolve the root cause
    defaults: ClassVar[dict[str, Any]] = {}
    fix_needs_lookup: ClassVar[bool] = False  # reference fix must inspect the live system

    @abstractmethod
    def inject(self, lab: Lab, params: dict[str, Any]) -> dict:
        """Break the sandbox; return details the fix/revert steps need."""

    @abstractmethod
    def reference_fix(self, lab: Lab, details: dict) -> list[ActionCall]:
        """The allowlisted actions a perfect on-call engineer would take."""

    @abstractmethod
    def revert(self, lab: Lab, details: dict | None) -> None:
        """Undo the fault completely. Must be idempotent and work without details."""

    def ground_truth(self) -> dict:
        return {
            "fault_id": self.id,
            "owner": self.owner,
            "involved": list(self.involved),
            "symptom_tags": list(self.symptom_tags),
            "valid_fixes": list(self.valid_fixes),
        }


# -- helpers --------------------------------------------------------------------

def _terminate_app_sessions(lab: Lab, application_name: str) -> None:
    lab.sql(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE application_name = %s AND backend_type = 'client backend'",
        (application_name,),
    )


def _pids_for_app(lab: Lab, application_name: str) -> list[int]:
    rows = lab.sql(
        "SELECT pid FROM pg_stat_activity WHERE application_name = %s AND backend_type = 'client backend'",
        (application_name,),
    )
    return [r[0] for r in rows]


def _db_args(user: str, password: str, app: str) -> list[str]:
    return ["--user", user, "--password", password, "--app", app]


# -- network ----------------------------------------------------------------------

@register
class SourceLatency(Fault):
    id = "N1"
    title = "Heavy latency on the primary network path to the orders API"
    owner = "network"
    involved = ("network", "data")
    symptom_tags = ("data_stale", "ingest_timeout")
    valid_fixes = ("switch_source_endpoint",)
    defaults = {"latency_ms": 8000}

    def inject(self, lab, params):
        lab.toxiproxy.add_toxic("source_primary", "t1", "latency",
                                {"latency": int(params["latency_ms"]), "jitter": 500})
        return {}

    def reference_fix(self, lab, details):
        return [("switch_source_endpoint", {"endpoint": "secondary"})]

    def revert(self, lab, details):
        lab.toxiproxy.remove_toxic("source_primary", "t1")
        lab.runtime.restore_default("ingest")


@register
class WarehouseLinkResets(Fault):
    id = "N2"
    title = "Connections on the primary path to the warehouse are reset"
    owner = "network"
    involved = ("network", "data")
    symptom_tags = ("data_stale", "ingest_connection_error")
    valid_fixes = ("switch_warehouse_endpoint",)
    defaults = {"toxicity": 1.0}

    def inject(self, lab, params):
        lab.toxiproxy.add_toxic("warehouse_primary", "t2", "reset_peer", {"timeout": 0},
                                toxicity=float(params["toxicity"]))
        return {}

    def reference_fix(self, lab, details):
        return [("switch_warehouse_endpoint", {"endpoint": "secondary"})]

    def revert(self, lab, details):
        lab.toxiproxy.remove_toxic("warehouse_primary", "t2")
        lab.runtime.restore_default("ingest")


@register
class SourceHostUnresolvable(Fault):
    id = "N3"
    title = "Orders API hostname on the primary path no longer resolves"
    owner = "network"
    involved = ("network", "data")
    symptom_tags = ("data_stale", "ingest_connection_error")
    valid_fixes = ("rollback_service_config", "switch_source_endpoint")
    defaults = {"bad_url": "http://orders-source.internal:18080"}

    def inject(self, lab, params):
        bad_url = params["bad_url"]
        lab.runtime.update("ingest", lambda cfg: cfg["source"]["endpoints"].__setitem__("primary", bad_url))
        return {}

    def reference_fix(self, lab, details):
        return [("rollback_service_config", {"service": "ingest"})]

    def revert(self, lab, details):
        lab.runtime.restore_default("ingest")


# -- data -------------------------------------------------------------------------

@register
class SchemaDrift(Fault):
    id = "D1"
    title = "Source renamed `amount` to `total_amount`"
    owner = "data"
    involved = ("data",)
    symptom_tags = ("data_stale", "ingest_rejected_batch")
    valid_fixes = ("set_field_mapping",)

    def inject(self, lab, params):
        lab.set_source_faults(schema_drift=True)
        return {}

    def reference_fix(self, lab, details):
        return [("set_field_mapping", {"mapping": {"total_amount": "amount"}})]

    def revert(self, lab, details):
        lab.set_source_faults(schema_drift=False)
        lab.runtime.restore_default("ingest")


@register
class BadRows(Fault):
    id = "D2"
    title = "Source sends a batch with null amounts and re-sent orders"
    owner = "data"
    involved = ("data",)
    symptom_tags = ("data_stale", "transform_failed")
    valid_fixes = ("quarantine_rows",)
    defaults = {"batches": 1}
    fix_needs_lookup = True

    def inject(self, lab, params):
        lab.set_source_faults(bad_rows_batches=int(params["batches"]))
        return {}

    def reference_fix(self, lab, details):
        return [("quarantine_rows", {"batch_id": row[0]}) for row in lab.sql(actions.BAD_BATCHES_SQL)]

    def revert(self, lab, details):
        lab.set_source_faults(bad_rows_batches=0)
        for (batch_id,) in lab.sql(actions.BAD_BATCHES_SQL):
            actions.quarantine_rows(lab, batch_id)


@register
class TableLock(Fault):
    id = "D3"
    title = "Ad-hoc maintenance session holds an exclusive lock on raw.orders"
    owner = "data"
    involved = ("data",)
    symptom_tags = ("data_stale", "ingest_timeout", "transform_failed")
    valid_fixes = ("terminate_backend", "terminate_sessions")
    fix_needs_lookup = True
    APP = "adhoc-maintenance"

    def inject(self, lab, params):
        lab.compose.exec(
            "warehouse", "psql", "-U", "maint", "-d", "warehouse", "-c",
            "BEGIN; LOCK TABLE raw.orders IN ACCESS EXCLUSIVE MODE; SELECT pg_sleep(86400);",
            detach=True, env={"PGAPPNAME": self.APP},
        )
        return {"application_name": self.APP}

    def reference_fix(self, lab, details):
        return [("terminate_backend", {"pid": pid}) for pid in _pids_for_app(lab, self.APP)]

    def revert(self, lab, details):
        _terminate_app_sessions(lab, self.APP)


@register
class ConnectionLeak(Fault):
    id = "D4"
    title = "Reporting worker leaks connections until the warehouse runs out"
    owner = "data"
    involved = ("data",)
    symptom_tags = ("data_stale", "ingest_connection_error", "db_saturation")
    valid_fixes = ("terminate_sessions", "terminate_backend")
    defaults = {"max": 60}
    CONTAINER, APP = "reporting-worker", "reporting-worker"

    def inject(self, lab, params):
        lab.compose.start_worker(
            self.CONTAINER, "python", "report_worker.py",
            *_db_args("app_reporting", "reporting_pw", self.APP), "--max", str(int(params["max"])),
        )
        return {"container": self.CONTAINER}

    def reference_fix(self, lab, details):
        return [("terminate_sessions", {"application_name": self.APP})]

    def revert(self, lab, details):
        lab.compose.remove_container(self.CONTAINER)
        _terminate_app_sessions(lab, self.APP)


@register
class RunawayQuery(Fault):
    id = "D5"
    title = "Analyst notebook runs CPU-heavy queries against the warehouse"
    owner = "data"
    involved = ("data", "infra")
    symptom_tags = ("db_cpu_high", "dashboard_slow", "data_stale")
    valid_fixes = ("terminate_sessions", "terminate_backend")
    defaults = {"threads": 2, "fanout": 300}
    CONTAINER, APP = "analytics-notebook", "jupyter-notebook"

    def inject(self, lab, params):
        lab.compose.start_worker(
            self.CONTAINER, "python", "notebook_job.py",
            *_db_args("analyst", "analyst_pw", self.APP),
            "--threads", str(int(params["threads"])), "--fanout", str(int(params["fanout"])),
        )
        return {"container": self.CONTAINER}

    def reference_fix(self, lab, details):
        return [("terminate_sessions", {"application_name": self.APP})]

    def revert(self, lab, details):
        lab.compose.remove_container(self.CONTAINER)
        _terminate_app_sessions(lab, self.APP)


# -- infrastructure -------------------------------------------------------------------

@register
class CpuHog(Fault):
    id = "I1"
    title = "Unknown process burns the warehouse container's CPU"
    owner = "infra"
    involved = ("infra", "data")
    symptom_tags = ("db_cpu_high", "dashboard_slow", "data_stale")
    valid_fixes = ("kill_container_process",)
    defaults = {"procs": 8}
    fix_needs_lookup = True
    PIDFILE = "/tmp/.maint.pids"

    def inject(self, lab, params):
        lab.compose.exec("warehouse", "sh", "/usr/local/lib/maint/backup-compress.sh",
                         str(int(params["procs"])), detach=True)
        return {"pidfile": self.PIDFILE}

    def reference_fix(self, lab, details):
        out = lab.compose.exec("warehouse", "cat", self.PIDFILE, check=False)
        pids = [int(tok) for tok in out.split() if tok.isdigit()]
        return [("kill_container_process", {"service": "warehouse", "pids": pids})] if pids else []

    def revert(self, lab, details):
        lab.compose.exec(
            "warehouse", "sh", "-c",
            f"if [ -f {self.PIDFILE} ]; then kill -9 $(cat {self.PIDFILE}) 2>/dev/null; rm -f {self.PIDFILE}; fi; true",
            check=False,
        )


@register
class TransformOutOfMemory(Fault):
    id = "I2"
    title = "Transform container's memory limit is too small for dbt"
    owner = "infra"
    involved = ("infra", "data")
    symptom_tags = ("data_stale", "transform_failed")
    valid_fixes = ("set_memory_limit",)
    defaults = {"memory_mb": 64}
    NORMAL_MB = 512

    def inject(self, lab, params):
        # `docker update` is refused while dbt is mid-run above the new limit; retry between runs.
        for attempt in range(6):
            try:
                lab.compose.set_memory_limit("transform", int(params["memory_mb"]))
                return {}
            except DockerError:
                if attempt == 5:
                    raise
                time.sleep(5)
        return {}

    def reference_fix(self, lab, details):
        return [("set_memory_limit", {"service": "transform", "memory_mb": self.NORMAL_MB})]

    def revert(self, lab, details):
        lab.compose.set_memory_limit("transform", self.NORMAL_MB)


@register
class SpoolFull(Fault):
    id = "I3"
    title = "Ingest worker's spool disk is full"
    owner = "infra"
    involved = ("infra", "data")
    symptom_tags = ("data_stale", "ingest_rejected_batch")
    valid_fixes = ("purge_spool",)

    def inject(self, lab, params):
        lab.compose.exec("ingest", "sh", "-c",
                         "dd if=/dev/zero of=/spool/stale-backfill.tmp bs=1M count=64 2>/dev/null; true")
        return {}

    def reference_fix(self, lab, details):
        return [("purge_spool", {})]

    def revert(self, lab, details):
        lab.compose.exec("ingest", "sh", "-c", "find /spool -mindepth 1 -delete", check=False)


@register
class DashboardBadConfig(Fault):
    id = "I4"
    title = "Bad config deploy crash-loops the dashboard API"
    owner = "infra"
    involved = ("infra",)
    symptom_tags = ("dashboard_degraded",)
    valid_fixes = ("rollback_service_config",)

    def inject(self, lab, params):
        lab.runtime.update("metrics_api", lambda cfg: cfg.__setitem__("cache_ttl_s", "5s"))
        lab.compose.restart("metrics-api")
        return {}

    def reference_fix(self, lab, details):
        return [("rollback_service_config", {"service": "metrics_api"})]

    def revert(self, lab, details):
        lab.runtime.restore_default("metrics_api")
        lab.compose.restart("metrics-api")


# -- security ----------------------------------------------------------------------

@register
class CredentialMisuse(Fault):
    id = "S1"
    title = "Unapproved client bulk-exports raw orders with the ETL credential"
    owner = "security"
    involved = ("security", "data", "infra")
    symptom_tags = ("db_cpu_high", "db_saturation", "dashboard_slow")
    valid_fixes = ("rotate_db_credential", "terminate_sessions")
    defaults = {"threads": 6}
    CONTAINER = "bi-sync-7f3a"

    def inject(self, lab, params):
        password = lab.runtime.load("ingest")["warehouse"]["password"]
        lab.compose.start_worker(
            self.CONTAINER, "python", "bi_sync.py",
            *_db_args("etl", password, "psql"), "--threads", str(int(params["threads"])),
        )
        return {"container": self.CONTAINER}

    def reference_fix(self, lab, details):
        return [
            ("rotate_db_credential", {"role": "etl"}),
            ("terminate_sessions", {"usename": "etl", "exclude_application_name": "ingest"}),
        ]

    def revert(self, lab, details):
        from psycopg import sql

        lab.compose.remove_container(self.CONTAINER)
        default_password = lab.runtime.default("ingest")["warehouse"]["password"]
        lab.execute(sql.SQL("ALTER ROLE etl PASSWORD {}").format(sql.Literal(default_password)))
        lab.runtime.restore_default("ingest")
        lab.sql(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE usename = 'etl' AND application_name <> 'ingest' AND backend_type = 'client backend'"
        )


@register
class LoginBruteForce(Fault):
    id = "S2"
    title = "Password-guessing client hammers the dashboard login"
    owner = "security"
    involved = ("security", "infra")
    symptom_tags = ("dashboard_slow", "dashboard_degraded")
    valid_fixes = ("block_ip",)
    defaults = {"threads": 16}
    fix_needs_lookup = True
    CONTAINER = "web-client-2c91"

    def inject(self, lab, params):
        lab.compose.start_worker(self.CONTAINER, "python", "web_client.py", "--threads", str(int(params["threads"])))
        return {"container": self.CONTAINER}

    def reference_fix(self, lab, details):
        return [("block_ip", {"ip": lab.compose.container_ip(self.CONTAINER)})]

    def revert(self, lab, details):
        lab.compose.remove_container(self.CONTAINER)
        lab.runtime.restore_default("metrics_api")


# -- lifecycle ------------------------------------------------------------------------

def inject_fault(lab: Lab, fault_id: str, overrides: dict[str, Any] | None = None) -> dict:
    fault = FAULTS[fault_id]
    params = {**fault.defaults, **(overrides or {})}
    details = fault.inject(lab, params) or {}
    record = {
        "injection_id": new_id("inj"),
        "injected_at": utcnow_iso(),
        "params": params,
        "details": details,
        **fault.ground_truth(),
    }
    lab.ledger.append("injections", record)
    active = lab.ledger.load_active()
    active[fault_id] = record
    lab.ledger.save_active(active)
    return record


def apply_reference_fix(lab: Lab, fault_id: str, details: dict | None = None) -> list[str]:
    fault = FAULTS[fault_id]
    if details is None:
        details = lab.ledger.load_active().get(fault_id, {}).get("details", {})
    applied = []
    for name, params in fault.reference_fix(lab, details):
        actions.execute(lab, name, params, actor=f"reference_fix:{fault_id}")
        applied.append(name)
    return applied


def revert_fault(lab: Lab, fault_id: str) -> None:
    active = lab.ledger.load_active()
    record = active.pop(fault_id, None)
    FAULTS[fault_id].revert(lab, record["details"] if record else None)
    lab.ledger.save_active(active)
    lab.ledger.append("reverts", {"at": utcnow_iso(), "fault_id": fault_id,
                                  "injection_id": record["injection_id"] if record else None})


def revert_all(lab: Lab) -> dict[str, str]:
    """Best-effort return to baseline: every revert, then default configs and a clean proxy."""
    outcome = {}
    for fault_id in FAULTS:
        try:
            revert_fault(lab, fault_id)
            outcome[fault_id] = "ok"
        except Exception as exc:  # keep going; report what failed
            outcome[fault_id] = f"{type(exc).__name__}: {exc}"
    for step, fn in (("toxiproxy", lab.toxiproxy.reset), ("source", lab.clear_source_faults),
                     ("runtime", lambda: lab.runtime.init(overwrite=True))):
        try:
            fn()
            outcome[step] = "ok"
        except Exception as exc:
            outcome[step] = f"{type(exc).__name__}: {exc}"
    return outcome
