"""Red herrings and missing evidence, drawn independently of the fault.

Every item here is something the real sandbox can also produce (a mistyped password, a harmless
restart, an idle session, a dbt CPU burst, an old transient failure, a slow probe). None of it
is strong enough to breach an SLO on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from controller.evidence import ActiveQuery, ChangeEvent, EvidenceBundle

from .evidence import container
from .world import GATEWAY, World

HARMLESS_CHANGES = (
    ("prometheus", "scrape interval changed from 15s to 10s"),
    ("orders-api", "container restarted"),
    ("metrics-api", "image rebuilt (no config change)"),
    ("postgres-exporter", "image updated to v0.15.0"),
)
DROPPABLE = ("recent_changes", "top_processes", "probes", "active_queries")


@dataclass
class Noise:
    benign_logins: list[float] = field(default_factory=list)  # times of mistyped analyst passwords
    harmless_change: tuple[str, str] | None = None
    idle_transaction: bool = False
    transform_cpu_burst: bool = False
    transient_ingest_failure_at: float | None = None
    slow_probe: str | None = None
    missing: str | None = None  # one evidence source the tools failed to return

    def labels(self) -> list[str]:
        out = []
        if self.benign_logins:
            out.append(f"benign_failed_logins:{len(self.benign_logins)}")
        if self.harmless_change:
            out.append("harmless_change")
        if self.idle_transaction:
            out.append("idle_transaction")
        if self.transform_cpu_burst:
            out.append("transform_cpu_burst")
        if self.transient_ingest_failure_at is not None:
            out.append("old_transient_ingest_failure")
        if self.slow_probe:
            out.append("slow_healthy_probe")
        if self.missing:
            out.append(f"missing:{self.missing}")
        return out


def draw_noise(world: World) -> Noise:
    r = world.rng
    noise = Noise()
    if r.random() < 0.25:
        noise.benign_logins = sorted(r.uniform(-80, 200) for _ in range(r.randint(1, 4)))
    if r.random() < 0.25:
        noise.harmless_change = r.choice(HARMLESS_CHANGES)
    noise.idle_transaction = r.random() < 0.2
    noise.transform_cpu_burst = r.random() < 0.3
    if r.random() < 0.3:
        noise.transient_ingest_failure_at = r.uniform(-80, -45)
    if r.random() < 0.2:
        noise.slow_probe = r.choice(["source_secondary", "warehouse_secondary", "source_primary", "warehouse_primary"])
    if r.random() < 0.15:
        noise.missing = r.choice(DROPPABLE)
    return noise


def apply_evidence_noise(bundle: EvidenceBundle, world: World, noise: Noise, t_capture: float) -> None:
    r = world.rng
    recent = [t for t in noise.benign_logins if t <= t_capture]
    if recent:
        bundle.security.failed_logins_by_ip[GATEWAY] = bundle.security.failed_logins_by_ip.get(GATEWAY, 0) + len(recent)
    if noise.harmless_change:
        component, change = noise.harmless_change
        bundle.infra.recent_changes.append(ChangeEvent(at=world.iso(-r.uniform(30, 900)), component=component,
                                                       change=change))
    if noise.idle_transaction:
        bundle.data.active_queries.append(ActiveQuery(
            pid=world.pid("dbeaver"), usename="analyst", application_name="DBeaver 24.2.0", client_addr=GATEWAY,
            state="idle in transaction", duration_s=round(r.uniform(120, 900), 1),
            query="SELECT count(*) FROM marts.orders_mart"))
    if noise.transform_cpu_burst:
        container(bundle, "transform").cpu_pct = round(r.uniform(65, 95), 1)
    if noise.slow_probe:
        probe = next(p for p in bundle.network.probes if p.path == noise.slow_probe)
        if probe.error is None and probe.p50_ms is not None and probe.p50_ms < 100:
            probe.p50_ms = round(r.uniform(25, 90), 1)
    if noise.missing == "recent_changes":
        bundle.infra.recent_changes = []
    elif noise.missing == "top_processes":
        bundle.infra.top_processes = {}
    elif noise.missing == "probes":
        bundle.network.probes = []
    elif noise.missing == "active_queries":
        bundle.data.active_queries = []
