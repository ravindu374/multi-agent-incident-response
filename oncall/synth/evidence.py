"""The healthy-platform evidence bundle that scenarios and noise then modify."""
from __future__ import annotations

from controller.evidence import (
    ActiveQuery,
    ClientOrigin,
    ContainerStat,
    DataEvidence,
    DiskUsage,
    EvidenceBundle,
    InfraEvidence,
    NetworkEvidence,
    PathProbe,
    ProcessInfo,
    SecurityEvidence,
)

from .world import SERVICES, World

PATHS = (("source_primary", "orders-api"), ("source_secondary", "orders-api"),
         ("warehouse_primary", "warehouse"), ("warehouse_secondary", "warehouse"))
SOURCE_FIELDS = ("created_at", "customer_id", "order_id", "status", "store_id", "amount")
#                 cpu % range      memory MB range  limit MB
CONTAINER_PROFILE = {
    "warehouse": ((2, 12), (60, 140), 768),
    "orders-api": ((1, 4), (45, 60), None),
    "toxiproxy": ((0, 1), (10, 20), None),
    "ingest": ((1, 5), (40, 60), None),
    "transform": ((3, 40), (90, 180), 512),
    "metrics-api": ((1, 5), (55, 75), None),
    "postgres-exporter": ((0, 2), (15, 25), None),
    "prometheus": ((1, 4), (60, 120), None),
}
POSTGRES_PROCESSES = ("postgres", "postgres: checkpointer", "postgres: background writer", "postgres: walwriter",
                      "postgres: autovacuum launcher", "postgres: logical replication launcher")


def probe(bundle: EvidenceBundle, path: str) -> PathProbe:
    return next(p for p in bundle.network.probes if p.path == path)


def container(bundle: EvidenceBundle, service: str) -> ContainerStat:
    return next(c for c in bundle.infra.containers if c.service == service)


def add_container(bundle: EvidenceBundle, world: World, service: str, cpu: float, mem: float) -> None:
    bundle.infra.containers.append(ContainerStat(service=service, cpu_pct=round(cpu, 1), mem_mb=round(mem, 1)))


def baseline_bundle(world: World) -> EvidenceBundle:
    r, ips = world.rng, world.ips
    network = NetworkEvidence(probes=[PathProbe(path=p, target=t, p50_ms=round(r.uniform(0.6, 6.0), 1))
                                      for p, t in PATHS])

    sessions = {"ingest": r.randint(0, 1), "metrics-api": r.randint(0, 2), "dbt": r.randint(0, 1),
                "postgres_exporter": 1}
    queries = [ActiveQuery(pid=world.pid("exporter"), usename="postgres", application_name="postgres_exporter",
                           client_addr=ips["postgres-exporter"], state="idle", duration_s=round(r.uniform(0.5, 9.5), 1),
                           query="SELECT pg_database.datname, pg_database_size(pg_database.datname) FROM pg_database")]
    if sessions["metrics-api"]:
        queries.append(ActiveQuery(pid=world.pid("dashboard"), usename="dashboard", application_name="metrics-api",
                                   client_addr=ips["metrics-api"], state="idle", duration_s=round(r.uniform(0.1, 4), 1),
                                   query="SELECT coalesce(sum(revenue), 0), coalesce(sum(order_count), 0), "
                                         "max(_source_max_created_at) FROM marts.orders_mart WHERE minute >= ..."))
    data = DataEvidence(active_queries=queries, connections_by_app=sessions,
                        source_sample_fields=list(SOURCE_FIELDS))

    containers = []
    for service in SERVICES:
        (c_lo, c_hi), (m_lo, m_hi), limit = CONTAINER_PROFILE[service]
        containers.append(ContainerStat(service=service, cpu_pct=round(r.uniform(c_lo, c_hi), 1),
                                        mem_mb=round(r.uniform(m_lo, m_hi), 1), mem_limit_mb=limit))
    warehouse_top = [ProcessInfo(pid=1 if i == 0 else world.pid(f"pg-{i}"), cpu_pct=round(r.uniform(0, 1.5), 1),
                                 command=cmd) for i, cmd in enumerate(POSTGRES_PROCESSES)]
    infra = InfraEvidence(
        containers=containers,
        top_processes={"warehouse": warehouse_top},
        disks=[DiskUsage(service="ingest", mount="/spool", used_pct=round(r.uniform(0, 2), 1)),
               DiskUsage(service="warehouse", mount="/var/lib/postgresql/data", used_pct=round(r.uniform(4, 18), 1))],
    )

    origins = [
        ClientOrigin(usename="etl", application_name="ingest", client_addr=ips["toxiproxy"], sessions=sessions["ingest"]),
        ClientOrigin(usename="dashboard", application_name="metrics-api", client_addr=ips["metrics-api"],
                     sessions=sessions["metrics-api"]),
        ClientOrigin(usename="transformer", application_name="dbt", client_addr=ips["transform"],
                     sessions=sessions["dbt"]),
        ClientOrigin(usename="postgres", application_name="postgres_exporter", client_addr=ips["postgres-exporter"],
                     sessions=1),
    ]
    security = SecurityEvidence(db_client_origins=[o for o in origins if o.sessions > 0])
    return EvidenceBundle(network=network, data=data, infra=infra, security=security)
