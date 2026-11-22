"""The default agent pool: a fast and a strong specialist per domain, one generalist, one policy agent.

Two variants per domain are what make agent *selection* a real decision (otherwise choosing an agent
is the same as choosing a domain): the router trades the fast agent's low cost against the strong
agent's depth, and can fall back to the other variant or the generalist when one is unavailable.

Models are local Ollama models: fast = qwen2.5:3b (~1.9 GB), strong = llama3 8B (~4.7 GB). On a 6 GB GPU
both do not fit at once, so Ollama swaps them (a few seconds per switch). Change them freely. Costs and latencies are
planning estimates in relative units, to be replaced by measured values once agents run (Block 5).

    python -m controller agents --export agents/cards   # regenerate the JSON cards
"""
from __future__ import annotations

from . import DOMAINS
from .cards import (
    DOMAIN_INTENTS,
    DOMAIN_TOOLS,
    ROUTING_EXTENSION_URI,
    SHARED_TOOLS,
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    RoutingProfile,
)

FAST_MODEL, STRONG_MODEL = "ollama/qwen2.5:3b", "ollama/llama3"
PORTS = {"network": 9301, "data": 9311, "infra": 9321, "security": 9331, "generalist": 9341, "policy": 9351}

TIERS = {
    #          model         cost  latency  concurrency  blurb
    "fast": (FAST_MODEL, 1.0, 6.0, 4, "Quick first look with a small model; cheap, good on clear-cut incidents."),
    "strong": (STRONG_MODEL, 4.0, 20.0, 2, "Deep investigation with a larger model; slower and costlier, "
                                           "for ambiguous or high-impact incidents."),
}

SKILLS: dict[str, list[AgentSkill]] = {
    "network": [
        AgentSkill(id="diagnose-connectivity", name="Diagnose connectivity",
                   description="Find why jobs cannot reach the source system or the warehouse: timeouts, "
                               "connection resets, DNS failures and slow network paths or gateways.",
                   tags=["network", "latency", "timeout", "dns", "gateway", "connection-reset"],
                   examples=["ingest failed: ReadTimeout: timed out", "server closed the connection unexpectedly",
                             "Name or service not known"]),
        AgentSkill(id="reroute-traffic", name="Reroute traffic",
                   description="Move a job's traffic to a healthy alternative path or gateway.",
                   tags=["network", "failover", "routing"]),
    ],
    "data": [
        AgentSkill(id="diagnose-pipeline-failure", name="Diagnose pipeline failures",
                   description="Explain failed ingest and transform runs caused by the data itself: schema drift, "
                               "failed data-quality tests, bad or duplicated batches.",
                   tags=["data", "etl", "dbt", "schema", "data-quality"],
                   examples=["SchemaMismatch: source payload missing required field(s)",
                             "not_null_stg_orders_amount fail: Got 3 results"]),
        AgentSkill(id="diagnose-warehouse-contention", name="Diagnose warehouse contention",
                   description="Find sessions that block, leak connections or run away with warehouse "
                               "resources: locks, idle connection leaks, expensive queries.",
                   tags=["data", "database", "locks", "connections", "queries"],
                   examples=["canceling statement due to lock timeout",
                             "remaining connection slots are reserved"]),
    ],
    "infra": [
        AgentSkill(id="diagnose-compute-pressure", name="Diagnose compute pressure",
                   description="Find CPU, memory and disk problems in the platform's containers: runaway "
                               "processes, out-of-memory kills, full disks.",
                   tags=["infra", "cpu", "memory", "oom", "disk", "capacity"],
                   examples=["dbt build was killed (SIGKILL, exit 137)", "No space left on device"]),
        AgentSkill(id="recover-deployment", name="Recover failed deployments",
                   description="Recover components broken by a bad configuration change or crash loop.",
                   tags=["infra", "deployment", "config", "crash-loop", "rollback"],
                   examples=["invalid configuration in metrics_api.json, refusing to start"]),
    ],
    "security": [
        AgentSkill(id="investigate-credential-misuse", name="Investigate credential misuse",
                   description="Investigate service credentials used by unexpected clients or for bulk data export.",
                   tags=["security", "credentials", "exfiltration", "access"],
                   examples=["service role etl used by unexpected application(s)"]),
        AgentSkill(id="investigate-login-attacks", name="Investigate login attacks",
                   description="Investigate spikes of failed logins and block abusive clients.",
                   tags=["security", "brute-force", "authentication"],
                   examples=["212 failed dashboard logins in the last 60s"]),
    ],
}


def _card(agent_id: str, description: str, port: int, profile: RoutingProfile, skills: list[AgentSkill]) -> AgentCard:
    return AgentCard(
        name=agent_id,
        description=description,
        supported_interfaces=[AgentInterface(url=f"http://localhost:{port}/a2a")],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True, extensions=[AgentExtension(
            uri=ROUTING_EXTENSION_URI, description="Routing profile used by the a2a-oncall controller.",
            params=profile.model_dump(by_alias=True))]),
        skills=skills,
    )


def default_pool() -> list[AgentCard]:
    cards = []
    for domain in DOMAINS:
        tools = list(SHARED_TOOLS) + list(DOMAIN_TOOLS[domain])
        for offset, (tier, (model, cost, latency, concurrency, blurb)) in enumerate(TIERS.items()):
            profile = RoutingProfile(role="specialist", domains=[domain], tier=tier, model=model, cost_units=cost,
                                     expected_latency_s=latency, max_concurrency=concurrency, tools=tools,
                                     allowed_intents=list(DOMAIN_INTENTS[domain]))
            cards.append(_card(f"{domain}-{tier}", f"{domain.capitalize()} specialist. {blurb}",
                               PORTS[domain] + offset, profile, SKILLS[domain]))

    all_tools = list(SHARED_TOOLS) + [t for d in DOMAINS for t in DOMAIN_TOOLS[d]]
    all_intents = sorted({i for d in DOMAINS for i in DOMAIN_INTENTS[d]})
    cards.append(_card(
        "generalist", "Generalist on-call agent with every domain's tools. Fallback when no specialist is "
                      "available, and the single-agent baseline (D) in experiments.", PORTS["generalist"],
        RoutingProfile(role="generalist", domains=list(DOMAINS), tier="strong", model=STRONG_MODEL, cost_units=6.0,
                       expected_latency_s=30.0, max_concurrency=1, tools=all_tools, allowed_intents=all_intents),
        [AgentSkill(id="general-incident-triage", name="General incident triage",
                    description="Investigate any platform incident end to end with all read-only tools.",
                    tags=["generalist", "triage"])]))
    cards.append(_card(
        "policy", "Policy agent. Checks every remediation plan against risk and permission rules before "
                  "hand-over, and decides whether a human must approve it.", PORTS["policy"],
        RoutingProfile(role="policy", model="rules+ollama/qwen2.5:3b", cost_units=0.2, expected_latency_s=1.0,
                       max_concurrency=8),
        [AgentSkill(id="validate-remediation-plan", name="Validate remediation plan",
                    description="Approve, reject or escalate a RemediationPlan.", tags=["policy", "safety"])]))
    return cards
