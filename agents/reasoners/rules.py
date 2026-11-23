"""Deterministic, evidence-driven reasoner: the checks an on-call engineer would make per domain.

Fast specialists only call the first FAST_TOOL_BUDGET tools of their domain (cheaper, quicker,
sometimes blind); strong specialists and the generalist call all of theirs. Each check either
finds a root cause (with a fix from the agent's allowed intents) or records what it saw.

Block 6c: when the task carries similar past incidents, a fast specialist first calls the tools that show
the evidence behind the fixes that resolved those incidents in its domain ("last time it was a schema
change: look at the source sample first"). History changes WHERE it looks, never what it concludes: every
identifier still comes from today's tool results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from controller import DOMAINS
from controller.cards import DOMAIN_TOOLS, RoutingProfile
from controller.checks import same_fix
from controller.plan import INTENTS, PlannedAction
from controller.proposal import Finding, InvestigationTask, Proposal, Review

from ..tools import Toolbox

FAST_TOOL_BUDGET = 2
# where an engineer looks for the evidence behind each kind of fix (stop_session: by what it names)
EVIDENCE_TOOL = {"map_schema_fields": "source_sample", "quarantine_records": "dbt_failures",
                 "stop_process": "top_processes", "free_storage": "disk_usage", "restore_config": "recent_changes",
                 "resize_compute": "container_stats", "restart_component": "container_stats",
                 "reroute_connection": "probe_paths", "rotate_credential": "db_client_origins",
                 "block_client": "failed_logins"}
STOP_SESSION_TOOL = {"pid": "locks", "pids": "active_queries", "application_name": "connections_by_app",
                     "usename": "active_queries", "client_addr": "active_queries"}
FAST_CONFIDENCE_FACTOR = 0.9
SLOW_PATH_MS = 3000
DISK_FULL_PCT = 95
CPU_HOT_PCT = 85
HOG_CONNECTIONS = 20
RUNAWAY_QUERY_S = 30
EXPECTED_FIELDS = ("created_at", "customer_id", "order_id", "status", "store_id", "amount")
CORE_APPS = {"ingest", "metrics-api", "dbt", "transform-runner", "postgres_exporter"}
SERVICE_USERS = {"etl", "transformer", "dashboard", "postgres", "app_reporting"}
EXPECTED_APPS = {"etl": {"ingest"}, "transformer": {"dbt", "transform-runner"}, "dashboard": {"metrics-api"}}


def action(intent: str, target: str, rationale: str, **params) -> PlannedAction:
    return PlannedAction(intent=intent, target=target, params=params, risk=INTENTS[intent].risk, rationale=rationale)


@dataclass
class Diagnosis:
    root_cause: str
    confidence: float
    actions: list[PlannedAction]


@dataclass
class Outcome:
    diagnosis: Diagnosis | None = None
    findings: list[Finding] = field(default_factory=list)
    points_to: str | None = None
    blind: bool = False  # the tools that mattered returned nothing

    def note(self, tool: str, fact: str) -> None:
        self.findings.append(Finding(tool=tool, fact=fact))

    def claim(self, root_cause: str, confidence: float, *actions: PlannedAction) -> None:
        if self.diagnosis is None:  # first (most specific) check wins
            self.diagnosis = Diagnosis(root_cause, confidence, list(actions))


# -- per-domain checks ---------------------------------------------------------------------------

def check_network(r: dict) -> Outcome:
    out = Outcome()
    probes = r.get("probe_paths")
    if probes is None:
        out.blind = True
        return out
    by_path = {p.path: p for p in probes}
    for p in probes:
        if not p.path.endswith("primary"):
            continue
        route = p.path.split("_")[0]
        alt = by_path.get(f"{route}_secondary")
        alt_ok = alt is not None and alt.error is None and (alt.p50_ms or 0) < 1000
        slow = p.p50_ms is not None and p.p50_ms > SLOW_PATH_MS
        if (p.error or slow) and alt_ok:
            problem = p.error or f"median latency {p.p50_ms:.0f} ms"
            out.note("probe_paths", f"{p.path} to {p.target}: {problem}; {alt.path} healthy ({alt.p50_ms} ms)")
            kind = ("name resolution" if p.error and "name resolution" in p.error
                    else "connection resets" if p.error else "high latency")
            out.claim(f"{kind} on the primary network path to the {p.target}", 0.85 if p.error else 0.8,
                      action("reroute_connection", "ingest", f"{alt.path} is healthy", route=route,
                             endpoint="secondary"))
    if out.diagnosis is None:
        out.note("probe_paths", "all network paths healthy (" + ", ".join(
            f"{p.path} {p.p50_ms} ms" for p in probes if p.p50_ms is not None) + ")")
    return out


def check_data(r: dict) -> Outcome:
    out = Outcome()
    fields = r.get("source_sample")
    if fields is not None and "amount" not in fields:
        renamed = [f for f in fields if f not in EXPECTED_FIELDS]
        out.note("source_sample", f"source payload fields {fields}: 'amount' missing")
        if renamed:
            out.claim(f"schema drift: the source renamed 'amount' to '{renamed[0]}'", 0.9,
                      action("map_schema_fields", "ingest", "map the renamed field back", mapping={renamed[0]: "amount"}))
    failures = r.get("dbt_failures")
    if failures:
        tests = [f.node.split(".")[2] if f.node.count(".") >= 2 else f.node for f in failures]
        out.note("dbt_failures", "failing data tests: " + "; ".join(f"{t} ({f.message})" for t, f in zip(tests, failures)))
        out.claim("bad records in the latest batch fail the data-quality tests", 0.85,
                  action("quarantine_records", "warehouse", "move the failing rows aside so dbt can rebuild",
                         tests=tests))
    locks = r.get("locks")
    if locks:
        blockers = {pid for lock in locks if not lock.granted for pid in lock.blocked_by}
        for lock in locks:
            if lock.granted and lock.pid in blockers:
                out.note("locks", f"pid {lock.pid} holds {lock.mode} on {lock.relation}, blocking other sessions")
                out.claim(f"session {lock.pid} holds an exclusive lock on {lock.relation}", 0.9,
                          action("stop_session", "warehouse", "it blocks ingest and dbt", pid=lock.pid))
    conns = r.get("connections_by_app")
    if conns:
        hogs = sorted(((n, app) for app, n in conns.items() if app not in CORE_APPS and n >= HOG_CONNECTIONS), reverse=True)
        if hogs:
            n, app = hogs[0]
            out.note("connections_by_app", f"application '{app}' holds {n} connections")
            out.claim(f"'{app}' holds {n} warehouse connections (connection leak or flood)", 0.8,
                      action("stop_session", "warehouse", "free the connection slots", application_name=app))
    queries = r.get("active_queries")
    if queries:
        for q in queries:
            if q.usename == "etl" and q.application_name not in EXPECTED_APPS["etl"]:
                out.note("active_queries", f"service role etl used by '{q.application_name}' from {q.client_addr}")
                out.points_to = "security"
        runaway = [q for q in queries if q.state == "active" and q.usename not in SERVICE_USERS
                   and q.duration_s >= RUNAWAY_QUERY_S]
        for q in runaway:
            out.note("active_queries", f"pid {q.pid} ({q.usename}/{q.application_name}) running {q.duration_s:.0f}s: "
                                       f"{q.query[:80]}")
        if runaway:
            out.claim(f"{len(runaway)} long-running query session(s) from {runaway[0].application_name} "
                      "are overloading the warehouse", 0.8,
                      action("stop_session", "warehouse", "end the runaway queries", pids=[q.pid for q in runaway]))
    out.blind = all(r.get(t) is None for t in r)
    return out


def check_infra(r: dict) -> Outcome:
    out = Outcome()
    disks = r.get("disk_usage")
    for d in disks or []:
        if d.used_pct >= DISK_FULL_PCT:
            out.note("disk_usage", f"{d.service}:{d.mount} {d.used_pct:.0f}% full; largest: {', '.join(d.largest_files)}")
            out.claim(f"{d.mount} on {d.service} is full", 0.9,
                      action("free_storage", d.service, "delete the stale files", mount=d.mount))
    changes = r.get("recent_changes") or []
    for c in changes:
        out.note("recent_changes", f"{c.at} {c.component}: {c.change}")
    containers = r.get("container_stats")
    for c in containers or []:
        if c.oom_kills and c.mem_limit_mb and c.mem_mb >= 0.9 * c.mem_limit_mb:
            changed = any(ch.component == c.service and "memory" in ch.change for ch in changes)
            out.note("container_stats", f"{c.service}: {c.oom_kills} OOM kill(s), {c.mem_mb:.0f}/{c.mem_limit_mb:.0f} MB")
            out.claim(f"{c.service} runs out of memory under a {c.mem_limit_mb:.0f} MB limit", 0.95 if changed else 0.85,
                      action("resize_compute", c.service, "restore a workable memory limit", memory_mb=512))
        if c.status == "restarting" or c.restarts >= 3:
            changed = any(ch.component == c.service for ch in changes)
            out.note("container_stats", f"{c.service} is {c.status} ({c.restarts} restarts)")
            out.claim(f"{c.service} crash-loops" + (" after a configuration change" if changed else ""),
                      0.9 if changed else 0.75,
                      action("restore_config", c.service, "roll back to the last known-good configuration"))
        if c.cpu_pct >= CPU_HOT_PCT:
            top = (r.get("top_processes") or {}).get(c.service)
            if top is None:
                out.note("container_stats", f"{c.service} CPU {c.cpu_pct:.0f}% (process list unavailable)")
                continue
            foreign = [p for p in top if p.cpu_pct >= 5 and not p.command.startswith("postgres")]
            backends = [p for p in top if p.cpu_pct >= 5 and p.command.startswith("postgres: ")]
            if foreign:
                out.note("top_processes", f"{c.service} CPU {c.cpu_pct:.0f}%: {len(foreign)} x '{foreign[0].command}'")
                out.claim(f"unknown process '{foreign[0].command}' is burning {c.service}'s CPU", 0.85,
                          action("stop_process", c.service, "the process is not part of the service",
                                 pids=[p.pid for p in foreign]))
            elif backends:
                out.note("top_processes", f"{c.service} CPU {c.cpu_pct:.0f}% is used by database sessions "
                                          f"({backends[0].command})")
                out.points_to = "data"
    if containers is None and disks is None:
        out.blind = True
    elif out.diagnosis is None and out.points_to is None and containers:
        hottest = max(containers, key=lambda c: c.cpu_pct)
        out.note("container_stats", f"all {len(containers)} containers running; hottest {hottest.service} "
                                    f"at {hottest.cpu_pct:.0f}% CPU; no OOM kills or restarts")
    return out


def check_security(r: dict) -> Outcome:
    out = Outcome()
    origins = r.get("db_client_origins")
    for o in origins or []:
        expected = EXPECTED_APPS.get(o.usename)
        if expected and o.application_name not in expected:
            out.note("db_client_origins", f"service role {o.usename} used by '{o.application_name}' from "
                                          f"{o.client_addr} ({o.sessions} sessions)")
            out.claim(f"the {o.usename} service credential is used by an unapproved client ({o.client_addr})", 0.85,
                      action("rotate_credential", "warehouse", "cut off the unapproved client", role=o.usename),
                      action("stop_session", "warehouse", "end its open sessions", usename=o.usename,
                             exclude_application_name=sorted(expected)[0]))
    logins = r.get("failed_logins")
    for ip, n in sorted((logins or {}).items(), key=lambda kv: -kv[1]):
        if n >= 50:
            out.note("failed_logins", f"{n} failed logins from {ip}")
            out.claim(f"password-guessing attack from {ip}", 0.85,
                      action("block_client", "metrics-api", "stop the attack", ip=ip))
        else:
            out.note("failed_logins", f"{n} failed login(s) from {ip} (normal level)")
    if origins is None and logins is None:
        out.blind = True
    elif out.diagnosis is None and origins:
        out.note("db_client_origins", "every service credential is used only by its own application")
    return out


CHECKS = {"network": check_network, "data": check_data, "infra": check_infra, "security": check_security}


def best_guess(domain: str, toolbox: Toolbox) -> PlannedAction | None:
    """The fix an engineer would try first in a domain when no check found a clear cause: grounded in
    the tool results (real pids, services, roles), but not aimed at anything proven. Used when the agent
    must propose (honest verdict kept) and by the careless reasoner (claimed as certain)."""
    def guess(intent: str, target: str, **params) -> PlannedAction:
        return action(intent, target, "best guess: no check found a clear cause", **params)

    def look(tool: str):  # reuse what the checks already fetched; call the tool only if they did not
        return toolbox.results[tool] if tool in toolbox.results else toolbox.try_call(tool)

    if domain == "data":
        queries = look("active_queries") or []
        return guess("stop_session", "warehouse", pid=queries[0].pid) if queries else None
    if domain == "infra":
        containers = look("container_stats") or []
        busiest = max(containers, key=lambda c: c.cpu_pct, default=None)
        return guess("restart_component", busiest.service) if busiest and busiest.service in (
            "ingest", "warehouse", "transform", "metrics-api") else None
    if domain == "network":
        return guess("reroute_connection", "ingest", route="source", endpoint="secondary")
    if domain == "security":
        origins = look("db_client_origins") or []
        return guess("rotate_credential", "warehouse", role=origins[0].usename) if origins else None
    return None


def evidence_tool(a: PlannedAction) -> str | None:
    if a.intent == "stop_session":
        return next((STOP_SESSION_TOOL[k] for k in a.params if k in STOP_SESSION_TOOL), None)
    return EVIDENCE_TOOL.get(a.intent)


def hinted_tools(domain: str, past_cases, tools: list[str]) -> list[str]:
    """The tools behind the fixes that resolved similar past incidents of this domain, most similar first."""
    hinted: list[str] = []
    for case in past_cases:
        if case.owner != domain:
            continue
        for a in case.fix_that_worked:
            t = evidence_tool(a)
            if t in tools and t not in hinted:
                hinted.append(t)
    return hinted


class RuleReasoner:
    name = "rules"

    def _domain_tools(self, domain: str, profile: RoutingProfile, past_cases=()) -> list[str]:
        tools = [t for t in DOMAIN_TOOLS[domain] if t in profile.tools]
        if not (profile.role == "specialist" and profile.tier == "fast"):
            return tools
        hinted = hinted_tools(domain, past_cases, tools)
        return (hinted + [t for t in tools if t not in hinted])[:FAST_TOOL_BUDGET]

    def investigate(self, task: InvestigationTask, agent_id: str, profile: RoutingProfile,
                    toolbox: Toolbox) -> Proposal:
        domains = [task.domain] if profile.role == "specialist" else [d for d in DOMAINS if d in profile.domains]
        outcomes, hints = {}, []
        for domain in domains:
            tools = self._domain_tools(domain, profile, task.past_cases)
            results = {t: toolbox.try_call(t) for t in tools}
            outcomes[domain] = CHECKS[domain](results)
            hinted = [t for t in hinted_tools(domain, task.past_cases, tools) if t in tools]
            if hinted and profile.tier == "fast" and profile.role == "specialist":
                hints.append(Finding(tool="history", fact=f"similar past {domain} incidents were resolved by fixes "
                                                          f"whose evidence is in {', '.join(hinted)}: looked there first"))

        diagnosed = [(o.diagnosis.confidence, d) for d, o in outcomes.items() if o.diagnosis]
        factor = FAST_CONFIDENCE_FACTOR if profile.tier == "fast" and profile.role == "specialist" else 1.0
        findings = [f for o in outcomes.values() for f in o.findings] + hints
        if diagnosed:
            _, domain = max(diagnosed)
            dx = outcomes[domain].diagnosis
            return Proposal(incident_id=task.incident.incident_id, agent_id=agent_id, domain=domain, verdict="owner",
                            root_cause=dx.root_cause, confidence=round(dx.confidence * factor, 3), findings=findings,
                            actions=dx.actions, points_to=None, reasoner=self.name)
        points = next((o.points_to for o in outcomes.values() if o.points_to), None)
        blind = all(o.blind for o in outcomes.values())
        root_cause = "" if blind else f"no root cause found in {', '.join(domains)}"
        guess = best_guess(task.domain, toolbox) if task.must_propose else None
        if guess:
            root_cause = f"{root_cause or 'my evidence is incomplete'}; best guess: {guess.intent} on {guess.target}"
        return Proposal(
            incident_id=task.incident.incident_id, agent_id=agent_id, domain=task.domain,
            verdict="unsure" if blind else "not_mine", root_cause=root_cause,
            confidence=0.3 if blind else round(0.7 * factor, 3), findings=findings, points_to=points,
            actions=[guess] if guess else [], reasoner=self.name)

    def review(self, task: InvestigationTask, agent_id: str, profile: RoutingProfile, toolbox: Toolbox) -> Review:
        """Check another agent's claim against my own domain's evidence (I investigate, then compare)."""
        claim = task.claim
        own = self.investigate(task.model_copy(update={"role": "primary", "claim": None, "must_propose": False}),
                               agent_id, profile, toolbox)
        seen = "; ".join(f.fact for f in own.findings[:2]) or "nothing unusual in my tools"

        def say(stance: str, reason: str, confidence: float) -> Review:
            return Review(incident_id=task.incident.incident_id, reviewer=agent_id, domain=task.domain,
                          claim_agent=claim.agent_id, claim_domain=claim.domain, stance=stance, reason=reason,
                          confidence=round(confidence, 3), reasoner=self.name)

        if own.verdict == "owner" and own.domain == claim.domain:
            if same_fix(own.actions, claim.actions):
                return say("supports", f"I found the same cause: {own.root_cause}", own.confidence)
            return say("contradicts", f"I found a different cause in {own.domain}: {own.root_cause}", own.confidence)
        if own.verdict == "owner":
            return say("contradicts", f"my {own.domain} evidence shows its own root cause: {own.root_cause}",
                       own.confidence)
        if own.points_to == claim.domain:
            return say("supports", f"my evidence points to {claim.domain}: {seen}", 0.7)
        if own.points_to:
            return say("contradicts", f"my evidence points to {own.points_to}, not {claim.domain}: {seen}", 0.5)
        if own.verdict == "unsure":
            return say("cannot_tell", f"I could not check {task.domain} (evidence missing)", 0.0)
        return say("cannot_tell", f"{task.domain} looks healthy, which says nothing for or against: {seen}", 0.5)

