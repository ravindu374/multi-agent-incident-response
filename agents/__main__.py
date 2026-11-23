"""Run expert agents.

    python -m agents investigate data/synthetic/dev/incidents.jsonl --agent data-strong   # in-process
    python -m agents serve data/synthetic/dev/incidents.jsonl                            # A2A servers
    python -m agents serve data/synthetic/dev/incidents.jsonl --memory models/memory     # + what history knows
    python -m agents ask data/synthetic/dev/incidents.jsonl --agent data-strong          # over A2A
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlparse

from controller.cards import SERVICE_ROLES
from controller.contracts import load_incidents
from controller.pool import default_pool
from controller.proposal import InvestigationTask, Proposal

from .history_agent import HistoryAgent
from .policy_agent import PolicyAgent
from .reasoners import ReasonerFailed
from .reasoners.ollama import OllamaReasoner
from .reasoners.rules import RuleReasoner
from .specialist import SpecialistAgent
from .tools import RecordedEvidence

INVESTIGATORS = [c.name for c in default_pool() if c.routing_profile().role not in SERVICE_ROLES]
SERVED = [c.name for c in default_pool()]  # the investigators + the policy and history agents


def print_proposal(p: Proposal) -> None:
    print(f"{p.incident_id}  {p.agent_id} -> {p.verdict.upper()} ({p.confidence:.2f})  {p.root_cause}")
    for a in p.actions:
        print(f"    propose {a.intent} on {a.target} {a.params}  [{a.risk} risk]")
    for f in p.findings:
        print(f"    - [{f.tool}] {f.fact}")
    if p.points_to:
        print(f"    points to: {p.points_to}")
    print(f"    tools used: {', '.join(p.tools_used) or 'none'}" + (f"   usage: {p.usage}" if p.usage else ""))
    print()


def pick(incidents: Path, ids: str | None, limit: int):
    reports = load_incidents(incidents)
    if ids:
        wanted = ids.split(",")
        reports = [r for r in reports if r.incident_id in wanted]
        if not reports:
            raise SystemExit(f"none of {wanted} in {incidents}")
        return reports
    return reports[:limit]


def build_agents(names: list[str], incidents: Path, reasoner_name: str, model: str | None,
                 allowed: list[str] = INVESTIGATORS, memory: Path | None = None, trust: str = "local") -> list:
    cards = {c.name: c for c in default_pool()}
    unknown = [n for n in names if n not in cards or n not in allowed]
    if unknown:
        raise SystemExit(f"unknown agent(s) {unknown}; choose from {allowed}")
    evidence = RecordedEvidence.from_incidents_file(incidents)
    agents = []
    for name in names:
        role = cards[name].routing_profile().role
        if role == "policy":
            agents.append(PolicyAgent(cards[name]))
            continue
        if role == "history":  # the memory it starts from (and keeps saving to), or an empty one
            from controller.memory import ExperienceStore

            store = ExperienceStore.load(memory) if memory and (Path(memory) / "store.json").exists() else None
            agents.append(HistoryAgent(cards[name], store, trust=trust, path=memory))
            continue
        reasoner = OllamaReasoner(model=model) if reasoner_name == "ollama" else RuleReasoner()
        agents.append(SpecialistAgent(cards[name], reasoner, evidence))
    return agents


def cmd_investigate(args) -> int:
    agent = build_agents([args.agent], args.incidents, args.reasoner, args.model)[0]
    domain = args.domain or agent.profile.domains[0]
    for report in pick(args.incidents, args.incident, args.limit):
        try:
            print_proposal(agent.handle(InvestigationTask(incident=report, domain=domain,
                                                          must_propose=args.must_propose)))
        except ReasonerFailed as e:  # over A2A this is a failed task; the controller asks a fallback agent
            print(f"{report.incident_id}  {agent.agent_id} FAILED: {e}\n")
    return 0


def cmd_serve(args) -> int:
    from .server import port_of, serve

    names = args.agents.split(",") if args.agents else SERVED
    agents = build_agents(names, args.incidents, args.reasoner, args.model, allowed=SERVED, memory=args.memory,
                          trust=args.trust)
    print(f"serving {len(agents)} A2A agent(s) with the {args.reasoner} reasoner (Ctrl+C to stop):")
    for a in agents:
        extra = ""
        if isinstance(a, HistoryAgent):
            extra = (f"   memory: {len(a.store)} past incidents" + (f", saved to {a.path}" if a.path else
                                                                    " (not saved: no --memory)"))
        print(f"  {a.agent_id:<16} http://{args.host}:{port_of(a)}/.well-known/agent-card.json{extra}")
    try:
        asyncio.run(serve(agents, host=args.host))
    except KeyboardInterrupt:
        print("stopped")
    return 0


def cmd_ask(args) -> int:
    from controller.cards import fetch_card
    from controller.wire import AgentCallFailed, ask

    cards = {c.name: c for c in default_pool()}
    if args.agent not in cards:
        raise SystemExit(f"unknown agent {args.agent!r}; choose from {sorted(cards)}")
    if args.url:
        base = args.url
    else:
        u = urlparse(cards[args.agent].supported_interfaces[0].url)
        base = f"{u.scheme}://{u.netloc}"
    try:
        card = fetch_card(base)  # A2A discovery: the live card, not the local copy
    except Exception as e:
        raise SystemExit(f"no agent card at {base} ({type(e).__name__}: {e}). "
                         "Is `python -m agents serve ...` running?") from None
    profile = card.routing_profile()
    domain = args.domain or profile.domains[0]
    print(f"discovered {card.name} at {base} ({profile.tier} {profile.role}, model {profile.model})\n")

    async def run():
        for report in pick(args.incidents, args.incident, args.limit):
            try:
                print_proposal(await ask(card, InvestigationTask(incident=report, domain=domain,
                                                                 must_propose=args.must_propose)))
            except AgentCallFailed as e:
                print(f"{report.incident_id}  {e}\n")

    asyncio.run(run())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agents")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("investigate", help="run one agent in-process on incidents that carry an evidence bundle")
    p.add_argument("incidents", type=Path)
    p.add_argument("--agent", default="data-strong")
    p.add_argument("--domain", default=None, help="domain to examine (default: the agent's own)")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_investigate)

    p = sub.add_parser("serve", help="run expert agents as A2A servers (ports from their cards)")
    p.add_argument("incidents", type=Path, help="incidents whose evidence bundles the agents' tools read")
    p.add_argument("--agents", default=None,
                   help=f"comma-separated (default: all {len(SERVED)}, incl. the policy and history agents)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--memory", type=Path, default=None,
                   help="the history agent's experience memory: loaded if it exists, saved after every incident")
    p.add_argument("--trust", choices=["local", "global", "off"], default="local",
                   help="how the history agent reports the agents' verified record (Block 6b)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ask", help="send InvestigationTasks to a running agent over A2A")
    p.add_argument("incidents", type=Path)
    p.add_argument("--agent", default="data-strong", help="which agent (its URL comes from the pool card)")
    p.add_argument("--url", default=None, help="agent base URL, overrides the pool card")
    p.add_argument("--domain", default=None)
    p.add_argument("--limit", type=int, default=3)
    p.set_defaults(func=cmd_ask)

    for name in ("investigate", "ask"):
        sub.choices[name].add_argument("--incident", default=None, help="comma-separated incident ids")
        sub.choices[name].add_argument("--must-propose", action="store_true",
                                       help="as the controller asks: hand in a best fix even if the evidence is weak")
    for name in ("investigate", "serve"):
        sp = sub.choices[name]
        sp.add_argument("--reasoner", choices=["rules", "ollama"], default="rules")
        sp.add_argument("--model", default=None, help="override the Ollama model from the agent card")

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
