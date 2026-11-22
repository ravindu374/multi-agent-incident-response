"""Inspect what the controller sees: python -m controller intake lab/incidents.jsonl"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import RuleClassifier, choose_team
from .contracts import load_incidents
from .intake import extract
from .plan import ExecutionReport, RemediationPlan
from .pool import default_pool
from .registry import AgentRegistry


def schema_documents() -> dict[str, str]:
    return {
        "remediation_plan.schema.json": json.dumps(RemediationPlan.model_json_schema(), indent=2) + "\n",
        "execution_report.schema.json": json.dumps(ExecutionReport.model_json_schema(), indent=2) + "\n",
    }


def write_schemas(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in schema_documents().items():
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)
    return written


def load_registry(cards_dir: Path) -> AgentRegistry:
    if cards_dir.exists() and any(cards_dir.glob("*.json")):
        return AgentRegistry.from_directory(cards_dir)
    registry = AgentRegistry()
    for card in default_pool():
        registry.register(card)
    return registry


def agents(args) -> int:
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        for card in default_pool():
            (args.export / f"{card.name}.json").write_text(json.dumps(card.to_json(), indent=2) + "\n",
                                                           encoding="utf-8")
        print(f"wrote {len(default_pool())} agent cards to {args.export}")
        return 0

    registry = load_registry(args.cards)
    for agent_id in args.down:
        registry.mark_down(agent_id)
    print(f"{'agent':16} {'role':10} {'domains':30} {'tier':6} {'model':24} {'cost':>5} {'latency':>8}  status")
    for agent_id in registry.agents():
        p, s = registry.profile(agent_id), registry.state(agent_id)
        print(f"{agent_id:16} {p.role:10} {','.join(p.domains) or '-':30} {p.tier:6} {p.model:24} "
              f"{p.cost_units:>5.1f} {p.expected_latency_s:>7.0f}s  {s.status}")

    if args.incidents:
        classifier = RuleClassifier()
        if args.model:
            from .learned import LogRegClassifier

            classifier = LogRegClassifier.load(args.model)
        print()
        for report in load_incidents(args.incidents)[: args.limit]:
            team = choose_team(classifier.predict(extract(report)))
            print(f"{report.incident_id}  team={team.team}  ({'confident' if team.confident else 'unsure'})")
            for domain in team.team:
                ok, excluded = registry.eligible(domain)
                print(f"  {domain:9} eligible: {', '.join(c.agent_id for c in ok) or 'NONE'}")
                for x in excluded:
                    print(f"  {'':9} skipped:  {x.agent_id} ({x.reason})")
    return 0


def route(args) -> int:
    from .conformal import ConformalCalibrator
    from .router import Router

    classifier = RuleClassifier()
    if args.model:
        from .learned import LogRegClassifier

        classifier = LogRegClassifier.load(args.model)
    calibrator = ConformalCalibrator.load(args.calibration) if args.calibration else None
    registry = load_registry(args.cards)
    for agent_id in args.down:
        registry.mark_down(agent_id)
    router = Router(classifier, registry, calibrator)
    for report in load_incidents(args.incidents)[: args.limit]:
        d = router.route(report)
        probs = "  ".join(f"{k}={v:.2f}" for k, v in sorted(d.owner_probs.items(), key=lambda kv: -kv[1]))
        print(f"{d.incident_id}  {d.priority}  team={d.team} ({'confident' if d.confident else 'unsure'})  {probs}")
        for role, choice in [("primary", d.primary)] + [("consult", c) for c in d.consults]:
            if choice:
                print(f"  {role:8} {choice.agent_id:16} for {choice.domain:9} P(owner)={choice.p_owner:.2f} "
                      f"P(success)={choice.expected_success:.2f} cost={choice.cost_units:g} "
                      f"utility={choice.utility:.1f}  fallbacks={choice.fallbacks}")
        if d.unassigned:
            print(f"  UNASSIGNED {d.unassigned}")
        print(f"  expected cost {d.expected_cost:g}" + ("   ** notify human **" if d.notify_human else ""))
        if args.why:
            for reason in d.reasons:
                print(f"    - {reason}")
        print()
    return 0


def deliberate(args) -> int:
    """Route each incident, then run the A2A rounds against running agent servers (python -m agents serve)."""
    import asyncio

    import httpx

    from .conformal import ConformalCalibrator
    from .orchestrator import A2ALink, OrchestratorPolicy, format_deliberation, probe_agents
    from .optimizer import OptimizerPolicy
    from .pipeline import Controller, format_plan
    from .router import Router

    classifier = RuleClassifier()
    if args.model:
        from .learned import LogRegClassifier

        classifier = LogRegClassifier.load(args.model)
    calibrator = ConformalCalibrator.load(args.calibration) if args.calibration else None
    registry = load_registry(args.cards)
    router = Router(classifier, registry, calibrator)
    policy = OrchestratorPolicy(cross_check=args.cross_check, follow_pointers=not args.no_pointers,
                                must_propose=not args.may_abstain)
    reports = load_incidents(args.incidents)
    if args.incident:
        wanted = args.incident.split(",")
        reports = [r for r in reports if r.incident_id in wanted]
    else:
        reports = reports[: args.limit]

    store = None
    if args.memory:
        from .memory import ExperienceStore

        store = ExperienceStore.load(args.memory)
        print(f"trusting agents by their verified record ({args.trust}): {len(store)} past incidents in {args.memory}\n")

    async def run() -> int:
        async with httpx.AsyncClient(timeout=args.timeout) as http:
            controller = Controller(router, A2ALink(http), policy, OptimizerPolicy(approval=not args.no_approval),
                                    memory=store, trust=args.trust)
            for report in reports:
                alive = await probe_agents(registry, http)  # discovery doubles as the heartbeat
                for agent_id in args.down:
                    registry.mark_down(agent_id)
                if not any(ok for a, ok in alive.items() if registry.profile(a).role != "policy"):
                    print("no agent answered at its card URL. Start them first:\n"
                          "  python -m agents serve data/synthetic/dev/incidents.jsonl")
                    return 1
                resolution = await controller.handle(report)
                print(format_deliberation(resolution.deliberation))
                print(format_plan(resolution.plan))
                if args.json:
                    print(resolution.plan.model_dump_json(indent=2))
                print()
        return 0

    return asyncio.run(run())


def memory(args) -> int:
    """What the experience memory holds, and which past cases it finds for given incidents."""
    from .memory import ExperienceStore, format_neighbours

    if not (args.store / "store.json").exists():
        print(f"no experience memory at {args.store}. Fill one first, e.g.:\n"
              "  python -m oncall remember --data data/synthetic/dev100 --model models/logreg100.json "
              "--calibration models/conformal.json --out models/memory")
        return 1
    store = ExperienceStore.load(args.store)
    s = store.summary()
    print(f"{args.store}: {s['experiences']} experiences; plans {s['plans']}; feedback from {s['sources']}; "
          f"verified owners {s['owners']}")
    for key, c in s["fixes"].items():
        known = c.get("right", 0) + c.get("wrong", 0)
        rate = f"{c.get('right', 0) / known:.0%} right of {known} checked" if known else "none checked"
        print(f"  {key:<26} fixes {sum(c.values()):>4}: {rate}, {c.get('unknown', 0)} unchecked")
    if args.incident:
        if not args.incidents or not args.incidents.exists():
            print("--incident needs --incidents FILE (the incidents file it is in)")
            return 1
        wanted = args.incident.split(",")
        for report in load_incidents(args.incidents):
            if report.incident_id in wanted:
                print(f"\n{report.incident_id}  {report.title}: the {args.k} most similar past incidents")
                print(format_neighbours(store.similar(report, k=args.k, before=report.opened_at if args.past_only
                                                      else None)) or "  (none)")
    return 0


def classify(path: Path, model: Path | None = None) -> None:
    if model:
        from .learned import LogRegClassifier

        classifier = LogRegClassifier.load(model)
    else:
        classifier = RuleClassifier()
    for report in load_incidents(path):
        scores = classifier.predict(extract(report))
        team = choose_team(scores)
        probs = "  ".join(f"{d}={p:.2f}" for d, p in scores.ranked())
        print(f"{scores.incident_id}  [{scores.method}]  {probs}")
        print(f"  team={team.team}  ({team.reason})")
        print(f"  evidence={scores.evidence}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m controller")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("intake", help="print the features extracted from each incident")
    p.add_argument("incidents", type=Path, nargs="?", default=Path("lab/incidents.jsonl"))
    p.add_argument("--text", action="store_true", help="also print the masked text")
    p = sub.add_parser("classify", help="print domain probabilities and the chosen team")
    p.add_argument("incidents", type=Path, nargs="?", default=Path("lab/incidents.jsonl"))
    p.add_argument("--model", type=Path, default=None, help="trained logreg model (default: rules)")
    p = sub.add_parser("schema", help="write the JSON Schemas of the output contract")
    p.add_argument("--out", type=Path, default=Path("docs/contract"))
    p = sub.add_parser("agents", help="list the agent pool; optionally show eligible agents per incident")
    p.add_argument("--cards", type=Path, default=Path("agents/cards"))
    p.add_argument("--export", type=Path, default=None, help="write the default pool's cards to this folder")
    p.add_argument("--incidents", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--down", nargs="*", default=[], help="simulate these agents being down")
    p = sub.add_parser("route", help="route each incident to a primary agent, consults and fallbacks")
    p.add_argument("incidents", type=Path, nargs="?", default=Path("lab/incidents.jsonl"))
    p.add_argument("--model", type=Path, default=None, help="trained logreg model (default: rules)")
    p.add_argument("--calibration", type=Path, default=None, help="conformal calibration (default: fixed thresholds)")
    p.add_argument("--cards", type=Path, default=Path("agents/cards"))
    p.add_argument("--down", nargs="*", default=[], help="simulate these agents being down")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--why", action="store_true", help="print the reasons behind each decision")
    p = sub.add_parser("deliberate", help="route, ask the agents over A2A, cross-check, optimize, policy -> plan")
    p.add_argument("incidents", type=Path, nargs="?", default=Path("lab/incidents.jsonl"))
    p.add_argument("--incident", default=None, help="comma-separated incident ids")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--model", type=Path, default=None, help="trained logreg model (default: rules)")
    p.add_argument("--calibration", type=Path, default=None, help="conformal calibration (default: fixed thresholds)")
    p.add_argument("--cards", type=Path, default=Path("agents/cards"))
    p.add_argument("--down", nargs="*", default=[], help="treat these agents as down")
    p.add_argument("--cross-check", choices=["always", "contested", "never"], default="always")
    p.add_argument("--no-pointers", action="store_true", help="do not follow agents' pointers to other domains")
    p.add_argument("--may-abstain", action="store_true",
                   help="an agent may answer 'not mine' without a fix (default: every chosen agent proposes one)")
    p.add_argument("--no-approval", action="store_true",
                   help="no 'a human approves the fix' option (earlier design)")
    p.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for one agent answer")
    p.add_argument("--json", action="store_true", help="also print the RemediationPlan JSON (what MCP receives)")
    p.add_argument("--memory", type=Path, default=None,
                   help="Block 6b: trust agents by their verified record in this experience memory (read only)")
    p.add_argument("--trust", choices=["local", "global"], default="local")
    p = sub.add_parser("memory", help="Block 6a: show the experience memory and the past cases it finds")
    p.add_argument("store", type=Path, nargs="?", default=Path("models/memory"))
    p.add_argument("--incidents", type=Path, default=None, help="incidents file holding --incident")
    p.add_argument("--incident", default=None, help="comma-separated incident ids to look up")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--past-only", action="store_true", help="only incidents opened before this one")
    args = parser.parse_args()

    if args.command == "agents":
        return agents(args)
    if args.command == "memory":
        return memory(args)

    if args.command == "schema":
        for path in write_schemas(args.out):
            print(f"wrote {path}")
        return 0

    if not args.incidents.exists():
        print(f"no incidents at {args.incidents} yet (the sandbox writes them during `python -m oncall validate`).")
        print(f"To try it on the hand-written samples:  python -m controller {args.command} samples/incidents.jsonl")
        return 1

    if args.command == "classify":
        classify(args.incidents, args.model)
        return 0
    if args.command == "route":
        return route(args)
    if args.command == "deliberate":
        return deliberate(args)

    for report in load_incidents(args.incidents):
        features = extract(report)
        print(f"{features.incident_id}  alerts={features.alerts}")
        print(f"  components={features.components}")
        print(f"  signals={json.dumps(features.signals)}")
        if args.text:
            print("  " + features.text.replace("\n", "\n  "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
