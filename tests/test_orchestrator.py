"""Block 5b: investigate in parallel, fall back, follow pointers, escalate, cross-check (locally and over A2A)."""
import asyncio
import unittest

import httpx

from agents.local import LocalLink, local_agents
from agents.reasoners import ReasonerFailed
from agents.reasoners.rules import RuleReasoner
from agents.tools import RecordedEvidence
from controller import DOMAINS
from controller.classifier import DomainScores
from controller.contracts import IncidentReport
from controller.evidence import EvidenceBundle
from controller.orchestrator import Orchestrator, OrchestratorPolicy, format_deliberation
from controller.plan import INTENTS, PlannedAction
from controller.pool import default_pool
from controller.proposal import Proposal
from controller.registry import AgentRegistry
from controller.router import Router
from oncall.synth import generate


class FixedClassifier:
    """Puts chosen probabilities on every incident, so a test decides which domains are on the team."""

    name = "fixed"

    def __init__(self, **probs: float):
        rest = (1.0 - sum(probs.values())) / max(1, len(DOMAINS) - len(probs))
        self.probs = {d: probs.get(d, rest) for d in DOMAINS}

    def predict(self, features) -> DomainScores:
        return DomainScores(incident_id=features.incident_id, method=self.name, owner_probs=self.probs,
                            involved_probs=self.probs)


class AlwaysClaims(RuleReasoner):
    """A careless agent: claims every incident with a fix that looks fine (like an over-eager LLM)."""

    def investigate(self, task, agent_id, profile, toolbox):
        queries = toolbox.try_call("active_queries") or []
        assert queries, "the baseline evidence always lists some sessions"
        return Proposal(incident_id=task.incident.incident_id, agent_id=agent_id, domain=task.domain,
                        verdict="owner", root_cause="a session is slowing the warehouse", confidence=0.95,
                        actions=[PlannedAction(intent="stop_session", target="warehouse",
                                               params={"pid": queries[0].pid},  # real pid, wrong culprit
                                               risk=INTENTS["stop_session"].risk, rationale="guess")],
                        reasoner="careless")


class Flaky:
    """A link where some agents always fail (transport errors), the rest answer normally."""

    def __init__(self, inner, broken: set[str]):
        self.inner, self.broken = inner, broken

    async def investigate(self, card, task):
        if card.name in self.broken:
            raise httpx.ConnectError("connection refused")
        return await self.inner.investigate(card, task)

    async def review(self, card, task):
        return await self.inner.review(card, task)


class OrchestratorCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        incidents, truths, _ = generate(per_fault=3, seed=21)
        cls.reports = {r["incident_id"]: IncidentReport.from_lab_record(r) for r in incidents}
        cls.truth = {t["incident_id"]: t for t in truths}
        cls.evidence = RecordedEvidence({r["incident_id"]: EvidenceBundle.model_validate(r["evidence"]["domain"])
                                         for r in incidents})

    def report(self, fault: str, complete: bool = True) -> IncidentReport:
        for incident_id, t in self.truth.items():
            if t["fault_id"] == fault and not (complete and any(n.startswith("missing:") for n in t["noise"])):
                return self.reports[incident_id]
        raise LookupError(fault)

    def deliberate(self, report, classifier, policy=None, link=None, agents=None):
        registry = AgentRegistry(clock=lambda: 0.0)
        for card in default_pool():
            registry.register(card)
        link = link or LocalLink(agents or local_agents(self.evidence))
        orchestrator = Orchestrator(Router(classifier, registry), link, policy)
        return asyncio.run(orchestrator.run(report)), registry


class RoundsTest(OrchestratorCase):
    def test_two_domains_investigate_and_the_other_supports_the_claim(self):
        d, _ = self.deliberate(self.report("D5"), FixedClassifier(infra=0.50, data=0.45))
        self.assertEqual(sorted(p.domain for p in d.proposals), ["data", "infra"])
        claim = d.claims()[0]
        self.assertEqual(claim.domain, "data")
        self.assertEqual(d.tally(claim)["supports"], 1)  # infra: "the CPU is used by database sessions"
        self.assertIn("round 2", format_deliberation(d))

    def test_a_failed_agent_is_replaced_by_its_fallback(self):
        inner = LocalLink(local_agents(self.evidence))
        d, registry = self.deliberate(self.report("D3"), FixedClassifier(data=0.97),
                               link=Flaky(inner, {"data-fast"}))
        self.assertEqual([c.ok for c in d.calls if c.kind == "investigate"][:2], [False, True])
        self.assertEqual(d.claims()[0].agent_id, "data-strong")
        self.assertEqual(registry.state("data-fast").failures, 1)  # the breaker counts it

    def test_an_llm_timeout_is_a_failed_call_not_an_unsure_answer(self):
        # the qwen run: data-fast timed out and answered "unsure 0.00", so no fallback was asked
        class TimesOut(RuleReasoner):
            def investigate(self, task, agent_id, profile, toolbox):
                raise ReasonerFailed("investigation with qwen2.5:3b failed: ReadTimeout: timed out")

        agents = local_agents(self.evidence)
        agents["data-fast"].reasoner = TimesOut()
        d, _ = self.deliberate(self.report("D3"), FixedClassifier(data=0.97), agents=agents)
        first = d.calls[0]
        self.assertEqual((first.agent_id, first.ok), ("data-fast", False))
        self.assertIn("ReadTimeout", first.detail)
        self.assertEqual(d.claims()[0].agent_id, "data-strong")
        self.assertIn("1 failed", format_deliberation(d))

    def test_a_pointer_brings_in_the_domain_the_router_missed(self):
        d, _ = self.deliberate(self.report("D5"), FixedClassifier(infra=0.97))
        self.assertTrue(any("pointed to data" in t for t in d.transfers), d.transfers)
        self.assertEqual(d.claims()[0].domain, "data")

    def test_nobody_claims_so_a_stronger_agent_is_asked(self):
        # D1 (renamed field) needs the source-sample tool, which data-fast's 2-tool budget does not reach
        report = self.report("D1")
        d, _ = self.deliberate(report, FixedClassifier(data=0.97))
        first = d.proposals[0]
        self.assertEqual((first.agent_id, first.verdict), ("data-fast", "not_mine"))
        self.assertTrue(d.escalations)
        self.assertEqual(d.claims()[0].agent_id, "data-strong")
        no_escalation, _ = self.deliberate(report, FixedClassifier(data=0.97), OrchestratorPolicy(escalate=False))
        self.assertEqual(no_escalation.claims(), [])

    def test_cross_check_contradicts_a_careless_claim(self):
        agents = local_agents(self.evidence)
        careless = agents["data-fast"]
        careless.reasoner = AlwaysClaims()
        d, _ = self.deliberate(self.report("I1"), FixedClassifier(infra=0.50, data=0.45), agents=agents)
        by_agent = {c.agent_id: c for c in d.claims()}
        self.assertIn("data-fast", by_agent)  # the false claim is made...
        self.assertGreaterEqual(d.tally(by_agent["data-fast"])["contradicts"], 1)  # ...and challenged
        self.assertEqual(d.tally(by_agent["infra-fast"])["contradicts"], 1)  # the careless agent objects too

    def test_every_chosen_agent_must_hand_in_a_fix(self):
        # I1: infra has the real cause; data was chosen too (0.45), so it must give its best guess
        d, _ = self.deliberate(self.report("I1"), FixedClassifier(infra=0.50, data=0.45))
        by_domain = {p.domain: p for p in d.proposals}
        guess = by_domain["data"]
        self.assertNotEqual(guess.verdict, "owner")  # it still says how weak its evidence is
        self.assertTrue(guess.actions, guess.findings)  # ...but hands in a fix
        self.assertIn("best guess", guess.root_cause)
        self.assertIn(guess, d.suggestions())
        self.assertNotIn(guess, d.claims())
        self.assertGreaterEqual(d.tally(guess)["contradicts"], 1)  # infra's evidence objects to the guess
        self.assertIn("best guess (not_mine) by data-fast", format_deliberation(d))

    def test_agents_may_abstain_in_the_earlier_design(self):
        d, _ = self.deliberate(self.report("I1"), FixedClassifier(infra=0.50, data=0.45),
                               OrchestratorPolicy(must_propose=False))
        self.assertEqual(d.suggestions(), d.claims())
        self.assertEqual([p.actions for p in d.proposals if p.domain == "data"], [[]])

    def test_policy_never_skips_the_cross_check(self):
        d, _ = self.deliberate(self.report("D5"), FixedClassifier(infra=0.50, data=0.45), OrchestratorPolicy(cross_check="never"))
        self.assertEqual(d.reviews, [])
        self.assertTrue(all(c.kind == "investigate" for c in d.calls))


class OverA2ATest(OrchestratorCase):
    def test_the_rounds_work_over_real_a2a(self):
        try:
            from agents.server import build_app, port_of
            from controller.orchestrator import A2ALink, probe_agents
        except ImportError:
            self.skipTest("a2a-sdk is not installed")
        agents = local_agents(self.evidence)
        apps = {port_of(a): build_app(a) for a in agents.values()}

        async def by_port(scope, receive, send):  # one in-process "network" with every agent on its own port
            app = apps.get(scope["server"][1])
            if app is None:  # nothing listens there (the policy agent is not served yet)
                await send({"type": "http.response.start", "status": 404, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            await app(scope, receive, send)

        async def run():
            registry = AgentRegistry(clock=lambda: 0.0)
            for card in default_pool():
                registry.register(card)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=by_port), timeout=30) as http:
                alive = await probe_agents(registry, http, now=0.0)
                router = Router(FixedClassifier(infra=0.50, data=0.45), registry)
                d = await Orchestrator(router, A2ALink(http)).run(self.report("D5"), now=0.0)
            return alive, d

        alive, d = asyncio.run(run())
        self.assertFalse(alive["policy"])  # not served yet (Block 5c), so it is marked down
        self.assertTrue(alive["data-strong"])
        local, _ = self.deliberate(self.report("D5"), FixedClassifier(infra=0.50, data=0.45))
        self.assertEqual([p.model_dump() for p in d.proposals], [p.model_dump() for p in local.proposals])
        self.assertEqual([r.model_dump() for r in d.reviews], [r.model_dump() for r in local.reviews])


if __name__ == "__main__":
    unittest.main()
