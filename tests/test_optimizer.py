"""Block 5c: beliefs, staged plans, escalation, the policy agent, the second look, and plans over A2A."""
import asyncio
import unittest

import httpx
from pydantic import ValidationError

from agents.local import LocalLink, local_agents
from agents.policy_agent import PolicyAgent
from agents.reasoners.careless import CarelessReasoner
from agents.tools import RecordedEvidence
from controller import DOMAINS
from controller.contracts import AlertIn, IncidentReport
from controller.evidence import EvidenceBundle
from controller.optimizer import Optimizer, OptimizerPolicy
from controller.orchestrator import Deliberation
from controller.pipeline import Controller
from controller.plan import INTENTS, PlannedAction, RemediationPlan
from controller.policy import PolicyRequest, check_plan
from controller.pool import default_pool
from controller.proposal import Proposal, Review
from controller.registry import AgentRegistry
from controller.router import Router, RoutingDecision
from oncall.synth import generate
from tests.test_orchestrator import FixedClassifier

TIERS = {"data-fast": "fast", "data-strong": "strong", "infra-fast": "fast", "infra-strong": "strong",
         "security-strong": "strong", "generalist": "generalist"}


def act(intent: str, target: str = "warehouse", **params) -> PlannedAction:
    return PlannedAction(intent=intent, target=target, params=params, risk=INTENTS[intent].risk, rationale="t")


def claim(agent: str, domain: str, *actions, confidence: float = 0.8) -> Proposal:
    return Proposal(incident_id="i1", agent_id=agent, domain=domain, verdict="owner", root_cause=f"{agent}'s cause",
                    confidence=confidence, actions=list(actions))


def not_mine(agent: str, domain: str) -> Proposal:
    return Proposal(incident_id="i1", agent_id=agent, domain=domain, verdict="not_mine", confidence=0.7)


def review(reviewer: str, domain: str, of: Proposal, stance: str, confidence: float = 0.8) -> Review:
    return Review(incident_id="i1", reviewer=reviewer, domain=domain, claim_agent=of.agent_id,
                  claim_domain=of.domain, stance=stance, confidence=confidence)


def deliberation(probs: dict, priority: str = "P3", proposals=(), reviews=()) -> Deliberation:
    owner_probs = {d: probs.get(d, 0.0) for d in DOMAINS}
    team = [d for d, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
    routing = RoutingDecision(incident_id="i1", classifier="fixed", calibrated=False, priority=priority,
                              owner_probs=owner_probs, team=team, confident=True)
    return Deliberation(incident_id="i1", routing=routing, proposals=list(proposals), reviews=list(reviews))


class BeliefTest(unittest.TestCase):
    opt = Optimizer(TIERS)

    def test_a_strong_claim_in_a_likely_domain_is_acted_on_in_one_stage(self):
        plan = self.opt.plan(deliberation({"data": 0.95}, proposals=[claim("data-strong", "data", act(
            "stop_session", pid=7))]))
        self.assertEqual((plan.kind, [a.stage for a in plan.actions]), ("remediate", [1]))
        self.assertGreater(plan.confidence, 0.8)
        self.assertTrue(plan.options[1].chosen and not plan.options[0].chosen)

    def test_no_claim_means_escalate(self):
        plan = self.opt.plan(deliberation({"data": 0.9}, proposals=[not_mine("data-fast", "data")]))
        self.assertEqual((plan.kind, plan.actions), ("escalate", []))
        self.assertIn("no agent found", plan.root_cause)

    def test_self_reported_confidence_is_ignored(self):
        loud = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[claim(
            "data-fast", "data", act("stop_session", pid=7), confidence=0.99)]))[0]
        quiet = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[claim(
            "data-fast", "data", act("stop_session", pid=7), confidence=0.55)]))[0]
        self.assertAlmostEqual(loud.belief, quiet.belief)

    def test_agreeing_agents_merge_into_one_stronger_hypothesis(self):
        one = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[
            claim("data-fast", "data", act("stop_session", pid=7))]))
        two = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[
            claim("data-fast", "data", act("stop_session", pid=7)),
            claim("data-strong", "data", act("stop_session", pids=[7]))]))
        self.assertEqual((len(one), len(two)), (1, 1))
        self.assertGreater(two[0].p_fix, one[0].p_fix)

    def test_another_domains_contradiction_lowers_the_domain(self):
        c = claim("data-fast", "data", act("stop_session", pid=7))
        base = self.opt.hypotheses(deliberation({"data": 0.5, "infra": 0.5}, proposals=[c]))[0]
        hit = self.opt.hypotheses(deliberation({"data": 0.5, "infra": 0.5}, proposals=[c],
                                               reviews=[review("infra-fast", "infra", c, "contradicts")]))[0]
        self.assertLess(hit.p_domain, base.p_domain)

    def test_a_high_risk_fix_never_runs_alone(self):
        props = [claim("data-fast", "data", act("map_schema_fields", "ingest", mapping={"a": "amount"}))]
        for priority in ("P3", "P1"):
            plan = self.opt.plan(deliberation({"data": 0.6}, priority, props))
            self.assertEqual(plan.kind, "approve")  # a human checks it; cheaper than a human from zero
            self.assertNotIn("fix data (data-fast)", [o.label for o in plan.options])  # no automatic option
            self.assertTrue(any("map_schema_fields is high risk" in e for e in plan.evidence))
        # the earlier design: priced as if it ran alone, so the priority decided between running and a human
        old = Optimizer(TIERS, OptimizerPolicy(approval=False))
        self.assertEqual(old.plan(deliberation({"data": 0.6}, "P3", props)).kind, "escalate")
        self.assertEqual(old.plan(deliberation({"data": 0.6}, "P1", props)).kind, "remediate")

    def test_a_confident_high_risk_fix_goes_to_approval_not_to_a_human_from_zero(self):
        # the qwen + llama3 run, syn-7-00046: data-strong's correct mapping, belief 0.94, P3 -> was escalated
        fix = act("map_schema_fields", "ingest", mapping={"total_amount": "amount"})
        plan = self.opt.plan(deliberation({"data": 0.96}, "P3", [claim("data-strong", "data", fix)],
                                          reviews=[]))
        self.assertEqual((plan.kind, plan.actions[0].intent), ("approve", "map_schema_fields"))
        self.assertLess(plan.expected_loss, self.opt.human_cost("P3"))

    def test_a_weak_guess_is_not_worth_a_humans_check(self):
        guess = Proposal(incident_id="i1", agent_id="data-fast", domain="data", verdict="not_mine",
                         root_cause="best guess", confidence=0.5,
                         actions=[act("map_schema_fields", "ingest", mapping={"a": "amount"})])
        self.assertEqual(self.opt.plan(deliberation({"data": 0.5, "infra": 0.5}, "P3", [guess])).kind, "escalate")

    def test_a_safe_confident_fix_still_runs_alone(self):
        plan = self.opt.plan(deliberation({"data": 0.95}, proposals=[claim("data-strong", "data", act(
            "stop_session", pid=7))]))
        self.assertEqual(plan.kind, "remediate")
        self.assertFalse(any(o.label.startswith("a human approves") for o in plan.options))

    def test_a_best_guess_counts_but_less_than_a_claim(self):
        fix = act("stop_session", pid=7)
        guess = Proposal(incident_id="i1", agent_id="data-strong", domain="data", verdict="not_mine",
                         root_cause="nothing clear; best guess", confidence=0.7, actions=[fix])
        g = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[guess]))
        c = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[claim("data-strong", "data", fix)]))
        self.assertEqual(len(g), 1)  # the guess is a hypothesis the optimizer can use...
        self.assertLess(g[0].belief, c[0].belief)  # ...worth less than the same fix claimed
        self.assertIn("guess: not_mine", g[0].why[1])

    def test_a_guess_that_agrees_with_a_claim_adds_to_it_under_the_claims_label(self):
        fix = act("stop_session", pid=7)
        guess = Proposal(incident_id="i1", agent_id="data-fast", domain="data", verdict="unsure",
                         root_cause="best guess", confidence=0.3, actions=[fix])
        alone = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[claim("data-strong", "data", fix)]))[0]
        both = self.opt.hypotheses(deliberation({"data": 0.9}, proposals=[guess, claim("data-strong", "data", fix)]))
        self.assertEqual(len(both), 1)
        self.assertGreater(both[0].p_fix, alone.p_fix)
        self.assertEqual(both[0].label, "data: data-strong's cause")

    def test_two_rival_causes_become_stages_cheapest_bet_first(self):
        bad_rows = claim("data-strong", "data", act("quarantine_records"))
        config = claim("infra-strong", "infra", act("restore_config", "transform"))
        plan = self.opt.plan(deliberation({"data": 0.5, "infra": 0.5}, "P2", [config, bad_rows]))
        self.assertEqual(plan.kind, "remediate")  # both reversible, nothing high risk: may run alone
        self.assertEqual([(a.stage, a.intent) for a in plan.actions], [(1, "quarantine_records"), (2, "restore_config")])
        RemediationPlan.model_validate_json(plan.model_dump_json())  # a valid contract document

    def test_unsure_rivals_in_a_p1_go_to_a_human_likeliest_first(self):
        lock = claim("data-strong", "data", act("stop_session", pid=7))
        hog = claim("infra-strong", "infra", act("restore_config", "transform"))
        plan = self.opt.plan(deliberation({"data": 0.5, "infra": 0.5}, "P1", [lock, hog]))
        self.assertEqual(plan.kind, "approve")  # P1 and only 44% sure: the policy wants a human's yes
        self.assertEqual([(a.stage, a.intent) for a in plan.actions], [(1, "stop_session"), (2, "restore_config")])
        self.assertLess(plan.expected_loss, self.opt.human_cost("P1"))
        RemediationPlan.model_validate_json(plan.model_dump_json())


class PolicyTest(unittest.TestCase):
    def plan(self, *actions, confidence=0.9, kind="remediate"):
        return RemediationPlan(incident_id="i1", kind=kind, owner="data", team=["data"], root_cause="x",
                               confidence=confidence, actions=list(actions))

    def test_safe_confident_plan_runs_automatically(self):
        d = check_plan(self.plan(act("restore_config", "metrics-api")))
        self.assertEqual((d.approved, d.needs_human), (True, False))

    def test_high_risk_or_irreversible_and_unsure_needs_a_human(self):
        self.assertTrue(check_plan(self.plan(act("rotate_credential", role="etl"))).needs_human)
        self.assertTrue(check_plan(self.plan(act("stop_session", pid=7), confidence=0.6)).needs_human)
        self.assertFalse(check_plan(self.plan(act("stop_session", pid=7), confidence=0.9)).needs_human)

    def test_p1_with_an_unsure_plan_needs_a_human(self):
        self.assertTrue(check_plan(self.plan(act("restore_config", "metrics-api"), confidence=0.5), "P1").needs_human)

    def test_non_executable_plans_are_refused_and_escalations_never_run(self):
        bad = self.plan(act("stop_session", "prometheus", pid=7))
        self.assertFalse(check_plan(bad).approved)
        esc = self.plan(confidence=0.0, kind="escalate")
        self.assertEqual((check_plan(esc).approved, check_plan(esc).needs_human), (False, True))

    def test_an_approval_plan_may_run_once_a_human_says_yes(self):
        d = check_plan(self.plan(act("map_schema_fields", "ingest", mapping={"a": "amount"}), kind="approve"))
        self.assertEqual((d.approved, d.needs_human), (True, True))
        self.assertIn("a human approves each stage before it runs", d.reasons)
        bad = check_plan(self.plan(act("stop_session", "prometheus", pid=7), kind="approve"))
        self.assertFalse(bad.approved)  # hard rules hold for approval plans too


class RealizedLossTest(unittest.TestCase):
    """How the evaluation scores what the executor would really do with each kind of plan."""
    opt = Optimizer(TIERS)
    TRUTH = {"valid_intents": ["map_schema_fields"], "fix_targets": {"map_schema_fields": [{"mapping": ["total_amount"]}]}}
    SEEN = "total_amount 7"
    RIGHT = act("map_schema_fields", "ingest", mapping={"total_amount": "amount"})
    WRONG = act("stop_session", pid=7)

    def test_under_approval_a_wrong_stage_never_runs(self):
        from oncall.team_eval import realized_loss

        approved = realized_loss([[self.WRONG], [self.RIGHT]], self.TRUTH, self.SEEN, self.opt, "P3", approve=True)
        # two fixes to check, then only the right one runs, with a fifth of its risk
        expected = self.opt.approval_cost("P3", 2) + 0.2 * self.opt.risk_cost([self.RIGHT]) + 0.5
        self.assertAlmostEqual(approved, expected)
        alone = realized_loss([[self.WRONG], [self.RIGHT]], self.TRUTH, self.SEEN, self.opt, "P3")
        self.assertGreater(alone, approved)  # alone, the wrong session is killed first
        nothing_right = realized_loss([[self.WRONG]], self.TRUTH, self.SEEN, self.opt, "P3", approve=True)
        self.assertAlmostEqual(nothing_right, self.opt.approval_cost("P3", 1) + self.opt.human_cost("P3"))

    def test_the_oracle_follows_the_same_rules(self):
        from oncall.team_eval import oracle_loss

        # a correct high-risk fix may not run alone, even knowing the truth: the best is to have it approved
        self.assertAlmostEqual(oracle_loss([self.RIGHT], self.TRUTH, self.SEEN, self.opt, "P3"),
                               self.opt.approval_cost("P3", 1) + 0.2 * 5.0 + 0.5)
        old = Optimizer(TIERS, OptimizerPolicy(approval=False))
        self.assertAlmostEqual(oracle_loss([self.RIGHT], self.TRUTH, self.SEEN, old, "P3"), 5.5)


class ContractTest(unittest.TestCase):
    def test_stage_rules(self):
        with self.assertRaises(ValidationError):  # gap: stage 2 without stage 1
            RemediationPlan(incident_id="i", owner="data", team=["data"], root_cause="x", confidence=0.5,
                            actions=[act("stop_session", pid=1).model_copy(update={"stage": 2})])
        with self.assertRaises(ValidationError):
            RemediationPlan(incident_id="i", kind="escalate", owner="data", team=["data"], root_cause="x",
                            confidence=0.5, actions=[act("stop_session", pid=1)])
        with self.assertRaises(ValidationError):
            RemediationPlan(incident_id="i", owner="data", team=["data"], root_cause="x", confidence=0.5)

    def test_reversibility_matches_the_sandbox(self):
        from oncall.actions import ACTIONS
        from oncall.intents import INTENT_ACTIONS

        for intent, actions in INTENT_ACTIONS.items():
            with self.subTest(intent=intent):
                self.assertEqual(INTENTS[intent].reversible, all(ACTIONS[a].reversible for a in actions))


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        incidents, truths, _ = generate(per_fault=3, seed=31)
        cls.reports = {r["incident_id"]: IncidentReport.from_lab_record(r) for r in incidents}
        cls.truth = {t["incident_id"]: t for t in truths}
        cls.evidence = RecordedEvidence({r["incident_id"]: EvidenceBundle.model_validate(r["evidence"]["domain"])
                                         for r in incidents})

    def report(self, fault):
        return next(self.reports[i] for i, t in self.truth.items() if t["fault_id"] == fault
                    and not any(n.startswith("missing:") for n in t["noise"]))

    def controller(self, classifier, agents=None, without=()):
        registry = AgentRegistry(clock=lambda: 0.0)
        for card in default_pool():
            if card.name not in without:
                registry.register(card)
        return Controller(Router(classifier, registry), LocalLink(agents or local_agents(self.evidence)))

    def test_a_careless_claim_triggers_a_second_look(self):
        agents = local_agents(self.evidence)
        agents["data-fast"].reasoner = CarelessReasoner()
        controller = self.controller(FixedClassifier(data=0.97), agents)
        res = asyncio.run(controller.handle(self.report("D1")))
        self.assertTrue(any("data-strong" in e for e in res.deliberation.escalations), res.deliberation.escalations)
        best = controller.optimizer.hypotheses(res.deliberation)[0]
        self.assertEqual(best.agents, ["data-strong"])  # the real fix (renamed field) now leads
        # two rival claims, a high-risk fix and a P3 incident: either the real fix goes first, or a human decides
        self.assertTrue(res.plan.kind == "escalate" or res.plan.actions[0].intent == "map_schema_fields")

    def test_without_a_policy_agent_nothing_is_approved(self):
        res = asyncio.run(self.controller(FixedClassifier(data=0.97), without={"policy"}).handle(self.report("D3")))
        self.assertEqual(res.plan.kind, "remediate")
        self.assertFalse(res.plan.policy.approved)
        self.assertIn("no policy agent", res.plan.policy.reasons[0])

    def test_the_policy_agent_answers_over_a2a(self):
        try:
            from agents.server import build_app
            from controller.wire import ask_policy
        except ImportError:
            self.skipTest("a2a-sdk is not installed")
        card = next(c for c in default_pool() if c.name == "policy")
        plan = RemediationPlan(incident_id="i1", owner="security", team=["security"], root_cause="x",
                               confidence=0.9, actions=[act("rotate_credential", role="etl")])

        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(PolicyAgent(card)))) as http:
                return await ask_policy(card, PolicyRequest(plan=plan, priority="P1"), http=http)

        decision = asyncio.run(run())
        self.assertEqual(decision, check_plan(plan, "P1"))
        self.assertTrue(decision.needs_human)


if __name__ == "__main__":
    unittest.main()
