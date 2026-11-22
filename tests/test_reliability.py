"""Block 6b: trust each agent by its verified record (counts -> numbers, router, optimizer, orchestration)."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from agents.local import LocalLink, local_agents
from agents.reasoners.shy import ShyReasoner
from controller.intake import extract
from controller.memory import Experience, ExperienceStore, ProposalRecord, ReviewRecord
from controller.optimizer import Optimizer
from controller.orchestrator import Orchestrator
from controller.pipeline import Controller
from controller.pool import default_pool
from controller.registry import AgentRegistry
from controller.reliability import ReliabilityModel, count
from controller.router import Router
from controller.trust import LLR_CAP, AgentRecord, Counts, TrustTable
from oncall.memory_eval import build_memory
from oncall.synth import generate
from oncall.synth.generate import write_dataset
from oncall.team_eval import evaluate_team
from tests.test_memory import LOCK, guess, remembered, report
from tests.test_optimizer import TIERS, act, deliberation
from tests.test_orchestrator import FixedClassifier, OrchestratorCase


def record(agent: str, domain: str, verdict: str, label, owned: bool) -> ProposalRecord:
    return ProposalRecord(agent_id=agent, domain=domain, tier=TIERS.get(agent, "fast"), verdict=verdict,
                          confidence=0.5, actions=[act("stop_session", pid=1)] if label else [], label=label,
                          domain_owned=owned)


def experience(i: int, proposals, reviews=()) -> Experience:
    e = remembered(f"e{i}", LOCK.format(i), f"2026-11-01T10:{i % 60:02d}:00+00:00")
    e.proposals, e.reviews = list(proposals), list(reviews)
    return e


def pool_registry() -> AgentRegistry:
    registry = AgentRegistry(clock=lambda: 0.0)
    for card in default_pool():
        registry.register(card)
    return registry


class TrustMathTest(unittest.TestCase):
    def test_no_record_keeps_the_planning_guess(self):
        t = TrustTable()
        self.assertEqual(t.success("new-agent", 0.7), 0.7)
        self.assertEqual(t.fix("new-agent", "owner", 0.62), 0.62)
        self.assertEqual(t.verdict_llr("new-agent", "owner", 1.8), 1.8)
        self.assertEqual(t.review_llr("new-agent", True, "supports", 0.8), 0.8)

    def test_many_cases_move_it_to_what_was_measured(self):
        t = TrustTable(mode="global", agents={"a": AgentRecord(success=Counts(hits=4, n=100))})
        self.assertAlmostEqual(t.success("a", 0.7), (4 + 4 * 0.7) / 104)  # ~0.07, not the guessed 0.70
        few = TrustTable(mode="global", agents={"a": AgentRecord(success=Counts(hits=0, n=2))})
        self.assertGreater(few.success("a", 0.7), 0.4)  # two cases are not enough to give up on it

    def test_incidents_like_this_one_count_more_than_the_whole_memory(self):
        # good on the whole, but it failed every time on incidents like this one
        agents = {"a": AgentRecord(success=Counts(hits=90, n=100))}
        near = {"a": AgentRecord(success=Counts(hits=0, n=10))}
        local = TrustTable(mode="local", agents=agents, near=near)
        globally = TrustTable(mode="global", agents=agents, near=near)
        self.assertLess(local.success("a", 0.7), 0.2)
        self.assertGreater(globally.success("a", 0.7), 0.85)

    def test_a_verdict_is_evidence_only_if_it_differs_between_owned_and_not_owned(self):
        honest = AgentRecord(verdict_owned={"owner": 95, "not_mine": 5}, verdict_not_owned={"owner": 2, "not_mine": 98})
        shy = AgentRecord(verdict_owned={"not_mine": 100}, verdict_not_owned={"not_mine": 100})
        t = TrustTable(agents={"honest": honest, "shy": shy})
        self.assertGreater(t.verdict_llr("honest", "owner", 1.0), 2.5)
        self.assertLess(t.verdict_llr("honest", "not_mine", -0.3), -2.5)
        self.assertAlmostEqual(t.verdict_llr("shy", "not_mine", -0.3), 0.0, places=1)  # says it every time
        lopsided = TrustTable(agents={"x": AgentRecord(verdict_owned={"owner": 10000},
                                                       verdict_not_owned={"not_mine": 10000})})
        self.assertEqual(lopsided.verdict_llr("x", "owner", 1.0), LLR_CAP)
        one_sided = TrustTable(agents={"x": AgentRecord(verdict_owned={"owner": 50})})
        self.assertEqual(one_sided.verdict_llr("x", "owner", 1.0), 1.0)  # never seen "not owned": no comparison


class CountTest(unittest.TestCase):
    def test_what_is_counted(self):
        claim_ = ReviewRecord(reviewer="infra-fast", domain="infra", claim_agent="data-fast", claim_domain="data",
                              stance="supports", claim_label="right", claim_owned=True)
        same = ReviewRecord(reviewer="data-strong", domain="data", claim_agent="data-fast", claim_domain="data",
                            stance="contradicts", claim_label="wrong", claim_owned=True)
        unchecked = ReviewRecord(reviewer="data-strong", domain="data", claim_agent="data-fast", claim_domain="data",
                                 stance="supports", claim_label="unknown", claim_owned=True)
        records = count([
            experience(1, [record("data-fast", "data", "owner", "right", True)], [claim_]),
            experience(2, [record("data-fast", "data", "not_mine", "wrong", True)], [same, unchecked]),
            experience(3, [record("data-fast", "data", "not_mine", None, True)]),  # no fix in the owner's domain
            experience(4, [record("data-fast", "data", "not_mine", "unknown", True)]),  # never checked: left out
            experience(5, [record("data-fast", "data", "not_mine", "wrong", False)]),  # not its domain's incident
        ])
        r = records["data-fast"]
        self.assertEqual((r.success.hits, r.success.n), (1, 3))
        self.assertEqual({v: (c.hits, c.n) for v, c in r.fix.items()}, {"owner": (1, 1), "not_mine": (0, 1)})
        self.assertEqual(r.verdict_owned, {"owner": 1, "not_mine": 3})
        self.assertEqual(r.verdict_not_owned, {"not_mine": 1})
        self.assertEqual(records["infra-fast"].review_true, {"other:supports": 1})
        self.assertEqual(records["data-strong"].review_false, {"same:contradicts": 1})
        self.assertNotIn("same:supports", records["data-strong"].review_true)

    def test_never_learns_from_the_incident_it_is_deciding(self):
        store = ExperienceStore()
        store.add(experience(1, [record("data-fast", "data", "owner", "right", True)]))
        store.add(experience(2, [record("data-fast", "data", "owner", "wrong", True)]))
        model = ReliabilityModel(store, mode="global")
        table = model.table(extract(report("e1", LOCK.format(1))))
        self.assertEqual(table.agents["data-fast"].success.hits, 0)  # e1 (the right one) is left out

    def test_a_changed_memory_is_counted_again(self):
        store = ExperienceStore()
        model = ReliabilityModel(store, mode="global")
        query = extract(report("q", LOCK.format(9)))
        self.assertEqual(model.table(query).agents, {})
        store.add(experience(1, [record("data-fast", "data", "owner", "right", True)]))
        self.assertEqual(model.table(query).agents["data-fast"].success.n, 1)


class Stub:
    def __init__(self, table):
        self.t = table

    def table(self, features):
        return self.t


class RoutingWithTrustTest(OrchestratorCase):
    def router(self, table=None) -> Router:
        return Router(FixedClassifier(data=0.97), pool_registry(), trust_model=Stub(table) if table is not None else None)

    def test_an_agent_that_keeps_failing_is_no_longer_sent_first(self):
        incident = self.report("D1")
        self.assertEqual(self.router().route(incident).primary.agent_id, "data-fast")  # cheap, planning guess
        table = TrustTable(mode="global", agents={"data-fast": AgentRecord(success=Counts(hits=3, n=100)),
                                                  "data-strong": AgentRecord(success=Counts(hits=99, n=100))})
        decision = self.router(table).route(incident)
        self.assertEqual(decision.primary.agent_id, "data-strong")
        self.assertIs(decision.trust, table)  # the later rounds read the same numbers

    def test_with_a_record_nobody_is_doubted_just_for_not_saying_owner(self):
        agents = local_agents(self.evidence)
        for agent_id, agent in agents.items():
            if agent_id != "generalist":
                agent.reasoner = ShyReasoner()
        incident = self.report("D3")
        for table, escalations in ((None, 1), (TrustTable(mode="global"), 0)):
            router = Router(FixedClassifier(data=0.97), pool_registry(), trust_model=Stub(table) if table is not None else None)
            d = asyncio.run(Orchestrator(router, LocalLink(agents)).run(incident))
            self.assertEqual(len(d.escalations), escalations, table)


class OptimizerWithTrustTest(unittest.TestCase):
    def setUp(self):
        self.fix = act("stop_session", pid=6079)
        self.d = deliberation({"data": 0.9, "infra": 0.1}, proposals=[guess("data-fast", "data", self.fix)])
        self.opt = Optimizer(TIERS)

    def belief(self, table) -> float:
        self.d.routing.trust = table
        return self.opt.hypotheses(self.d)[0].belief

    def test_without_a_record_the_planning_numbers_are_used_exactly(self):
        self.assertEqual(self.belief(None), self.belief(TrustTable()))

    def test_a_guess_that_was_always_right_is_believed(self):
        planning = self.belief(None)
        shy_but_right = TrustTable(mode="global", agents={"data-fast": AgentRecord(
            fix={"not_mine": Counts(hits=100, n=100)}, verdict_owned={"not_mine": 100},
            verdict_not_owned={"not_mine": 100})})
        self.assertGreater(self.belief(shy_but_right), planning + 0.4)
        always_wrong = TrustTable(mode="global", agents={"data-fast": AgentRecord(
            fix={"not_mine": Counts(hits=0, n=100)})})
        self.assertLess(self.belief(always_wrong), planning / 3)

    def test_the_plan_says_why_it_trusts_whom(self):
        self.d.routing.trust = TrustTable(mode="global", agents={"data-fast": AgentRecord(
            success=Counts(hits=5, n=10), fix={"not_mine": Counts(hits=5, n=10)})})
        why = " ".join(self.opt.hypotheses(self.d)[0].why)
        self.assertIn("10 verified cases", why)


class LearningTest(unittest.TestCase):
    """The whole loop: a memory from one split makes the controller cheaper on another, and learning as it goes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        incidents, truths, manifest = generate(per_fault=5, seed=43)
        cls.data = Path(cls.tmp.name) / "data"
        write_dataset(cls.data, incidents, truths, manifest)
        store, _ = build_memory(cls.data, split="train")
        cls.memory = store.save(Path(cls.tmp.name) / "memory")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_verified_record_saves_agent_calls(self):
        planning = evaluate_team(self.data, split="val")
        learned = evaluate_team(self.data, split="val", memory=self.memory)
        self.assertEqual(learned["memory"]["experiences"], 42)
        self.assertLess(learned["cost_per_incident"], planning["cost_per_incident"])
        self.assertLessEqual(learned["loss"]["plan"], planning["loss"]["plan"] + 0.5)

    def test_the_memory_may_not_hold_the_incidents_being_scored(self):
        with self.assertRaises(ValueError):
            evaluate_team(self.data, split="train", memory=self.memory)

    def test_learning_as_it_goes_reports_a_curve(self):
        _, r = build_memory(self.data, split="train", trust="local", windows=3)
        self.assertEqual([(c["from"], c["to"]) for c in r["curve"]], [(1, 14), (15, 28), (29, 42)])
        self.assertEqual(r["labels"]["right"].get("truly wrong", 0), 0)  # reading the memory never corrupts labels

    def test_the_controller_refuses_an_unknown_trust_mode(self):
        with self.assertRaises(ValueError):
            Controller(Router(FixedClassifier(data=0.9), AgentRegistry()), None, trust="sometimes")


if __name__ == "__main__":
    unittest.main()
