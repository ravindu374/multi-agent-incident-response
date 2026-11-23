"""Block 6c: the history agent (lookup, remember, over A2A), past cases for the experts, precedent in the plan."""
import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from agents.history_agent import HistoryAgent
from agents.local import LocalLink, local_agents
from agents.reasoners.ollama import past_cases_text
from controller.history import HistoryQuery, HistoryReport, PastCase, RememberRequest, describe_fix, precedent
from controller.memory import ExperienceStore
from controller.optimizer import Optimizer, OptimizerPolicy
from controller.orchestrator import OrchestratorPolicy
from controller.pipeline import Controller
from controller.pool import default_pool
from controller.proposal import InvestigationTask
from controller.router import Router
from tests.test_memory import DISK, LOCK, guess, remembered, report
from tests.test_optimizer import TIERS, act, deliberation
from tests.test_orchestrator import FixedClassifier, OrchestratorCase
from tests.test_reliability import pool_registry

HISTORY_CARD = next(c for c in default_pool() if c.name == "history")


def case(owner: str, *fix, similarity: float = 0.97, incident_id: str = "past") -> PastCase:
    return PastCase(incident_id=incident_id, similarity=similarity, owner=owner, fix_that_worked=list(fix),
                    outcome="executor")


class PrecedentTest(unittest.TestCase):
    def test_counts_only_the_fix_domains_incidents_with_a_recorded_fix(self):
        cases = [case("data", act("stop_session", pid=1)), case("data", act("stop_session", pid=2)),
                 case("data", act("map_schema_fields", target="ingest", mapping={"a": "b"})),
                 case("infra", act("stop_process", pids=[3])),  # another domain: says nothing about data's fixes
                 case("data")]  # fixed by a person: no recorded fix
        same, other, llr = precedent(cases, "data", [act("stop_session", pid=99)])  # today's pid differs: same kind
        self.assertEqual((same, other), (2, 1))
        self.assertAlmostEqual(llr, 0.405, places=2)  # log(3 / 2)
        self.assertEqual(precedent(cases, "network", [act("reroute_connection", target="ingest", route="source",
                                                          endpoint="secondary")]), (0, 0, 0.0))

    def test_strong_precedent_is_capped(self):
        many = [case("data", act("stop_session", pid=i)) for i in range(100)]
        self.assertEqual(precedent(many, "data", [act("stop_session", pid=1)])[2], 3.0)
        self.assertEqual(precedent(many, "data", [act("quarantine_records", tests=["t"])])[2], -3.0)

    def test_past_fixes_are_shown_without_their_values(self):
        text = describe_fix([act("stop_session", pids=[6079, 5118])])
        self.assertEqual(text, ["stop_session on warehouse (pids)"])


class HistoryAgentTest(unittest.TestCase):
    def setUp(self):
        self.store = ExperienceStore()
        self.store.add(remembered("lock1", LOCK.format(1), "2026-11-01T10:00:00+00:00"))
        self.store.add(remembered("disk1", DISK.format(1), "2026-11-02T10:00:00+00:00", owner="infra"))

    def test_only_really_similar_incidents_come_back(self):
        found = HistoryAgent(HISTORY_CARD, self.store).lookup(HistoryQuery(incident=report("q", LOCK.format(7))))
        self.assertEqual([c.incident_id for c in found.cases], ["lock1"])  # the disk incident is not alike enough
        self.assertEqual(found.memory_size, 2)
        self.assertIsNotNone(found.trust)

    def test_modes(self):
        query = HistoryQuery(incident=report("q", LOCK.format(7)))
        self.assertIsNone(HistoryAgent(HISTORY_CARD, self.store, trust="off").lookup(query).trust)
        self.assertEqual(HistoryAgent(HISTORY_CARD, self.store, cases=False).lookup(query).cases, [])
        with self.assertRaises(ValueError):
            HistoryAgent(next(c for c in default_pool() if c.name == "policy"))

    def test_remembers_and_keeps_it_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            keeper = HistoryAgent(HISTORY_CARD, self.store, path=Path(tmp))
            new = remembered("lock2", LOCK.format(2), "2026-11-03T10:00:00+00:00")
            ack = keeper.remember(RememberRequest(experience=new.model_dump(mode="json")))
            self.assertEqual((ack.incident_id, ack.memory_size), ("lock2", 3))
            self.assertEqual(len(ExperienceStore.load(Path(tmp))), 3)

    def test_it_answers_over_a2a(self):
        try:
            from agents.server import build_app
            from controller.wire import ask_history, ask_remember
        except ImportError:
            self.skipTest("a2a-sdk is not installed")
        keeper = HistoryAgent(HISTORY_CARD, self.store)
        new = remembered("lock2", LOCK.format(2), "2026-11-03T10:00:00+00:00")
        expected = keeper.lookup(HistoryQuery(incident=report("q", LOCK.format(7))))  # before lock2 is stored

        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(keeper))) as http:
                found = await ask_history(HISTORY_CARD, HistoryQuery(incident=report("q", LOCK.format(7))), http=http)
                ack = await ask_remember(HISTORY_CARD, RememberRequest(experience=new.model_dump(mode="json")), http=http)
                return found, ack

        found, ack = asyncio.run(run())
        self.assertEqual(found, expected)  # the same answer over A2A as in-process
        self.assertEqual((ack.stored, ack.memory_size, len(self.store)), (True, 3, 3))


class ExpertsSeePastCasesTest(OrchestratorCase):
    def fast_data_agent(self):
        return next(c for c in default_pool() if c.name == "data-fast"), local_agents(self.evidence)["data-fast"]

    def test_a_fast_agent_looks_where_the_past_fix_was_found(self):
        incident = self.report("D1")  # schema drift: the evidence is in source_sample, beyond a fast agent's budget
        card, agent = self.fast_data_agent()
        alone = agent.handle(InvestigationTask(incident=incident, domain="data", must_propose=True))
        self.assertEqual(alone.verdict, "not_mine")
        past = [case("data", act("map_schema_fields", target="ingest", mapping={"order_amount": "amount"}))]
        helped = agent.handle(InvestigationTask(incident=incident, domain="data", must_propose=True, past_cases=past))
        self.assertEqual(helped.verdict, "owner")
        self.assertEqual(helped.actions[0].intent, "map_schema_fields")
        self.assertNotEqual(helped.actions[0].params, {"mapping": {"order_amount": "amount"}},
                            "the mapping comes from today's evidence, not from the past case")
        self.assertIn("source_sample", helped.tools_used)

    def test_another_domains_past_fixes_change_nothing(self):
        incident = self.report("D1")
        _, agent = self.fast_data_agent()
        past = [case("infra", act("free_storage", target="ingest", mount="/spool"))]
        task = InvestigationTask(incident=incident, domain="data", must_propose=True, past_cases=past)
        self.assertEqual(agent.handle(task).verdict, "not_mine")

    def test_the_llm_prompt_shows_kinds_of_fix_not_old_ids(self):
        task = InvestigationTask(incident=self.report("D3"), domain="data",
                                 past_cases=[case("data", act("stop_session", pid=6079))])
        text = past_cases_text(task)
        self.assertIn("stop_session on warehouse (pid)", text)
        self.assertNotIn("6079", text)
        self.assertEqual(past_cases_text(InvestigationTask(incident=self.report("D3"), domain="data")), "")


class PrecedentInThePlanTest(unittest.TestCase):
    def setUp(self):
        self.lock, self.other = act("stop_session", pid=6079), act("restart_component")
        self.d = deliberation({"data": 0.9, "infra": 0.1}, proposals=[guess("data-fast", "data", self.lock)])

    def belief(self, cases, policy=None) -> float:
        self.d.history = HistoryReport(incident_id="i1", cases=cases) if cases is not None else None
        return Optimizer(TIERS, policy).hypotheses(self.d)[0].belief

    def test_what_fixed_similar_incidents_moves_the_fix_not_the_domain(self):
        none = self.belief(None)
        backed = self.belief([case("data", act("stop_session", pid=i)) for i in range(10)])
        against = self.belief([case("data", act("quarantine_records", tests=["t"])) for _ in range(10)])
        self.assertGreater(backed, none + 0.3)
        self.assertLess(against, none)
        elsewhere = self.belief([case("infra", act("stop_process", pids=[3])) for _ in range(10)])
        self.assertAlmostEqual(elsewhere, none)  # past infra incidents say nothing about this data fix
        self.assertAlmostEqual(self.belief([case("data", act("stop_session", pid=1))] * 10,
                                           OptimizerPolicy(precedent=False)), none)

    def test_the_plan_lists_the_precedents(self):
        self.d.history = HistoryReport(incident_id="i1", cases=[case("data", act("stop_session", pid=1),
                                                                     incident_id="old-1")])
        plan = Optimizer(TIERS).plan(self.d)
        self.assertEqual([(p.incident_id, p.owner, p.fix) for p in plan.precedents],
                         [("old-1", "data", ["stop_session on warehouse (pid)"])])


class ControllerWithHistoryTest(OrchestratorCase):
    """The loop: look up first -> experts see past cases -> plan -> outcome sent back to the history agent."""

    def controller(self, keeper, history="on", share=True) -> tuple[Controller, list]:
        seen_tasks = []
        agents = local_agents(self.evidence)

        class Spy(LocalLink):
            async def investigate(self, card, task):
                seen_tasks.append(task)
                return await super().investigate(card, task)

        c = Controller(Router(FixedClassifier(data=0.97), pool_registry()), Spy(agents, history=keeper),
                       OrchestratorPolicy(share_past_cases=share), history=history)
        return c, seen_tasks

    def setUp(self):
        self.store = ExperienceStore()
        self.keeper = HistoryAgent(HISTORY_CARD, self.store)

    def test_the_history_agent_is_asked_first_and_told_the_outcome(self):
        from controller.memory import Feedback

        incident = self.report("D1")
        c, tasks = self.controller(self.keeper)
        res = asyncio.run(c.handle(incident))
        self.assertEqual(res.deliberation.calls[0].kind, "history")
        self.assertEqual(res.deliberation.history.memory_size, 0)
        feedback = Feedback(incident_id=incident.incident_id, resolved=True, fixed_by=res.plan.actions[:1])
        asyncio.run(c.remember(incident, res, feedback))
        self.assertEqual(len(self.store), 1)

        # the same kind of incident again: the past case reaches the experts and the plan
        again = next(r for i, r in self.reports.items() if self.truth[i]["fault_id"] == "D1"
                     and r.incident_id != incident.incident_id)
        res2 = asyncio.run(c.handle(again))
        self.assertEqual([x.incident_id for x in res2.deliberation.history.cases], [incident.incident_id])
        self.assertTrue(all(t.past_cases for t in tasks[-len(res2.deliberation.proposals):]))
        self.assertEqual(res2.plan.precedents[0].incident_id, incident.incident_id)

    def test_switches(self):
        incident = self.report("D3")
        c, tasks = self.controller(self.keeper, share=False)
        self.store.add(remembered("old", LOCK.format(1), "2026-11-01T10:00:00+00:00"))
        asyncio.run(c.handle(incident))
        self.assertTrue(all(not t.past_cases for t in tasks))  # history used, but not shown to the experts
        off, _ = self.controller(None, history="off")
        res = asyncio.run(off.handle(incident))
        self.assertNotIn("history", [x.kind for x in res.deliberation.calls])
        self.assertIsNone(res.deliberation.history)

    def test_no_history_agent_answering_is_a_note_not_a_failure(self):
        c, _ = self.controller(None)  # history on, but the link has no history agent
        res = asyncio.run(c.handle(self.report("D3")))
        self.assertIsNone(res.deliberation.history)
        self.assertTrue(any("no history agent answered" in n for n in res.deliberation.notes))
        self.assertEqual(res.plan.kind in ("remediate", "approve", "escalate"), True)


if __name__ == "__main__":
    unittest.main()
