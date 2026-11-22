"""Block 5a-1: evidence tools, the rule reasoner, and the Ollama reasoner (mocked)."""
import json
import random
import unittest

import httpx

from agents.reasoners import ReasonerFailed
from agents.reasoners.ollama import OllamaReasoner
from agents.reasoners.rules import RuleReasoner
from agents.specialist import SpecialistAgent
from agents.tools import RecordedEvidence, Toolbox, ToolNotAllowed
from controller.contracts import IncidentReport
from controller.evidence import EvidenceBundle
from controller.pool import default_pool
from controller.proposal import InvestigationTask, Proposal
from oncall.agent_eval import correct_fix, what_agent_saw
from oncall.synth import generate
from oncall.synth.evidence import baseline_bundle
from oncall.synth.world import make_world

CARDS = {c.name: c for c in default_pool()}


def agent(name: str, evidence, reasoner=None) -> SpecialistAgent:
    return SpecialistAgent(CARDS[name], reasoner or RuleReasoner(), evidence)


class SynthCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        incidents, truths, _ = generate(per_fault=4, seed=5)
        cls.reports = {r["incident_id"]: IncidentReport.from_lab_record(r) for r in incidents}
        cls.truth = {t["incident_id"]: t for t in truths}
        cls.evidence = RecordedEvidence({r["incident_id"]: EvidenceBundle.model_validate(r["evidence"]["domain"])
                                         for r in incidents})

    def cases(self, fault: str, complete_only: bool = True):
        for incident_id, t in self.truth.items():
            if t["fault_id"] == fault and not (complete_only and any(n.startswith("missing:") for n in t["noise"])):
                yield self.reports[incident_id], t

    def ask(self, name: str, report: IncidentReport, domain: str | None = None):
        a = agent(name, self.evidence)
        return a.handle(InvestigationTask(incident=report, domain=domain or a.profile.domains[0]))


class ToolTest(SynthCase):
    def test_agents_can_only_use_tools_on_their_card(self):
        report = next(iter(self.reports.values()))
        box = Toolbox(self.evidence, CARDS["network-fast"].routing_profile().tools, report.incident_id)
        with self.assertRaises(ToolNotAllowed):
            box.call("active_queries")
        self.assertIsNone(box.try_call("top_processes"))

    def test_missing_evidence_makes_the_agent_unsure(self):
        bundle = baseline_bundle(make_world(random.Random(0)))
        bundle.network.probes = []
        report = next(iter(self.reports.values()))
        p = agent("network-strong", RecordedEvidence({report.incident_id: bundle})).handle(
            InvestigationTask(incident=report, domain="network"))
        self.assertEqual(p.verdict, "unsure")


class RuleReasonerTest(SynthCase):
    STRONG_OWNER = {"N": "network-strong", "D": "data-strong", "I": "infra-strong", "S": "security-strong"}

    def test_strong_owner_agents_solve_their_faults(self):
        for fault in ("N1", "N2", "N3", "D1", "D2", "D3", "D4", "D5", "I1", "I2", "I3", "I4", "S1", "S2"):
            for report, truth in self.cases(fault):
                with self.subTest(fault=fault, incident=report.incident_id):
                    p = self.ask(self.STRONG_OWNER[fault[0]], report)
                    self.assertEqual(p.verdict, "owner", p.findings)
                    self.assertTrue({a.intent for a in p.actions} & set(truth["valid_intents"]), p.actions)
                    seen = what_agent_saw(p, report, self.evidence)
                    self.assertTrue(any(correct_fix(a, truth, seen) for a in p.actions), (p.actions, truth))
                    self.assertTrue(p.findings)  # it says what it saw

    def test_every_fault_but_d2_names_its_culprit(self):
        for truth in self.truth.values():
            with self.subTest(fault=truth["fault_id"]):
                self.assertEqual(bool(truth["fix_targets"]), truth["fault_id"] != "D2")
                self.assertTrue(set(truth["fix_targets"]) <= set(truth["valid_intents"]))

    def test_fast_agents_are_cheaper_but_have_blind_spots(self):
        for report, _ in self.cases("D1"):
            fast, strong = self.ask("data-fast", report), self.ask("data-strong", report)
            self.assertNotEqual(fast.verdict, "owner")  # the source payload tool is beyond its budget
            self.assertEqual(strong.verdict, "owner")
            self.assertLess(len(fast.tools_used), len(strong.tools_used))

    def test_a_shy_agent_loses_its_fix_unless_it_must_propose(self):
        from agents.reasoners.shy import ShyReasoner

        report, truth = next(self.cases("D3"))
        shy = agent("data-strong", self.evidence, ShyReasoner())
        abstain = shy.handle(InvestigationTask(incident=report, domain="data"))
        forced = shy.handle(InvestigationTask(incident=report, domain="data", must_propose=True))
        self.assertEqual((abstain.verdict, abstain.actions), ("not_mine", []))  # the right fix is thrown away
        self.assertEqual(forced.verdict, "not_mine")
        seen = what_agent_saw(forced, report, self.evidence)
        self.assertTrue(any(correct_fix(a, truth, seen) for a in forced.actions))  # ...or kept for the optimizer

    def test_a_chosen_agent_with_nothing_clear_gives_an_honest_best_guess(self):
        report, _ = next(self.cases("N1"))  # a network fault: the security agent sees nothing wrong
        p = agent("security-strong", self.evidence).handle(
            InvestigationTask(incident=report, domain="security", must_propose=True))
        self.assertEqual(p.verdict, "not_mine")
        self.assertIn("best guess", p.root_cause)
        self.assertEqual([a.intent for a in p.actions], ["rotate_credential"])
        self.assertEqual(self.ask("security-strong", report).actions, [])  # no guess unless asked for one

    def test_deep_evidence_separates_the_cpu_look_alikes(self):
        for report, _ in self.cases("I1"):
            self.assertEqual(self.ask("infra-strong", report).actions[0].intent, "stop_process")
            self.assertNotEqual(self.ask("data-strong", report).verdict, "owner")
        for report, _ in self.cases("D5"):
            infra = self.ask("infra-strong", report)
            self.assertNotEqual(infra.verdict, "owner")
            self.assertEqual(infra.points_to, "data")  # "the CPU is used by database sessions"
            self.assertEqual(self.ask("data-strong", report).actions[0].intent, "stop_session")

    def test_healthy_domain_says_not_mine(self):
        for report, _ in self.cases("I3"):
            p = self.ask("network-strong", report)
            self.assertIn(p.verdict, ("not_mine", "unsure"))
            self.assertEqual(p.actions, [])

    def test_generalist_finds_the_owner_domain_itself(self):
        solved = total = 0
        for fault in ("N2", "D3", "I4", "S2"):
            for report, truth in self.cases(fault):
                p = self.ask("generalist", report, domain="network")
                total += 1
                solved += p.verdict == "owner" and p.domain == truth["owner"]
        self.assertEqual(solved, total)

    def test_agent_refuses_a_domain_it_does_not_cover(self):
        report = next(iter(self.reports.values()))
        p = self.ask("network-fast", report, domain="data")
        self.assertEqual((p.verdict, p.confidence), ("unsure", 0.0))


class OllamaReasonerTest(SynthCase):
    """The LLM answer is mocked; what matters is how the agent parses and checks it."""

    def setUp(self):
        self.report, self.d3_truth = next(self.cases("D3"))
        locks = self.evidence.bundles[self.report.incident_id].data.locks
        self.culprit = next(lock.pid for lock in locks if lock.granted)
        self.victim = next(lock.pid for lock in locks if not lock.granted)

    def run_with(self, handler, must_propose: bool = False):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return agent("data-strong", self.evidence, OllamaReasoner(client=client, retry_wait_s=0)).handle(
            InvestigationTask(incident=self.report, domain="data", must_propose=must_propose))

    def answer(self, *actions, verdict="owner"):
        body = {"verdict": verdict, "root_cause": "maintenance session holds a lock", "confidence": 0.9,
                "findings": [{"tool": "locks", "fact": "a session holds AccessExclusiveLock"}],
                "actions": list(actions), "points_to": "none"}
        return lambda request: httpx.Response(200, json={"message": {"content": json.dumps(body)}})

    def test_parses_the_model_answer_and_drops_disallowed_intents(self):
        seen = {}

        def handler(request):
            body = json.loads(request.content)
            seen.update(body)
            answer = {"verdict": "owner", "root_cause": "maintenance session holds a lock", "confidence": 0.9,
                      "findings": [{"tool": "locks", "fact": f"pid {self.culprit} holds AccessExclusiveLock"}],
                      "actions": [{"intent": "stop_session", "target": "warehouse", "params": {"pid": self.culprit},
                                   "rationale": "it blocks ingest"},
                                  {"intent": "rotate_credential", "target": "warehouse", "rationale": "x"}],
                      "points_to": "none"}
            return httpx.Response(200, json={"message": {"content": json.dumps(answer)}, "prompt_eval_count": 900,
                                             "eval_count": 120, "total_duration": 4_500_000_000})

        p = self.run_with(handler)
        self.assertEqual(seen["model"], "llama3")  # from the data-strong card
        self.assertEqual(seen["format"]["properties"]["root_cause"]["minLength"], 20)  # a sentence, not a word
        self.assertEqual(seen["options"]["temperature"], 0)
        schema = seen["format"]["properties"]
        shapes = {(s["properties"]["intent"]["enum"][0], *s["properties"]["params"]["required"]):
                  s["properties"]["target"]["enum"] for s in schema["actions"]["items"]["anyOf"]}
        self.assertEqual(shapes[("stop_session", "pid")], ["warehouse"])  # a session can only be stopped in the DB
        self.assertIn(("stop_session", "application_name"), shapes)
        self.assertNotIn(("stop_session",), shapes)  # a stop_session that names no session is not a valid answer
        self.assertNotIn("rotate_credential", json.dumps(schema["actions"]))  # not data-strong's intent
        self.assertIn("locks", schema["findings"]["items"]["properties"]["tool"]["enum"])
        self.assertIn('{"pid": <integer>}', seen["messages"][1]["content"])  # the prompt shows each param shape
        self.assertEqual((p.verdict, [a.intent for a in p.actions]), ("owner", ["stop_session"]))
        self.assertTrue(any("rotate_credential" in f.fact for f in p.findings))
        self.assertEqual(p.usage, {"prompt_tokens": 900, "completion_tokens": 120, "seconds": 4.5})
        self.assertIsNone(p.points_to)

    def test_the_model_observes_before_it_judges(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return self.answer()(request)

        self.run_with(handler)
        # both models once wrote "not mine" first and then described their own domain's fault
        self.assertEqual(list(seen["format"]["properties"]),
                         ["findings", "root_cause", "verdict", "points_to", "confidence", "actions"])
        self.assertEqual(seen["format"]["required"], list(seen["format"]["properties"]))
        system = seen["messages"][0]["content"]
        self.assertIn("WHAT THE DATA DOMAIN COVERS", system)  # the domain's skills, from its Agent Card
        self.assertIn("Diagnose warehouse contention", system)
        self.assertIn("SchemaMismatch: source payload missing required field(s)", system)
        # an example, not a placeholder: llama3 copied '"<field the source sends now>"' as the key once
        self.assertIn('"mapping": {"cust_ref": "customer_id"} (from the name the source sends now',
                      seen["messages"][1]["content"])
        self.assertNotIn('"<field', seen["messages"][1]["content"])

    def test_server_error_fails_the_task(self):
        # no fake "unsure": the controller counts a failed call and asks the domain's fallback agent
        with self.assertRaisesRegex(ReasonerFailed, "HTTPStatusError"):
            self.run_with(lambda request: httpx.Response(500, text="model not found"))

    def test_answers_are_bounded(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return self.answer()(request)

        self.run_with(handler)
        self.assertEqual(seen["options"]["num_predict"], 1024)
        self.assertEqual(seen["format"]["properties"]["findings"]["maxItems"], 8)
        cut = lambda request: httpx.Response(200, json={"message": {"content": '{"findings": [{"tool": "lo'},
                                                        "done_reason": "length"})
        with self.assertRaisesRegex(ReasonerFailed, "cut off after 1024 tokens"):
            self.run_with(cut)

    def test_a_rejected_fix_gets_one_repair_turn(self):
        invented = next(n for n in range(60, 9000) if n not in (self.culprit, self.victim)
                        and str(n) not in self.report.model_dump_json())
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            pid = invented if len(bodies) == 1 else self.culprit
            return self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": pid},
                                "rationale": "it holds the lock"})(request)

        p = self.run_with(handler)
        self.assertEqual(len(bodies), 2)  # one repair turn, no more
        repair = bodies[1]["messages"]
        self.assertEqual([m["role"] for m in repair], ["system", "user", "assistant", "user"])
        self.assertIn(f"pid={invented}", repair[3]["content"])  # the model sees what was rejected, and why
        self.assertIn(f'"pid": {invented}', repair[2]["content"])  # its own first answer
        self.assertEqual((p.verdict, [a.params for a in p.actions]), ("owner", [{"pid": self.culprit}]))
        self.assertTrue(any(f.fact.startswith("revised after the checks rejected") for f in p.findings))
        self.assertEqual(p.usage["calls"], 2)

    def test_a_repair_that_fails_again_is_still_dropped(self):
        calls = []

        def handler(request):
            calls.append(1)
            return self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": 99999},
                                "rationale": "x"})(request)

        p = self.run_with(handler)
        self.assertEqual((len(calls), p.verdict, p.actions), (2, "unsure", []))

    def test_no_repair_turn_for_fixes_that_would_be_dropped_anyway(self):
        calls = []

        def handler(request):  # "not mine" may not carry fixes when the agent may abstain
            calls.append(1)
            return self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": 99999},
                                "rationale": "x"}, verdict="not_mine")(request)

        self.run_with(handler, must_propose=False)
        self.assertEqual(len(calls), 1)

    def test_one_server_error_is_retried(self):
        calls = []

        def handler(request):  # Ollama's 500 while it swaps models on a small GPU, then a real answer
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(500, text="model runner has unexpectedly stopped")
            return self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": self.culprit},
                                "rationale": "it holds the lock"})(request)

        p = self.run_with(handler)
        self.assertEqual((len(calls), p.verdict), (2, "owner"))

    def test_ownership_without_a_fix_is_downgraded(self):
        self.assertEqual(self.run_with(self.answer()).verdict, "unsure")

    def test_the_llama3_answer_is_not_executable(self):
        # what llama3 really said for a D3 lock: right intent, but no session named and no real target
        p = self.run_with(self.answer({"intent": "stop_session", "target": "ingest session", "params": {},
                                       "rationale": "the ingest session hit a lock timeout"}))
        self.assertEqual((p.verdict, p.actions), ("unsure", []))
        notes = " ".join(f.fact for f in p.findings if f.tool == "action-check")
        self.assertIn("not a platform component", notes)
        self.assertIn("must say which pid", notes)

    def test_the_second_llama3_answer_is_not_executable_either(self):
        # after the prompt fix: right culprit in the root cause, but the fix still named nothing
        p = self.run_with(self.answer({"intent": "stop_session", "target": "prometheus", "params": {},
                                       "rationale": "adhoc-maintenance session holds the lock"}))
        self.assertEqual(p.verdict, "unsure")
        notes = " ".join(f.fact for f in p.findings if f.tool == "action-check")
        self.assertIn("acts on warehouse, not prometheus", notes)

    def test_an_invented_pid_is_dropped(self):
        invented = next(n for n in range(60, 9000) if n not in (self.culprit, self.victim)
                        and str(n) not in self.report.model_dump_json())
        p = self.run_with(self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": invented},
                                       "rationale": "x"}))
        self.assertEqual(p.verdict, "unsure")
        self.assertTrue(any(f"pid={invented}" in f.fact for f in p.findings))

    def test_a_not_mine_answer_carries_no_fix_and_no_self_pointer(self):
        # what llama3 did: "not mine" with a fix attached, pointing at its own domain
        p = self.run_with(self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": self.culprit},
                                       "rationale": "x"}, verdict="not_mine"))
        p_points = self.run_with(lambda r: httpx.Response(200, json={"message": {"content": json.dumps(
            {"verdict": "not_mine", "root_cause": "the data domain looks healthy", "confidence": 0.8,
             "findings": [], "actions": [], "points_to": "data"})}}))
        self.assertEqual((p.verdict, p.actions), ("not_mine", []))
        self.assertTrue(any("cannot propose fixes" in f.fact for f in p.findings))
        self.assertIsNone(p_points.points_to)

    def test_a_chosen_agent_must_hand_in_a_fix(self):
        seen = {}
        fix = {"intent": "stop_session", "target": "warehouse", "params": {"pid": self.culprit}, "rationale": "x"}

        def handler(request):
            seen.update(json.loads(request.content))
            return self.answer(fix, verdict="not_mine")(request)

        p = self.run_with(handler, must_propose=True)
        self.assertEqual(seen["format"]["properties"]["actions"]["minItems"], 1)  # no answer without a fix
        self.assertIn("always at least one", seen["messages"][0]["content"])
        # what qwen did (right pid, "not mine"): the fix is kept, the verdict says how sure the agent was
        self.assertEqual((p.verdict, [a.params for a in p.actions]), ("not_mine", [{"pid": self.culprit}]))
        self.assertNotIn("minItems", self.run_with_schema(must_propose=False)["actions"])

    def run_with_schema(self, must_propose: bool) -> dict:
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return self.answer()(request)

        self.run_with(handler, must_propose=must_propose)
        return seen["format"]["properties"]

    def test_a_chosen_agent_still_cannot_invent_a_fix(self):
        invented = {"intent": "stop_session", "target": "warehouse", "params": {"pid": 99999}, "rationale": "x"}
        p = self.run_with(self.answer(invented, verdict="not_mine"), must_propose=True)
        self.assertEqual(p.actions, [])
        self.assertTrue(any("pid=99999" in f.fact for f in p.findings))

    def review_with(self, handler):
        claim = Proposal(incident_id=self.report.incident_id, agent_id="infra-fast", domain="infra", verdict="owner",
                         root_cause="the warehouse runs out of CPU", confidence=0.8)
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return agent("data-strong", self.evidence, OllamaReasoner(client=client, retry_wait_s=0)).review(
            InvestigationTask(incident=self.report, domain="data", role="review", claim=claim))

    def test_review_parses_the_stance(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            body = {"stance": "contradicts", "reason": f"pid {self.culprit} holds a lock", "confidence": 0.8}
            return httpx.Response(200, json={"message": {"content": json.dumps(body)}})

        r = self.review_with(handler)
        self.assertEqual((r.stance, r.claim_agent, r.reviewer), ("contradicts", "infra-fast", "data-strong"))
        self.assertIn("the warehouse runs out of CPU", seen["messages"][1]["content"])  # the claim is shown
        self.assertEqual(seen["format"]["properties"]["stance"]["enum"], ["supports", "contradicts", "cannot_tell"])
        self.assertEqual(list(seen["format"]["properties"]), ["reason", "stance", "confidence"])  # fact, then side
        self.assertIn("WHAT THE DATA DOMAIN COVERS", seen["messages"][0]["content"])
        self.assertIn("locks", r.tools_used)

    def test_review_error_fails_the_task(self):
        with self.assertRaises(ReasonerFailed):
            self.review_with(lambda request: httpx.Response(500, text="boom"))

    def test_the_victim_pid_passes_the_agent_check(self):
        # grounded (it is in the lock table) but wrong: only ground truth can catch this (agent_eval)
        p = self.run_with(self.answer({"intent": "stop_session", "target": "warehouse", "params": {"pid": self.victim},
                                       "rationale": "x"}))
        self.assertEqual((p.verdict, p.actions[0].params), ("owner", {"pid": self.victim}))
        seen = what_agent_saw(p, self.report, self.evidence)
        self.assertFalse(correct_fix(p.actions[0], self.d3_truth, seen))  # the evaluation does not count it


if __name__ == "__main__":
    unittest.main()
