"""Block 6a: the experience memory (labels from verified feedback, similarity search, disk) and filling it."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller.checks import same_fix
from controller.contracts import AlertIn, IncidentReport
from controller.intake import extract
from controller.memory import Experience, ExperienceStore, Feedback, HashingEmbedder, ProposalRecord, experience_from
from controller.plan import ActionOutcome, ExecutionReport, RemediationPlan
from controller.proposal import Proposal
from oncall.feedback import simulated_feedback
from oncall.memory_eval import build_memory, evaluate_retrieval
from oncall.synth import generate
from oncall.synth.generate import write_dataset
from tests.test_optimizer import TIERS, act, claim, deliberation, not_mine, review

LOCK = "ingest failed 2 runs in a row; last error: LockNotAvailable: canceling statement due to lock timeout (pid {})"
DISK = "ingest failed 2 runs in a row; last error: OSError: [Errno 28] No space left on device: '/spool/b{}.json'"


def report(incident_id: str, detail: str, opened_at: str = "2026-11-01T10:00:00+00:00") -> IncidentReport:
    return IncidentReport(incident_id=incident_id, opened_at=opened_at, title="IngestJobFailing",
                          alerts=[AlertIn(name="IngestJobFailing", detail=detail)])


def guess(agent: str, domain: str, *actions, verdict: str = "not_mine") -> Proposal:
    return Proposal(incident_id="i1", agent_id=agent, domain=domain, verdict=verdict, confidence=0.6,
                    root_cause=f"{agent}'s best guess", actions=list(actions))


def plan(*stages, kind: str = "remediate", owner: str = "data") -> RemediationPlan:
    actions = [a.model_copy(update={"stage": n}) for n, stage in enumerate(stages, 1) for a in stage]
    return RemediationPlan(incident_id="i1", kind=kind, owner=owner, team=[owner], root_cause="x", confidence=0.9,
                           actions=actions)


def remembered(incident_id: str, detail: str, opened_at: str, owner: str = "data") -> Experience:
    f = extract(report(incident_id, detail, opened_at))
    return Experience(incident_id=incident_id, opened_at=opened_at, text=f.text, alerts=f.alerts, signals=f.signals,
                      components=f.components, priority="P3", owner_probs={}, team=[owner], plan_kind="escalate",
                      plan_confidence=0.0, outcome=Feedback(incident_id=incident_id, resolved=True, owner=owner),
                      owner=owner)


class SameFixTest(unittest.TestCase):
    def test_identifiers_of_different_kinds_are_not_the_same_fix(self):
        # "session 556" and "the reporting-worker sessions" may be different sessions
        self.assertFalse(same_fix([act("stop_session", pid=556)], [act("stop_session", application_name="worker")]))

    def test_overlapping_identifiers_of_one_kind_are(self):
        self.assertTrue(same_fix([act("stop_session", pids=[1, 2])], [act("stop_session", pid=2)]))
        self.assertFalse(same_fix([act("stop_session", pids=[1, 2])], [act("stop_session", pid=3)]))

    def test_without_identifiers_the_target_decides(self):
        self.assertTrue(same_fix([act("restart_component")], [act("restart_component")]))
        self.assertFalse(same_fix([act("restart_component")], [act("restart_component", target="ingest")]))


class LabelTest(unittest.TestCase):
    """Labels say only what the verified feedback shows: right, wrong, or unknown (never a guess)."""

    def setUp(self):
        self.right, self.other = act("stop_session", pid=6079), act("stop_session", pid=4321)
        self.lock_claim = claim("data-strong", "data", self.right)
        self.data_guess = guess("data-fast", "data", self.other)
        self.infra_claim = claim("infra-fast", "infra", act("restart_component"))
        reviews = [review("infra-strong", "infra", self.lock_claim, "supports"),
                   review("data-strong", "data", self.data_guess, "contradicts"),
                   review("data-fast", "data", self.infra_claim, "contradicts"),
                   review("network-fast", "network", self.lock_claim, "cannot_tell")]
        self.d = deliberation({"data": 0.8, "infra": 0.2}, proposals=[
            self.lock_claim, self.data_guess, self.infra_claim, not_mine("network-fast", "network")], reviews=reviews)
        self.report = report("i1", LOCK.format(6079))

    def experience(self, feedback: Feedback, the_plan: RemediationPlan | None = None) -> Experience:
        return experience_from(self.report, self.d, the_plan or plan([self.right]), feedback, TIERS)

    def labels(self, e: Experience) -> dict:
        return {p.agent_id: p.label for p in e.proposals}

    def test_the_executor_alone_verifies_only_what_ran(self):
        e = self.experience(Feedback(incident_id="i1", resolved=True, fixed_by=[self.right]))
        self.assertEqual(e.owner, "data")  # inferred: the domain whose fix resolved it
        self.assertEqual(self.labels(e), {"data-strong": "right", "data-fast": "unknown", "infra-fast": "wrong",
                                          "network-fast": None})
        why = {p.agent_id: p.label_why for p in e.proposals}
        self.assertEqual(why["data-fast"], "not tried")  # might have been another valid fix: not called wrong
        self.assertEqual(why["infra-fast"], "the owner was data")
        owned = {p.agent_id: p.domain_owned for p in e.proposals}
        self.assertEqual(owned, {"data-strong": True, "data-fast": True, "infra-fast": False, "network-fast": False})
        self.assertEqual(next(p.tier for p in e.proposals if p.agent_id == "data-strong"), "strong")

    def test_reviews_are_right_when_they_back_a_right_fix_or_object_to_a_wrong_one(self):
        e = self.experience(Feedback(incident_id="i1", resolved=True, fixed_by=[self.right]))
        right = {(r.reviewer, r.claim_agent): r.right for r in e.reviews}
        self.assertEqual(right, {("infra-strong", "data-strong"): True,   # supported the fix that worked
                                 ("data-strong", "data-fast"): None,      # the guess was never checked
                                 ("data-fast", "infra-fast"): True,       # objected to a wrong-domain fix
                                 ("network-fast", "data-strong"): None})  # cannot tell is no opinion

    def test_a_fix_that_ran_and_left_it_broken_is_wrong(self):
        e = self.experience(Feedback(incident_id="i1", resolved=True, fixed_by=[self.right], failed=[self.other]),
                            plan([self.other], [self.right]))
        self.assertEqual(self.labels(e)["data-fast"], "wrong")

    def test_the_operators_postmortem_decides_when_there_is_one(self):
        e = self.experience(Feedback(incident_id="i1", resolved=True, owner="data", source="operator",
                                     credits={"data-fast": True, "data-strong": False, "infra-fast": False}))
        self.assertEqual(self.labels(e), {"data-strong": "wrong", "data-fast": "right", "infra-fast": "wrong",
                                          "network-fast": None})
        self.assertTrue(all(p.label_why == "operator's postmortem" for p in e.proposals if p.actions))

    def test_without_a_postmortem_or_a_fix_that_worked_nothing_is_known(self):
        e = self.experience(Feedback(incident_id="i1", resolved=True), plan(kind="escalate"))
        self.assertIsNone(e.owner)
        self.assertEqual(set(self.labels(e).values()), {"unknown", None})
        self.assertTrue(all(p.domain_owned is None for p in e.proposals))

    def test_feedback_must_be_about_this_incident(self):
        with self.assertRaises(ValueError):
            self.experience(Feedback(incident_id="other", resolved=True))


class FromReportTest(unittest.TestCase):
    def setUp(self):
        self.first, self.second = act("stop_session", pid=11), act("restart_component")
        self.plan = plan([self.first], [self.second])

    def read(self, statuses, resolved, **kw) -> Feedback:
        outcomes = [ActionOutcome(intent=a.intent, status=s) for a, s in zip(self.plan.actions, statuses)]
        return Feedback.from_report(self.plan, ExecutionReport(incident_id="i1", outcomes=outcomes,
                                                               resolved=resolved), **kw)

    def test_the_last_stage_that_ran_fixed_it_and_the_earlier_ones_did_not(self):
        fb = self.read(["done", "done"], True)
        self.assertEqual([a.intent for a in fb.fixed_by], ["restart_component"])
        self.assertEqual([a.intent for a in fb.failed], ["stop_session"])
        self.assertEqual(fb.source, "executor")

    def test_stopped_after_stage_one(self):
        fb = self.read(["done", "skipped"], True)
        self.assertEqual([a.intent for a in fb.fixed_by], ["stop_session"])
        self.assertEqual(fb.failed, [])

    def test_still_broken_means_everything_that_ran_failed(self):
        fb = self.read(["done", "done"], False, owner="infra", credits={"infra-fast": True})
        self.assertEqual((fb.fixed_by, len(fb.failed), fb.source), ([], 2, "operator"))

    def test_an_unverified_report_teaches_nothing(self):
        with self.assertRaises(ValueError):
            self.read(["done", "done"], None)

    def test_outcomes_are_matched_by_intent_when_the_counts_differ(self):
        report_ = ExecutionReport(incident_id="i1", resolved=True,
                                  outcomes=[ActionOutcome(intent="restart_component", status="done")])
        fb = Feedback.from_report(self.plan, report_)
        self.assertEqual([a.intent for a in fb.fixed_by], ["restart_component"])


class EmbedderTest(unittest.TestCase):
    def embed(self, detail: str) -> np.ndarray:
        return HashingEmbedder().embed(extract(report("x", detail)))

    def test_the_same_text_always_gives_the_same_unit_vector(self):
        a, b = self.embed(LOCK.format(5)), self.embed(LOCK.format(5))  # new embedder each time: no hidden state
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_pids_and_numbers_do_not_matter_but_the_failure_does(self):
        same = float(self.embed(LOCK.format(11)) @ self.embed(LOCK.format(98765)))
        different = float(self.embed(LOCK.format(11)) @ self.embed(DISK.format(3)))
        self.assertGreater(same, 0.99)
        self.assertLess(different, 0.8)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = ExperienceStore()
        self.store.add(remembered("lock1", LOCK.format(1), "2026-11-01T10:00:00+00:00"))
        self.store.add(remembered("disk1", DISK.format(1), "2026-11-02T10:00:00+00:00", owner="infra"))
        self.store.add(remembered("lock2", LOCK.format(2), "2026-11-03T10:00:00+00:00"))

    def ids(self, query, **kw) -> list[str]:
        return [e.incident_id for _, e in self.store.similar(query, **kw)]

    def test_most_similar_first(self):
        self.assertEqual(self.ids(report("q", LOCK.format(77)), k=3)[2], "disk1")
        self.assertEqual(len(self.ids(report("q", LOCK.format(77)), k=1)), 1)

    def test_never_the_incident_itself_nor_the_future_nor_excluded_ones(self):
        self.assertNotIn("lock1", self.ids(report("lock1", LOCK.format(1))))
        self.assertEqual(self.ids(report("q", LOCK.format(7)), before="2026-11-02T10:00:00+00:00"), ["lock1"])
        self.assertEqual(self.ids(report("q", LOCK.format(7)), exclude={"lock1", "lock2"}), ["disk1"])

    def test_an_empty_memory_finds_nothing(self):
        self.assertEqual(ExperienceStore().similar(report("q", LOCK.format(7))), [])

    def test_the_same_incident_again_replaces_the_old_record(self):
        self.store.add(remembered("lock1", LOCK.format(1), "2026-11-01T10:00:00+00:00", owner="security"))
        self.assertEqual(len(self.store), 3)
        self.assertEqual(next(e.owner for e in self.store.experiences if e.incident_id == "lock1"), "security")

    def test_saved_and_loaded_it_finds_the_same_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(Path(tmp))
            loaded = ExperienceStore.load(Path(tmp))
            query = report("q", LOCK.format(7))
            self.assertEqual([(round(s, 6), e.incident_id) for s, e in loaded.similar(query)],
                             [(round(s, 6), e.incident_id) for s, e in self.store.similar(query)])
            self.assertEqual(loaded.summary(), self.store.summary())
            loaded.add(remembered("disk2", DISK.format(2), "2026-11-04T10:00:00+00:00"))  # still growable
            self.assertEqual(len(loaded), 4)

    def test_a_damaged_store_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(Path(tmp))
            np.save(Path(tmp) / "vectors.npy", np.zeros((2, 2048), dtype=np.float32))
            with self.assertRaises(ValueError):
                ExperienceStore.load(Path(tmp))
            meta = json.loads((Path(tmp) / "store.json").read_text(encoding="utf-8"))
            (Path(tmp) / "store.json").write_text(json.dumps({**meta, "format": "other"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperienceStore.load(Path(tmp))


class SqlExportTest(unittest.TestCase):
    """controller/memory_sql.py writes rows for controller/memory_schema.sql (loaded for real with psql once)."""

    def test_rows_quotes_and_empty_values(self):
        from controller.memory_sql import to_sql

        store = ExperienceStore()
        e = remembered("it's-1", LOCK.format(1), "2026-11-01T10:00:00+00:00")
        e.proposals = [ProposalRecord(agent_id="data-fast", domain="data", tier="fast", verdict="owner",
                                      confidence=0.9, root_cause="session 1's lock", label="right")]
        store.add(e)
        sql = to_sql(store)
        self.assertTrue(sql.startswith("BEGIN;") and sql.rstrip().endswith("COMMIT;"))
        self.assertIn("'it''s-1'", sql)  # a quote inside a value is doubled, not left to break the statement
        self.assertIn("'session 1''s lock'", sql)
        self.assertEqual(sql.count("INSERT INTO experiences"), 1)
        self.assertEqual(sql.count("INSERT INTO proposals"), 1)
        self.assertEqual(sql.count("DELETE FROM experiences"), 1)  # loading again replaces, never duplicates
        self.assertIn("NULL", sql)  # e.g. no postmortem credits
        self.assertEqual(sql.count("::real[]"), 1)


class SimulatedFeedbackTest(unittest.TestCase):
    """The lab reports only what a real platform could: an operator speaks only when a person was involved."""

    truth = {"owner": "data", "valid_intents": ["stop_session"], "fix_targets": {"stop_session": [{"pid": [6079]}]}}
    seen = "lock held by pid 6079; pid 4321 is waiting"

    def setUp(self):
        self.right, self.other = act("stop_session", pid=6079), act("stop_session", pid=4321)
        self.d = deliberation({"data": 0.9}, proposals=[claim("data-strong", "data", self.right),
                                                        guess("data-fast", "data", self.other)])

    def test_a_plan_that_fixed_it_on_its_own_gets_only_the_executors_report(self):
        fb = simulated_feedback(self.d, plan([self.other], [self.right]), self.truth, self.seen)
        self.assertEqual((fb.source, fb.credits, fb.owner), ("executor", None, None))
        self.assertEqual((fb.fixed_by[0].params, fb.failed[0].params), ({"pid": 6079}, {"pid": 4321}))

    def test_a_person_involved_means_a_postmortem(self):
        for kind, stages in (("approve", ([self.other], [self.right])), ("escalate", ())):
            fb = simulated_feedback(self.d, plan(*stages, kind=kind), self.truth, self.seen)
            self.assertEqual(fb.source, "operator", kind)
            self.assertEqual(fb.credits, {"data-strong": True, "data-fast": False}, kind)
        fb = simulated_feedback(self.d, plan([self.other], [self.right], kind="approve"), self.truth, self.seen)
        self.assertEqual((fb.fixed_by[0].params, fb.failed), ({"pid": 6079}, []))  # the rejected stage never ran

    def test_a_plan_that_did_not_fix_it_hands_over_to_a_person(self):
        fb = simulated_feedback(self.d, plan([self.other]), self.truth, self.seen)
        self.assertEqual((fb.source, len(fb.failed), fb.fixed_by), ("operator", 1, []))

    def test_noise_makes_the_operator_wrong(self):
        fb = simulated_feedback(self.d, plan(kind="escalate"), self.truth, self.seen, noise=1.0,
                                rng=random.Random(1))
        self.assertEqual(fb.credits, {"data-strong": False, "data-fast": True})
        self.assertNotEqual(fb.owner, "data")


class FillingTest(unittest.TestCase):
    """The whole loop in the lab: controller -> plan -> verified feedback -> learn -> memory."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        incidents, truths, manifest = generate(per_fault=5, seed=41)  # 3 train, 1 val, 1 test per fault
        write_dataset(Path(cls.tmp.name), incidents, truths, manifest)
        cls.data = Path(cls.tmp.name)
        cls.store, cls.result = build_memory(cls.data, split="train")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_incident_is_remembered_oldest_first(self):
        self.assertEqual(len(self.store), self.result["incidents"])
        self.assertEqual(len(self.store), 42)
        times = [e.opened_at for e in self.store.experiences]
        self.assertEqual(times, sorted(times))

    def test_labels_from_realistic_feedback_never_contradict_ground_truth(self):
        labels = self.result["labels"]
        self.assertEqual(labels["right"].get("truly wrong", 0), 0)
        self.assertEqual(labels["wrong"].get("truly right", 0), 0)
        self.assertGreater(labels["right"].get("truly right", 0), 0)
        self.assertEqual(self.result["owners"].get("wrong", 0), 0)

    def test_an_operator_on_every_incident_leaves_nothing_unknown(self):
        _, r = build_memory(self.data, split="train", mode="full", limit=10)
        self.assertEqual(r["labels"]["unknown"], {})
        self.assertEqual(r["sources"], {"operator": 10})

    def test_an_always_wrong_operator_teaches_the_opposite(self):
        _, r = build_memory(self.data, split="train", mode="full", noise=1.0, limit=10)
        self.assertEqual(r["labels"]["right"].get("truly right", 0), 0)
        self.assertEqual(r["owners"].get("right", 0), 0)

    def test_similar_past_incidents_are_found_for_new_ones(self):
        r = evaluate_retrieval(self.store, self.data, split="val")
        self.assertEqual(r["queries"], 14)
        self.assertGreaterEqual(r["same_fault_top1"], 0.5)
        self.assertEqual(len(r["per_fault"]), 14)  # every fault reported


if __name__ == "__main__":
    unittest.main()
