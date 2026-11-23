"""Block 6d: the experience memory in PostgreSQL.

The row tests run everywhere. The database tests run only when A2A_TEST_DB names a database they may wipe,
e.g. a throwaway server: never point it at a memory you want to keep.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller.memory import ExperienceStore, Feedback, HashingEmbedder, ProposalRecord, ReviewRecord
from controller.memory_pg import (DSN_ENV, dsn_of, embedder_name, experience_of, is_database, open_store, rows)
from controller.memory_sql import to_sql
from tests.test_memory import DISK, LOCK, remembered, report
from tests.test_optimizer import act

TEST_DB = os.environ.get("A2A_TEST_DB")


def full_experience(incident_id: str = "inc-1", owner: str = "data"):
    """An experience with every kind of field filled, including a quote and an operator's postmortem."""
    e = remembered(incident_id, LOCK.format(7), "2026-11-01T10:00:00+00:00", owner=owner)
    e.proposals = [
        ProposalRecord(agent_id="data-fast", domain="data", tier="fast", reasoner="rules", verdict="owner",
                       confidence=0.63, root_cause="session 7's lock", actions=[act("stop_session", pid=7)],
                       label="right", label_why="the fix that resolved it", domain_owned=True),
        ProposalRecord(agent_id="infra-fast", domain="infra", tier="fast", verdict="not_mine", confidence=0.7,
                       points_to="data", domain_owned=False)]
    e.reviews = [ReviewRecord(reviewer="infra-fast", domain="infra", claim_agent="data-fast", claim_domain="data",
                              stance="supports", right=True, claim_label="right", claim_owned=True)]
    e.outcome = Feedback(incident_id=incident_id, resolved=True, source="operator", owner=owner,
                         fixed_by=[act("stop_session", pid=7)], credits={"data-fast": True})
    e.plan_kind, e.plan_actions, e.plan_confidence = "remediate", [act("stop_session", pid=7)], 0.873
    return e


class RowsTest(unittest.TestCase):
    def test_an_experience_survives_the_trip_through_rows(self):
        e = full_experience()
        vector = HashingEmbedder().embed(e.features())
        exp, answers, reviews = rows(e, vector, embedder_name(HashingEmbedder()))
        self.assertEqual((len(answers), len(reviews), len(exp["embedding"])), (2, 1, 2048))
        self.assertEqual(experience_of(exp, answers, reviews).model_dump(), e.model_dump())

    def test_where_the_memory_lives(self):
        self.assertTrue(is_database("postgres") and is_database("postgresql://u@h/db"))
        self.assertFalse(is_database("models/memory"))
        old = os.environ.pop(DSN_ENV, None)
        try:
            with self.assertRaises(ValueError):  # the URL (with its password) belongs in the environment
                dsn_of("postgres")
        finally:
            if old is not None:
                os.environ[DSN_ENV] = old
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(open_store(Path(tmp) / "new")), 0)  # a new folder: an empty file store
            with self.assertRaises(FileNotFoundError):
                open_store(Path(tmp) / "new", must_exist=True)

    def test_the_sql_export_writes_the_same_columns(self):
        store = ExperienceStore()
        store.add(full_experience())
        sql = to_sql(store)
        self.assertIn("postmortem_owner", sql)
        self.assertIn("'session 7''s lock'", sql)
        self.assertIn("key = 'changes'", sql)  # readers of the database notice the new rows


@unittest.skipUnless(TEST_DB, "set A2A_TEST_DB to a throwaway PostgreSQL database to run these")
class DatabaseTest(unittest.TestCase):
    def setUp(self):
        from controller.memory_pg import PostgresExperienceStore, _psycopg

        conn = _psycopg().connect(TEST_DB, autocommit=True)
        conn.execute("DROP VIEW IF EXISTS agent_fix_record, reviewer_record; "
                     "DROP TABLE IF EXISTS reviews, proposals, experiences, memory_meta CASCADE")
        conn.close()
        self.open = lambda **kw: PostgresExperienceStore(TEST_DB, **kw)
        self.stores = []

    def tearDown(self):
        for s in self.stores:
            s.close()

    def store(self, **kw):
        s = self.open(**kw)
        self.stores.append(s)
        return s

    def test_what_one_program_stores_another_reads_back_exactly(self):
        e = full_experience()
        self.store().add(e)
        other = self.store()
        self.assertEqual([x.model_dump() for x in other.experiences], [e.model_dump()])
        self.assertTrue(np.array_equal(other.matrix[0], HashingEmbedder().embed(e.features())))

    def test_the_same_incident_again_replaces_its_rows(self):
        s = self.store()
        s.add(full_experience())
        s.add(full_experience(owner="security"))
        n = s.conn.execute("SELECT (SELECT count(*) FROM experiences) AS e, (SELECT count(*) FROM proposals) AS p, "
                           "(SELECT count(*) FROM reviews) AS r").fetchone()
        self.assertEqual((n["e"], n["p"], n["r"]), (1, 2, 1))
        self.assertEqual(self.store().experiences[0].owner, "security")

    def test_a_write_by_another_program_is_noticed(self):
        a, b = self.store(), self.store()
        a.add(full_experience("one"))
        self.assertTrue(b.refresh())
        self.assertFalse(b.refresh())  # nothing new since
        b.add(full_experience("two"))
        a.add(full_experience("three"))  # a missed "two": its write shows the counter jumped, so it reads again
        self.assertEqual(sorted(x.incident_id for x in a.experiences), ["one", "three", "two"])

    def test_it_finds_the_same_cases_as_the_file_store(self):
        files, db = ExperienceStore(), self.store()
        for e in (remembered("lock1", LOCK.format(1), "2026-11-01T10:00:00+00:00"),
                  remembered("disk1", DISK.format(1), "2026-11-02T10:00:00+00:00", owner="infra"),
                  remembered("lock2", LOCK.format(2), "2026-11-03T10:00:00+00:00")):
            files.add(e)
            db.add(e)
        query = report("q", LOCK.format(9))
        self.assertEqual([(round(s, 6), e.incident_id) for s, e in self.store().similar(query, k=3)],
                         [(round(s, 6), e.incident_id) for s, e in files.similar(query, k=3)])

    def test_vectors_from_another_embedder_are_refused(self):
        self.store().add(full_experience())
        with self.assertRaises(ValueError):
            self.store(embedder=HashingEmbedder(dim=1024))

    def test_clear(self):
        s = self.store()
        s.add(full_experience("one"))
        s.add(full_experience("two"))
        self.assertEqual((s.clear(), len(s), len(self.store())), (2, 0, 0))

    def test_the_history_agent_learns_into_postgres(self):
        from agents.history_agent import HistoryAgent
        from controller.history import HistoryQuery
        from controller.pool import default_pool
        from oncall.memory_eval import build_memory
        from oncall.synth import generate
        from oncall.synth.generate import write_dataset

        with tempfile.TemporaryDirectory() as tmp:
            incidents, truths, manifest = generate(per_fault=5, seed=47)
            write_dataset(Path(tmp), incidents, truths, manifest)
            store, r = build_memory(Path(tmp), split="train", trust="local", limit=20, store=self.store())
            self.assertEqual(r["labels"]["right"].get("truly wrong", 0), 0)
            reopened = self.store()
            self.assertEqual(len(reopened), 20)
            keeper = HistoryAgent(next(c for c in default_pool() if c.name == "history"), reopened)
            first = reopened.experiences[0]
            found = keeper.lookup(HistoryQuery(incident=report(first.incident_id + "-again", first.text)))
            self.assertIsNotNone(found.trust)
            self.assertEqual(found.memory_size, 20)


if __name__ == "__main__":
    unittest.main()
