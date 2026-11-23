"""Block 6d - The experience memory in PostgreSQL: the same store, with its records in tables.

    PostgresExperienceStore  an ExperienceStore whose every add() is first written to PostgreSQL, in one
                             transaction (the incident, every agent's answer, every review, and +1 on a change
                             counter). It reads everything when it opens and again whenever the counter shows
                             that another program wrote (refresh), so several programs can share one memory:
                             the history agent, `oncall remember`, an operator's tool
    open_store(spec)         "models/memory" (a folder of files), "postgres" (the database in the
                             A2A_MEMORY_DB environment variable) or a postgresql://... URL
    rows / experience_of     one Experience <-> its table rows (also used by memory_sql.py)

The tables are controller/memory_schema.sql; a store creates any that are missing. "Find similar" stays an
exact cosine search in Python over the stored embeddings (as with files) until pgvector is installed.

Driver: psycopg (pip install -e ".[db]"). On Windows it runs on the libpq of the PostgreSQL install (found
under Program Files, or in A2A_LIBPQ_DIR): the DLL bundled in "psycopg[binary]" can be blocked by Windows
Application Control, and the installed PostgreSQL's own library is already trusted there.
Keep the password out of code and git: put the URL in A2A_MEMORY_DB.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .memory import Experience, ExperienceStore, Feedback, ProposalRecord, ReviewRecord

DSN_ENV = "A2A_MEMORY_DB"
LIBPQ_ENV = "A2A_LIBPQ_DIR"
DATABASE_WORDS = ("postgres", "postgresql", "db")
SCHEMA = Path(__file__).with_name("memory_schema.sql")

EXPERIENCE_COLUMNS = ("incident_id", "opened_at", "recorded_at", "text", "alerts", "signals", "components",
                      "priority", "owner_probs", "team", "plan_kind", "plan_actions", "plan_confidence", "owner",
                      "postmortem_owner", "resolved", "outcome_source", "fixed_by", "failed", "credits",
                      "verified_at", "embedder", "embedding")
PROPOSAL_COLUMNS = ("incident_id", "seq", "agent_id", "domain", "tier", "reasoner", "verdict", "confidence",
                    "points_to", "root_cause", "actions", "label", "label_why", "domain_owned")
REVIEW_COLUMNS = ("incident_id", "seq", "reviewer", "domain", "claim_agent", "claim_domain", "stance", "was_right",
                  "claim_label", "claim_owned")
JSON_COLUMNS = frozenset({"signals", "owner_probs", "plan_actions", "fixed_by", "failed", "credits", "actions"})
TIME_COLUMNS = frozenset({"opened_at", "recorded_at", "verified_at"})
ARRAY_COLUMNS = frozenset({"alerts", "components", "team"})


# -- one Experience <-> its rows ----------------------------------------------------------------------

def embedder_name(embedder) -> str:
    spec = embedder.spec()
    return f"{spec['name']}/{spec['dim']}"


def rows(e: Experience, vector, embedder: str) -> tuple[dict, list[dict], list[dict]]:
    d = e.model_dump(mode="json")
    o = d["outcome"]
    experience = {
        "incident_id": e.incident_id, "opened_at": e.opened_at, "recorded_at": e.recorded_at, "text": e.text,
        "alerts": e.alerts, "signals": d["signals"], "components": e.components, "priority": e.priority,
        "owner_probs": d["owner_probs"], "team": e.team, "plan_kind": e.plan_kind, "plan_actions": d["plan_actions"],
        "plan_confidence": e.plan_confidence, "owner": e.owner, "postmortem_owner": o["owner"],
        "resolved": o["resolved"], "outcome_source": o["source"], "fixed_by": o["fixed_by"], "failed": o["failed"],
        "credits": o["credits"], "verified_at": o["verified_at"], "embedder": embedder,
        "embedding": [float(x) for x in vector]}
    proposals = [{"incident_id": e.incident_id, "seq": n, **{k: p[k] for k in PROPOSAL_COLUMNS[2:] if k in p}}
                 for n, p in enumerate(d["proposals"])]
    reviews = [{"incident_id": e.incident_id, "seq": n, "was_right": r["right"],
                **{k: r[k] for k in REVIEW_COLUMNS[2:] if k in r}} for n, r in enumerate(d["reviews"])]
    return experience, proposals, reviews


def _iso(value) -> str | None:
    if value is None or isinstance(value, str):
        return value
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="seconds" if value.microsecond == 0 else "microseconds")


def experience_of(row: dict, proposals: list[dict], reviews: list[dict]) -> Experience:
    outcome = Feedback(incident_id=row["incident_id"], resolved=row["resolved"], source=row["outcome_source"],
                       owner=row["postmortem_owner"], fixed_by=row["fixed_by"] or [], failed=row["failed"] or [],
                       credits=row["credits"], verified_at=_iso(row["verified_at"]))
    return Experience(
        incident_id=row["incident_id"], opened_at=_iso(row["opened_at"]), recorded_at=_iso(row["recorded_at"]),
        text=row["text"], alerts=list(row["alerts"] or []), signals=row["signals"] or {},
        components=list(row["components"] or []), priority=row["priority"], owner_probs=row["owner_probs"],
        team=list(row["team"] or []), plan_kind=row["plan_kind"], plan_actions=row["plan_actions"] or [],
        plan_confidence=row["plan_confidence"], outcome=outcome, owner=row["owner"],
        proposals=[ProposalRecord(**{k: p[k] for k in PROPOSAL_COLUMNS[2:]}) for p in proposals],
        reviews=[ReviewRecord(right=r["was_right"], **{k: r[k] for k in REVIEW_COLUMNS[2:] if k != "was_right"})
                 for r in reviews])


# -- the driver --------------------------------------------------------------------------------------

def _libpq_dirs() -> list[str]:
    dirs = [os.environ.get(LIBPQ_ENV, "")]
    root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "PostgreSQL"
    if root.is_dir():
        versions = [p for p in root.iterdir() if p.name.isdigit()]
        dirs += [str(p / "bin") for p in sorted(versions, key=lambda p: -int(p.name))]
    return [d for d in dirs if d and (Path(d) / "libpq.dll").exists()]


def _psycopg():
    if "psycopg" not in sys.modules and sys.platform == "win32":
        os.environ.setdefault("PSYCOPG_IMPL", "python")  # PostgreSQL's own libpq, not the bundled DLL
        found = _libpq_dirs()
        if found:
            os.environ["PATH"] = found[0] + os.pathsep + os.environ.get("PATH", "")
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError("the PostgreSQL memory needs psycopg: pip install -e \".[db]\" (on Windows it also needs "
                           f"a PostgreSQL install's libpq.dll, or set {LIBPQ_ENV} to the folder holding it)") from e
    return psycopg


def is_database(spec) -> bool:
    s = str(spec)
    return s in DATABASE_WORDS or s.startswith(("postgresql://", "postgres://"))


def dsn_of(spec) -> str:
    if str(spec) in DATABASE_WORDS:
        dsn = os.environ.get(DSN_ENV)
        if not dsn:
            raise ValueError(f"set {DSN_ENV} to the database URL first, e.g. "
                             "postgresql://postgres:YOUR_PASSWORD@localhost:5432/a2a_memory")
        return dsn
    return str(spec)


def open_store(spec, must_exist: bool = False) -> ExperienceStore:
    """A memory by name: a folder of files, or a PostgreSQL database ("postgres" = the URL in A2A_MEMORY_DB)."""
    if is_database(spec):
        return PostgresExperienceStore(dsn_of(spec))
    folder = Path(spec)
    if (folder / "store.json").exists():
        return ExperienceStore.load(folder)
    if must_exist:
        raise FileNotFoundError(f"no experience memory at {folder}")
    return ExperienceStore()


# -- the store -----------------------------------------------------------------------------------------

class PostgresExperienceStore(ExperienceStore):
    def __init__(self, dsn: str, embedder=None, create: bool = True):
        super().__init__(embedder)
        psycopg = _psycopg()
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb

        self._jsonb = Jsonb
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        info = self.conn.info
        self.where = f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}"  # no password
        self._changes = -1
        try:
            self.conn.execute("SET TIME ZONE 'UTC'")
            if create:
                self.conn.execute(SCHEMA.read_text(encoding="utf-8"))  # only creates what is missing
            self._check_embedder()
            self.refresh()
        except Exception:
            self.conn.close()  # refused (another embedder, no tables, ...): do not leave the connection open
            raise

    def _check_embedder(self) -> None:
        spec = self.embedder.spec()
        row = self.conn.execute("SELECT value FROM memory_meta WHERE key = 'embedder'").fetchone()
        empty = self.conn.execute("SELECT NOT EXISTS (SELECT 1 FROM experiences) AS empty").fetchone()["empty"]
        if row is not None and row["value"] != spec and not empty:
            raise ValueError(f"{self.where}: its vectors were made by {row['value']}, not {spec}")
        if row is None or row["value"] != spec:
            self.conn.execute("INSERT INTO memory_meta (key, value) VALUES ('embedder', %s) "
                              "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", [self._jsonb(spec)])

    def _counter(self) -> int:
        row = self.conn.execute("SELECT (value #>> '{}')::bigint AS n FROM memory_meta WHERE key = 'changes'").fetchone()
        return int(row["n"]) if row else 0

    def refresh(self) -> bool:
        """Read everything again if another program wrote since we last read. True if it did."""
        with self.conn.transaction():
            self.conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")  # one consistent snapshot
            changes = self._counter()
            if changes == self._changes:
                return False
            found = self.conn.execute(f"SELECT {', '.join(EXPERIENCE_COLUMNS)} FROM experiences "
                                      "ORDER BY opened_at NULLS LAST, incident_id").fetchall()
            answers: dict[str, list] = {}
            for p in self.conn.execute(f"SELECT {', '.join(PROPOSAL_COLUMNS)} FROM proposals ORDER BY incident_id, seq"):
                answers.setdefault(p["incident_id"], []).append(p)
            reviews: dict[str, list] = {}
            for r in self.conn.execute(f"SELECT {', '.join(REVIEW_COLUMNS)} FROM reviews ORDER BY incident_id, seq"):
                reviews.setdefault(r["incident_id"], []).append(r)
        self.experiences = [experience_of(row, answers.get(row["incident_id"], []), reviews.get(row["incident_id"], []))
                            for row in found]
        self._vectors = [np.asarray(row["embedding"], dtype=np.float32) for row in found]
        self._matrix, self._changes = None, changes
        self.version += 1
        return True

    def _values(self, row: dict, columns: tuple[str, ...]) -> list[Any]:
        return [self._jsonb(row[c]) if c in JSON_COLUMNS and row[c] is not None else row[c] for c in columns]

    def add(self, experience: Experience) -> None:
        vector = self.embedder.embed(experience.features())
        exp, answers, reviews = rows(experience, vector, embedder_name(self.embedder))
        casts = {c: "::timestamptz" for c in TIME_COLUMNS} | {"embedding": "::real[]"}
        marks = ", ".join(f"%s{casts.get(c, '')}" for c in EXPERIENCE_COLUMNS)
        with self.conn.transaction():
            self.conn.execute("DELETE FROM experiences WHERE incident_id = %s", [experience.incident_id])
            self.conn.execute(f"INSERT INTO experiences ({', '.join(EXPERIENCE_COLUMNS)}) VALUES ({marks})",
                              self._values(exp, EXPERIENCE_COLUMNS))
            with self.conn.cursor() as cur:
                if answers:
                    cur.executemany(f"INSERT INTO proposals ({', '.join(PROPOSAL_COLUMNS)}) VALUES "
                                    f"({', '.join(['%s'] * len(PROPOSAL_COLUMNS))})",
                                    [self._values(a, PROPOSAL_COLUMNS) for a in answers])
                if reviews:
                    cur.executemany(f"INSERT INTO reviews ({', '.join(REVIEW_COLUMNS)}) VALUES "
                                    f"({', '.join(['%s'] * len(REVIEW_COLUMNS))})",
                                    [self._values(r, REVIEW_COLUMNS) for r in reviews])
            changes = self._bump()
        if changes == self._changes + 1:  # only our own write since we last read: update what we hold
            self._changes = changes
            self._put(experience, vector)
        else:  # another program wrote too: read everything again
            self.refresh()

    def _bump(self) -> int:
        row = self.conn.execute("UPDATE memory_meta SET value = to_jsonb((value #>> '{}')::bigint + 1) "
                                "WHERE key = 'changes' RETURNING (value #>> '{}')::bigint AS n").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO memory_meta (key, value) VALUES ('changes', '1')")
            return 1
        return int(row["n"])

    def clear(self) -> int:
        """Delete every remembered incident (their answers and reviews go with them). Returns how many."""
        with self.conn.transaction():
            n = self.conn.execute("WITH gone AS (DELETE FROM experiences RETURNING 1) SELECT count(*) AS n FROM gone"
                                  ).fetchone()["n"]
            self._bump()
        self._changes = -1
        self.refresh()
        return int(n)

    def save(self, folder=None) -> str:
        """Every add() is already in the database; nothing to save. Returns where it lives (no password)."""
        return self.where

    def close(self) -> None:
        self.conn.close()
