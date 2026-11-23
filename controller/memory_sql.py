"""Export an experience memory as SQL for the PostgreSQL tables in controller/memory_schema.sql.

    python -m controller memory models/memory --export-sql models/memory.sql
    psql -U postgres -d a2a_memory -f controller/memory_schema.sql     # the tables (once)
    psql -U postgres -d a2a_memory -f models/memory.sql                # the rows

Plain text, no database driver needed (the live store is controller/memory_pg.py; both use its rows(), so
the export and the store write the same thing). Each incident replaces any earlier row with the same id
(its answers and reviews go with it), so loading a newer export again is safe. One transaction: all or
nothing.
"""
from __future__ import annotations

import json
from typing import Any

from .memory import ExperienceStore
from .memory_pg import (ARRAY_COLUMNS, EXPERIENCE_COLUMNS, JSON_COLUMNS, PROPOSAL_COLUMNS, REVIEW_COLUMNS,
                        TIME_COLUMNS, embedder_name, rows)


def literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def jsonb(v: Any) -> str:
    return "NULL" if v is None else literal(json.dumps(v)) + "::jsonb"


def text_array(xs: list[str]) -> str:
    return "ARRAY[" + ", ".join(literal(x) for x in xs) + "]::text[]" if xs else "'{}'::text[]"


def value(column: str, v: Any) -> str:
    if column in JSON_COLUMNS:
        return jsonb(v)
    if column in ARRAY_COLUMNS:
        return text_array(v)
    if column == "embedding":
        return "'{" + ",".join(f"{float(x):.9g}" for x in v) + "}'::real[]"
    if column in TIME_COLUMNS and v is not None:
        return literal(v) + "::timestamptz"
    return literal(v)


def insert(table: str, columns: tuple[str, ...], row: dict) -> str:
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(value(c, row[c]) for c in columns)});"


def to_sql(store: ExperienceStore) -> str:
    spec = store.embedder.spec()
    out = ["BEGIN;",
           f"INSERT INTO memory_meta (key, value) VALUES ('embedder', {jsonb(spec)}) "
           "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;"]
    for e, vector in zip(store.experiences, store.matrix):
        experience, answers, reviews = rows(e, vector, embedder_name(store.embedder))
        out.append(f"DELETE FROM experiences WHERE incident_id = {literal(e.incident_id)};")
        out.append(insert("experiences", EXPERIENCE_COLUMNS, experience))
        out += [insert("proposals", PROPOSAL_COLUMNS, a) for a in answers]
        out += [insert("reviews", REVIEW_COLUMNS, r) for r in reviews]
    out.append("UPDATE memory_meta SET value = to_jsonb((value #>> '{}')::bigint + 1) WHERE key = 'changes';")
    out.append("COMMIT;")
    return "\n".join(out) + "\n"
