-- a2a-oncall experience memory in PostgreSQL (the history agent's store; see controller/memory.py).
--
-- One row per closed incident (experiences), one per agent answer (proposals), one per review (reviews).
-- The same facts as models/memory/experiences.jsonl, in tables, so they can be counted with SQL and written
-- by several programs safely. The embedding is a plain array of numbers: the "find similar" search runs
-- in Python (exact cosine) until pgvector is installed, when the column can become vector(n).
--
-- Use a database of its own (never the sandbox warehouse, the platform this project breaks on purpose):
--   createdb -U postgres a2a_memory
--   psql -U postgres -d a2a_memory -f controller/memory_schema.sql
-- Safe to run again: it only creates what does not exist yet.

CREATE TABLE IF NOT EXISTS memory_meta (
    key   text PRIMARY KEY,                -- "format", "version", "embedder", "changes"
    value jsonb NOT NULL
);
INSERT INTO memory_meta (key, value) VALUES
    ('format', '"a2a-oncall/experience-memory"'),
    ('version', '1'),
    ('changes', '0')                       -- +1 on every write, so each reader knows when to read again
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS experiences (
    incident_id     text PRIMARY KEY,
    opened_at       timestamptz,
    recorded_at     timestamptz NOT NULL DEFAULT now(),
    -- the incident, as the controller saw it (masked: no pids, times or addresses)
    text            text NOT NULL,
    alerts          text[] NOT NULL DEFAULT '{}',
    signals         jsonb NOT NULL DEFAULT '{}',   -- observed evidence type -> lines showing it
    components      text[] NOT NULL DEFAULT '{}',
    -- the decision
    priority        text NOT NULL CHECK (priority IN ('P1', 'P2', 'P3')),
    owner_probs     jsonb NOT NULL,                -- the classifier's probabilities, before any agent answered
    team            text[] NOT NULL,
    plan_kind       text NOT NULL CHECK (plan_kind IN ('remediate', 'approve', 'escalate')),
    plan_actions    jsonb NOT NULL DEFAULT '[]',
    plan_confidence double precision NOT NULL CHECK (plan_confidence BETWEEN 0 AND 1),
    -- the verified outcome (executor's report + the operator's postmortem when a person was involved)
    owner           text CHECK (owner IN ('network', 'data', 'infra', 'security')),  -- NULL = not known
    postmortem_owner text CHECK (postmortem_owner IN ('network', 'data', 'infra', 'security')),  -- the operator's word
    resolved        boolean NOT NULL,
    outcome_source  text NOT NULL CHECK (outcome_source IN ('executor', 'operator')),
    fixed_by        jsonb NOT NULL DEFAULT '[]',   -- the actions after which it was healthy again
    failed          jsonb NOT NULL DEFAULT '[]',   -- actions that ran and did not fix it
    credits         jsonb,                         -- postmortem: agent id -> its fix would have fixed it
    verified_at     timestamptz,
    -- for "find similar incidents"
    embedder        text NOT NULL,                 -- e.g. "hashing-v1/2048"
    embedding       real[] NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    incident_id   text NOT NULL REFERENCES experiences (incident_id) ON DELETE CASCADE,
    seq           smallint NOT NULL,               -- order of the answers (round 1 first)
    agent_id      text NOT NULL,
    domain        text NOT NULL,
    tier          text NOT NULL,                   -- fast / strong / generalist
    reasoner      text NOT NULL DEFAULT '',        -- rules, ollama:qwen2.5:3b, ...
    verdict       text NOT NULL CHECK (verdict IN ('owner', 'unsure', 'not_mine')),
    confidence    double precision NOT NULL,       -- self-reported; stored, never trusted
    points_to     text,
    root_cause    text NOT NULL DEFAULT '',
    actions       jsonb NOT NULL DEFAULT '[]',
    label         text CHECK (label IN ('right', 'wrong', 'unknown')),  -- NULL = it handed in no fix
    label_why     text NOT NULL DEFAULT '',
    domain_owned  boolean,                         -- was its domain the verified owner? NULL = not known
    PRIMARY KEY (incident_id, seq)
);

CREATE TABLE IF NOT EXISTS reviews (
    incident_id   text NOT NULL REFERENCES experiences (incident_id) ON DELETE CASCADE,
    seq           smallint NOT NULL,
    reviewer      text NOT NULL,
    domain        text NOT NULL,                   -- the evidence the reviewer used
    claim_agent   text NOT NULL,
    claim_domain  text NOT NULL,
    stance        text NOT NULL CHECK (stance IN ('supports', 'contradicts', 'cannot_tell')),
    was_right     boolean,                         -- backed a right fix / objected to a wrong one; NULL = no opinion
    claim_label   text CHECK (claim_label IN ('right', 'wrong', 'unknown')),
    claim_owned   boolean,
    PRIMARY KEY (incident_id, seq)
);

-- upgrades for tables made by the first version of this file (no-ops on new tables)
ALTER TABLE experiences ADD COLUMN IF NOT EXISTS postmortem_owner text
    CHECK (postmortem_owner IN ('network', 'data', 'infra', 'security'));
ALTER TABLE experiences ALTER COLUMN plan_confidence TYPE double precision;
ALTER TABLE proposals ALTER COLUMN confidence TYPE double precision;

CREATE INDEX IF NOT EXISTS experiences_opened_at ON experiences (opened_at);
CREATE INDEX IF NOT EXISTS proposals_agent ON proposals (agent_id, verdict);
CREATE INDEX IF NOT EXISTS reviews_reviewer ON reviews (reviewer);

-- What the memory knows about each agent's fixes when its domain owned the incident (Block 6b counts).
CREATE OR REPLACE VIEW agent_fix_record AS
SELECT agent_id, verdict,
       count(*) FILTER (WHERE label = 'right')   AS right_fixes,
       count(*) FILTER (WHERE label = 'wrong')   AS wrong_fixes,
       count(*) FILTER (WHERE label = 'unknown') AS unchecked_fixes,
       round(count(*) FILTER (WHERE label = 'right')::numeric
             / nullif(count(*) FILTER (WHERE label IN ('right', 'wrong')), 0), 3) AS right_share
FROM proposals
WHERE domain_owned AND label IS NOT NULL
GROUP BY agent_id, verdict;

-- How often each reviewer was right (it backed a right fix or objected to a wrong one).
CREATE OR REPLACE VIEW reviewer_record AS
SELECT reviewer,
       count(*) FILTER (WHERE was_right)     AS right_reviews,
       count(*) FILTER (WHERE NOT was_right) AS wrong_reviews,
       count(*) FILTER (WHERE was_right IS NULL) AS no_opinion
FROM reviews
GROUP BY reviewer;
