# a2a-oncall

Adaptive, domain-aware routing of production incidents across specialist AI agents that talk over the
[A2A protocol](https://a2a-protocol.org/latest/).

The use case is on-call for a small food-delivery analytics platform. When something breaks, a controller
has to decide which specialist (network, data, infra, security) owns the incident, which others to consult,
and how to learn from verified outcomes. The research questions are about that routing decision. The rest
of the system exists so those decisions can be measured honestly.

> **Status:** the controller (routing, A2A expert agents, cross-checking, the expected-loss optimizer,
> the policy agent, human approval) and the experience memory, learned agent trust and the history agent are built and tested (255 unit tests) on
> synthetic incidents, with rule-based and local-LLM agents (Ollama). The Docker sandbox is built and its
> compose file validates, but it has not yet been run end to end (see [Calibration](#calibration)).

## The platform under test

```
orders-api ──(toxiproxy)──► ingest ──(toxiproxy)──► warehouse (Postgres 16) ──► transform (dbt build) ──► metrics-api
 source system            every 15 s, spools        raw → staging → marts        every 30 s, tests          dashboard backend
                          to /spool first                                         gate the mart              (login + revenue)
                  Prometheus + postgres-exporter scrape everything; ops.job_runs records every job run
```

A health oracle judges the platform against SLOs. The main one: the dashboard data is at most 120 s
behind the source. The oracle also checks job success streaks, dashboard availability and p95 latency,
warehouse connection saturation, unexpected use of the ETL credential, and failed-login rate. "Resolved"
always means every check passes, never an LLM's opinion.

## Fault catalogue

The **owner** is the domain whose fix resolves the root cause (remediation-based ownership), not the
place where the symptom appears. Every fault shares a symptom with a fault owned by a different domain,
so the alert text alone never identifies the owner. `tests/test_faults_catalog.py` enforces this.

| id | injected fault | what on-call sees | owner | reference fix (allowlisted action) |
|---|---|---|---|---|
| N1 | 8 s latency on the primary source path (Toxiproxy) | ingest `ReadTimeout`, mart stale | network | `switch_source_endpoint(secondary)` |
| N2 | connection resets on the primary warehouse path | ingest connection errors | network | `switch_warehouse_endpoint(secondary)` |
| N3 | source hostname no longer resolves | ingest `ConnectError` | network | `rollback_service_config(ingest)` |
| D1 | source renames `amount` → `total_amount` | ingest `SchemaMismatch` | data | `set_field_mapping` (high risk) |
| D2 | batch with null amounts + re-sent orders | dbt tests fail, mart stale | data | `quarantine_rows(batch_id)` |
| D3 | maintenance session holds an exclusive lock | ingest lock timeout, dbt timeout | data | `terminate_backend(pid)` |
| D4 | reporting worker leaks connections | "too many clients", saturation | data | `terminate_sessions(app)` |
| D5 | analyst notebook runs CPU-heavy queries | warehouse CPU, slow dashboard | data | `terminate_sessions(app)` |
| I1 | unknown process burns warehouse CPU | warehouse CPU, slow dashboard | infra | `kill_container_process(pids)` |
| I2 | transform memory limit too small | dbt killed (exit 137) | infra | `set_memory_limit(transform, 512)` |
| I3 | ingest spool disk full | ingest `OSError: No space left` | infra | `purge_spool` |
| I4 | bad config deploy | dashboard crash-loops | infra | `rollback_service_config(metrics_api)` |
| S1 | unapproved client bulk-exports with the ETL credential | unexpected client, DB pressure | security | `rotate_db_credential` + `terminate_sessions` |
| S2 | password-guessing client | login-failure spike, slow dashboard | security | `block_ip` |

`python -m oncall faults actions` prints the full action allowlist with risk tiers and the domains
allowed to propose each action.

## Quick start

The stack needs Docker and a few GB of RAM. Use a **GitHub Codespace** (4-core); `.devcontainer/` sets
it up with Docker-in-Docker. Running it on a laptop with Docker Desktop works the same way.

```bash
pip install -e ".[dev]"                 # done automatically in the Codespace
python -m oncall init                   # live runtime configs from runtime/defaults
docker compose --profile lab build      # sandbox images + the toolbox image used by client-side faults
docker compose up -d
python -m oncall health                 # ~1-2 min until every check passes
python -m oncall validate N1            # one episode: inject -> detect -> reference fix -> recover -> revert
python -m oncall validate               # all 14 faults (~60-75 min)
```

Other commands:

```bash
python -m oncall faults list | actions
python -m oncall faults inject D3 [-p key=value]   # break it and leave it broken
python -m oncall watch                              # live incident detection in another terminal
python -m oncall faults fix D3                      # apply the reference fix
python -m oncall faults revert D3 | all
python -m oncall reset                              # back to baseline
python -m unittest discover -s tests -t .           # unit tests (no Docker needed)
```

No Docker? Generate development data and score the controller on it:

```bash
python -m oncall synth --per-fault 30 --seed 7      # 420 labelled incidents in data/synthetic/dev (~1 s)
python -m oncall evaluate --data data/synthetic/dev --split val
```

The generator (`oncall/synth/`) simulates platform state over time and runs it through the real oracle and
incident detector, so alert texts match the sandbox. Every incident gets an evidence bundle (one slice per
expert domain), random variants, red herrings and sometimes missing evidence. It is for development only:
final results must come from sandbox incidents.

Each validation episode appends to `lab/` (git-ignored). `incidents.jsonl` holds what a controller will see
(alerts + evidence). `ground_truth.jsonl` links each incident to the injected fault, its owner and its valid
fixes. `episodes.jsonl` records detection and recovery times, and `actions.jsonl` is the audit log.

## Information barrier

Agents in later phases must not see ground truth. They get no access to `lab/`, the Toxiproxy admin API,
orders-api `/admin/*`, the warehouse superuser, or the toolbox scripts. Their tools measure the system
(latency probes, `pg_stat_activity` via the read-only `agent_ro` role, container stats, logs, Prometheus),
the way a human on-call engineer would. Fault artefacts use neutral names for the same reason.

## Calibration

Some faults depend on the host's speed, so the first `validate` run may need tuning via `-p`:

- **I1 / D5** (CPU): if no alert fires, raise `procs` (I1) or `threads`/`fanout` (D5). The alert usually
  arrives as `DashboardSlow` or a transform timeout.
- **S2**: raise `threads` if the dashboard does not slow down.
- **I2**: `docker update` needs cgroup memory limits. If Docker-in-Docker refuses, run I2 on a plain
  Docker host.

When `validate` shows which alerts each fault really produces, update `symptom_tags` in
`oncall/faults.py` to match.

## Expert agents over A2A

Each expert agent is a real A2A v1.0 server built on the official `a2a-sdk` (JSON-RPC over HTTP). It serves
its Agent Card at `/.well-known/agent-card.json`, takes an `InvestigationTask` as a JSON data part of a
message, and returns an A2A task (submitted -> working -> completed) whose artifact `proposal` holds the
typed `Proposal`. A message without a task is *rejected*; an agent crash *fails* the task.

```bash
pip install -e ".[dev,agents]"
python -m agents serve data/synthetic/dev/incidents.jsonl            # 9 investigators, ports 9301-9341
python -m agents ask data/synthetic/dev/incidents.jsonl --agent data-strong --incident syn-7-00007
```

Then let the controller run a whole incident through the agents (Block 5b, `controller/orchestrator.py`):

```bash
python -m controller deliberate data/synthetic/dev/incidents.jsonl --model models/logreg100.json     --calibration models/conformal.json --incident syn-7-00022
python -m oncall eval-team --data data/synthetic/dev100 --model models/logreg100.json     --calibration models/conformal.json            # the same rounds in-process, scored against ground truth
```

The rounds: the routed agents investigate in parallel (a failed agent is replaced by its fallback).
Every agent the classifier chose must hand in its best fix, and its verdict (owner / unsure / not mine)
says how strongly its own evidence backs it; an invented fix is still dropped by the agent's checks
(an LLM agent first gets one repair turn that shows it why). An LLM call that fails or times out fails
the agent's task, and the domain's fallback agent is asked instead.
A "not mine" pointing to another domain brings that domain in; if nobody claims ownership, the strongest
agent not yet asked is escalated to; finally every proposed fix is cross-checked by the other agents
(supports / contradicts / cannot tell, from their own evidence). Agents never message each other:
everything goes through the controller. `--may-abstain` restores the earlier design, where "not mine"
comes without a fix.

Block 5c then combines all the fixes into one plan (`controller/optimizer.py`): P(domain) from the classifier
moved by the agents' evidence, times P(fix | domain) from how often that kind of agent is right, lowered
for a best guess (agents' self-reported confidence is ignored), then the staged plan with minimum expected loss (risk of each
stage + the chance it is still broken x the cost of calling a human), or "escalate to a human" if that is
cheaper. If the best plan is less than 70% sure, a stronger agent gets a second look first. The policy
agent (`controller/policy.py`, served over A2A) approves, refuses, or asks for a human, and the result is
the `RemediationPlan` an MCP executor receives (`--json` prints it; see docs/HANDOVER.md). A plan the
policy would not let run alone (high risk, irreversible while unsure, P1 while unsure) can instead go to
a human for approval (plan kind `approve`: a person checks each stage before it runs), which is cheaper
than a person starting from zero when the agents' fix is likely right; `--no-approval` turns that off.
`eval-team --careless data-fast` makes one agent over-eager to test how well this contains it;
`eval-team --shy data-fast ...` makes agents find the fix and still say "not mine" (what the small LLMs
did), to compare `--may-abstain` with the default.

`ask` discovers the agent from its live card and sends the task over A2A. `controller/wire.py` holds the
wire format and converts every card to the SDK's protobuf `AgentCard` strictly, so the cards stay
spec-compliant (`tests/test_a2a_server.py`).

## Experience memory, learned trust and the history agent (Blocks 6a-6c)

After an incident is closed, `Controller.remember` stores one experience (`controller/memory.py`): the
incident's masked text and an embedding, what every agent proposed and every review said, the plan, and
the **verified** outcome. That outcome is the executor's report (which stages ran, healthy again or not)
plus, only when a person was involved, the operator's postmortem (owner, and whose fix was right). Each
fix is labelled right / wrong / unknown from that alone: a fix nobody tried while the controller fixed the
incident on its own stays unknown. The store is a NumPy matrix + JSON lines with cosine search over a
fixed hashing embedder, so nothing is downloaded or trained.

```bash
python -m oncall remember --data data/synthetic/dev100 --model models/logreg100.json \
    --calibration models/conformal.json --out models/memory      # train split, oldest first (~5 s)
python -m oncall eval-memory --memory models/memory --data data/synthetic/dev100 --model models/logreg100.json
python -m controller memory models/memory --incidents data/synthetic/dev100/incidents.jsonl --incident syn-8-00002
```

`remember` simulates the feedback from ground truth (`oncall/feedback.py`; `--feedback full` = an operator
reviews every incident, `--noise 0.2` = 20% of operator answers wrong) and checks every label against
ground truth. `eval-memory` also hides every past case of the query's own fault (leave-one-fault-out), to
show what the memory finds for a kind of incident it has never seen.

**Learning from it (Block 6b).** With a memory, the controller trusts each agent by its verified record
instead of fixed planning numbers (`controller/trust.py`, `controller/reliability.py`): how often its fix
was right when its domain owned the incident (on the whole memory and, more strongly, on past incidents at
least 0.85 alike), and how much its verdicts and reviews actually say. An agent with no record keeps the
planning numbers. The router uses this to choose agents, the optimizer to weigh their fixes, and "nobody
said *owner*" no longer forces an escalation: the plan's own confidence decides.

```bash
python -m oncall eval-team --data data/synthetic/dev100 --model models/logreg100.json \
    --calibration models/conformal.json --memory models/memory          # read only; memory from train
python -m oncall remember --data data/synthetic/dev100 --model models/logreg100.json \
    --calibration models/conformal.json --trust local --out models/memory-online   # learn as it goes
```

Rule agents, dev100 val (280), memory from train (840), trust only: agent cost per incident 9.89 -> 3.71
with the same loss (3.28) for honest agents; a careless data-fast 3.44 -> 3.29 loss; agents that find the
fix but say "not mine" 4.85 -> 3.39 loss (best possible 3.04). With 20% of operator answers wrong the loss
holds and cost rises (more second looks); at 40% on every incident it breaks down. Details in docs/PLAN.md.

**The history agent (Block 6c).** The memory lives behind its own Agent Card and A2A server
(`agents/history_agent.py`, role `history`, port 9361). The controller asks it once per incident, before
routing: it answers with the past incidents at least 0.85 alike (verified owner, the fix that worked, what
each agent said then) and every agent's verified record. The router uses the record; the experts see the
past incidents (an LLM agent in its prompt, a rule agent as "look at these tools first"), shown as kinds of
fix (`map_schema_fields on ingest (mapping)`), never with old ids; the optimizer counts precedent for each
fix (how often this kind of fix resolved similar incidents of that domain, never which domain owns it);
the plan lists the precedents (`RemediationPlan.precedents`, contract 1.3). After the incident, the
controller sends the verified outcome back.

```bash
python -m agents serve data/synthetic/dev/incidents.jsonl --memory models/memory   # experts + policy + history
python -m controller deliberate data/synthetic/dev/incidents.jsonl --model models/logreg100.json \
    --calibration models/conformal.json --incident syn-7-00046                   # asks the history agent first
python -m oncall eval-team --data data/synthetic/dev100 --model models/logreg100.json \
    --calibration models/conformal.json --memory models/memory  [--trust off] [--no-past-cases] [--no-precedent]
```

**In PostgreSQL (Block 6d).** Every command that takes a memory folder also takes `postgres`: the
database whose URL is in the `A2A_MEMORY_DB` environment variable (keep the password there, never in git).
Records go into tables (`controller/memory_schema.sql`: experiences, proposals, reviews, plus the views
`agent_fix_record` and `reviewer_record`), each incident in one transaction, and several programs can share
the memory (each re-reads when another wrote). "Find similar" stays an exact search in Python until
pgvector is installed. Needs `pip install -e ".[db]"`; on Windows the driver uses the libpq of the
PostgreSQL install (Windows Application Control may block the DLL in `psycopg[binary]`).

```powershell
createdb -U postgres a2a_memory                                       # a database of its own
$env:A2A_MEMORY_DB = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/a2a_memory"
python -m oncall remember --data data/synthetic/dev100 --model models/logreg100.json `
    --calibration models/conformal.json --out postgres                  # creates the tables if needed
python -m agents serve data/synthetic/dev/incidents.jsonl --memory postgres   # the history agent on PostgreSQL
python -m controller memory postgres --export-sql models/memory.sql     # or: rows for psql, no driver needed
```

Database tests run only with `A2A_TEST_DB` set to a throwaway database (they wipe it).

`agents serve --memory FOLDER` saves the store after every remembered incident. Rule agents, dev100 val: honest
agent cost 9.89 (no history) -> 3.71 (trust) -> 2.88 (trust + past cases + precedent) at loss 3.28-3.29,
escalations 220 -> 2; learning as it goes from an empty memory, 2.2-2.5 per incident. On fault types the
memory never saw, it finds almost nothing similar (0.28 per incident) and nothing wrong runs.

## Layout

```
docker-compose.yml        the platform under test
sandbox/                  service code: orders_api, ingest, transform (dbt), metrics_api, toolbox, warehouse
runtime/defaults/         deployed (known-good) service configs; live copies are written to runtime/
oncall/                   lab: faults, actions, oracle, incidents, episodes, CLI
controller/               the intelligent controller, built block by block (never imports oncall/)
agents/                   expert agents: evidence tools, reasoners (rules / Ollama), A2A servers (agents/server.py)
agents/cards/             A2A Agent Cards of the default agent pool (python -m controller agents --export agents/cards)
samples/                  hand-written incidents for checking controller intake (not training data)
tests/                    unit tests for the pure logic
docs/PLAN.md              phases, exit criteria, design decisions
docs/HANDOVER.md          output contract for an executor (MCP / Microsoft Fabric) + Fabric mapping
```
