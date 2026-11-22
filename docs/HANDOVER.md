# Handover: from the controller to an executor (MCP / Microsoft Fabric)

The controller diagnoses incidents and **proposes** remediation. It never executes anything.
An executor (for example an MCP-based service in front of Microsoft Fabric) carries the plan out
and reports back. This page is the contract between the two sides.

```
platform logs/alerts -> controller + A2A expert agents -> RemediationPlan -> executor (MCP) -> platform
                                ^                                                |
                                +----------------- ExecutionReport --------------+
```

## 1. What the executor receives: `RemediationPlan`

JSON Schema: [`contract/remediation_plan.schema.json`](contract/remediation_plan.schema.json)
(generated from `controller/plan.py` with `python -m controller schema`).

```json
{
  "schema_version": "1.2",
  "incident_id": "syn-7-00022",
  "created_at": "2026-11-20T10:16:02+00:00",
  "kind": "remediate",
  "owner": "data",
  "team": ["data", "infra"],
  "root_cause": "3 long-running query session(s) from jupyter-notebook are overloading the warehouse",
  "confidence": 0.94,
  "evidence": ["[data] P(data)=0.97 [classifier 0.45; data-fast claims it; infra-fast (infra) supports], ..."],
  "actions": [
    {"intent": "stop_session", "target": "warehouse", "params": {"pids": [7081, 5118, 2127]},
     "risk": "medium", "rationale": "end the runaway queries", "stage": 1}
  ],
  "expected_loss": 5.78,
  "options": [
    {"label": "escalate to a human now", "actions": [], "p_fixed": 0.0, "expected_loss": 8.5, "chosen": false},
    {"label": "fix data (data-fast, data-strong)", "actions": ["stage 1: stop_session on warehouse ..."],
     "p_fixed": 0.85, "expected_loss": 5.78, "chosen": true}
  ],
  "policy": {"approved": true, "needs_human": false,
             "reasons": ["low/medium risk, reversible or confident: may run automatically"]}
}
```

Rules for the executor:

- Run nothing unless `policy.approved` is true. If `policy.needs_human` is true, get an operator's approval first.
- `kind` says who acts:
  - `"remediate"`: run the stages on your own (the policy agent found nothing that needs a person).
  - `"approve"`: the same stages, but **ask an operator before each stage**: show the stage, its
    `rationale`, `root_cause` and `evidence`. A rejected stage does not run (report it as `rejected`) and
    the next stage is offered; if no stage is approved, or none fixes it, the incident goes to the
    operator. The controller chooses this for fixes the policy agent will not let run alone (high risk,
    irreversible while the plan is unsure, P1 while unsure) when a person checking a ready fix is cheaper
    than a person starting from zero. The operator's yes / no is verified feedback too (Block 6).
- `kind: "escalate"` means: do nothing automatically; page a human. `options` lists what the agents suggested
  and why acting looked worse than asking a person.
- Run the plan **stage by stage**: run every action of stage 1, re-check the platform's health, and run
  stage 2 only if the incident is still open, and so on. Stop as soon as it is resolved. (The optimizer
  orders stages so that the most likely fix per unit of risk goes first.)
- Refuse any `intent` you do not implement, and report it as `rejected`.
- `risk` is a floor set by the controller. The executor may treat an action as riskier, never as safer.
- Every parameter name and type is fixed by `PARAM_SCHEMAS` in `controller/plan.py` (e.g. `stop_session`
  takes `pid` | `pids` | `application_name` | `usename` | `client_addr`).

## 2. What the executor sends back: `ExecutionReport`

JSON Schema: [`contract/execution_report.schema.json`](contract/execution_report.schema.json)

```json
{"schema_version": "1.0", "incident_id": "inc-20261119-101500-a3f2",
 "outcomes": [{"intent": "stop_process", "status": "done", "detail": "killed pids 41, 42"}],
 "resolved": true, "verified_at": "2026-11-19T10:22:40+00:00"}
```

`resolved` must come from the platform's own health checks (job succeeded, data fresh again).
It must not come from an LLM's opinion. The controller learns only from verified reports.

Report **one outcome per action of the plan, in the plan's order** (`skipped` for stages that did not run
because the incident was already fixed, `rejected` for stages an operator said no to). The controller reads
the last stage that ran as the fix that resolved it, and earlier stages as fixes that did not
(`Feedback.from_report` in `controller/memory.py`). When a person was involved (an approval, an escalation,
or a plan that did not fix it), an optional postmortem helps the controller learn: the owning domain, and
for each agent's suggested fix (listed in `options`) whether it would have fixed the incident
(`Feedback.owner` and `Feedback.credits`). Without it, fixes that were never tried stay "unknown".

## 3. Intents and how each platform implements them

The controller speaks in platform-neutral intents. The Fabric column is an indicative mapping:
confirm each operation in the Fabric REST / MCP documentation before implementing it.

| intent | min risk | sandbox action (implemented) | Microsoft Fabric counterpart (indicative) |
|---|---|---|---|
| `reroute_connection` | low | `switch_source_endpoint`, `switch_warehouse_endpoint` | point the pipeline's connection at another gateway / connection |
| `restore_config` | medium | `rollback_service_config` | redeploy the previous item version (Git integration / deployment pipeline) |
| `map_schema_fields` | high | `set_field_mapping` | update the Copy activity / Dataflow Gen2 column mapping. `mapping` goes **from the field the source sends now to the field the pipeline expects**: `{"total_amount": "amount"}` |
| `quarantine_records` | low | `quarantine_rows` | run a notebook or T-SQL that moves failing rows to a quarantine table |
| `stop_session` | medium | `terminate_backend`, `terminate_sessions` | kill the Warehouse session, or cancel the job instance |
| `stop_process` | medium | `kill_container_process` | cancel the job instance that is consuming capacity. Never targets the component itself: pid 1 or the warehouse's own `postgres` processes are rejected before a plan is made (a runaway query is `stop_session`) |
| `resize_compute` | medium | `set_memory_limit` | resize the Spark environment, or scale the capacity SKU |
| `free_storage` | medium | `purge_spool` | delete staging files in the Lakehouse `Files` area |
| `restart_component` | medium | `restart_service` | re-run the item job |
| `rotate_credential` | high | `rotate_db_credential` | update the connection's credentials / rotate the service principal secret |
| `block_client` | medium | `block_ip` | remove the principal's workspace role, or tighten network rules |

## 4. The sandbox as a miniature Fabric-style platform

Development and evaluation run on a local sandbox because faults can be injected on purpose,
each incident has a known correct answer, and nothing costs money. Each part corresponds to a
Fabric component:

| sandbox | Fabric counterpart |
|---|---|
| `orders-api` (source system) | external source reached through a data gateway |
| Toxiproxy network paths | gateway / network path to the source and the warehouse |
| `ingest` job | Data Factory pipeline with a Copy activity |
| `warehouse` (Postgres) | Fabric Warehouse / Lakehouse SQL analytics endpoint |
| `transform` (dbt build + tests) | notebooks / dbt jobs / Dataflow Gen2 with data-quality checks |
| `metrics-api` | Power BI semantic model and report |
| container CPU / memory | Fabric capacity (CU) and Spark compute |
| `ops.job_runs` | Monitoring hub / job instances |
| Prometheus | workspace monitoring / Capacity Metrics app |

The four domains in Fabric terms: **network** = gateways and connectivity, **data** = pipelines, schemas and
data quality, **infra** = capacity and compute, **security** = identities, credentials and access.

## 5. Fabric's own MCP servers

Microsoft publishes official Fabric MCP servers. According to Microsoft's 2026 announcement, the local
server is generally available, and the remote *Core* server is in public preview. Core exposes Fabric
REST operations as typed tools and signs in through Microsoft Entra ID. An executor for this
project would most likely wrap Core, translating intents into those tools. The local server
packages API specifications for AI-assisted development and does not touch a live workspace.
Check Microsoft Learn for the current status before building on either.

Sources: [Fabric MCP announcement](https://blog.fabric.microsoft.com/zh-tw/blog/introducing-fabric-mcp-public-preview),
[Fabric platform MCP update](https://community.fabric.microsoft.com/t5/Fabric-Updates-Blog/Empowering-admins-and-developers-with-a-Fabric-platform-ready/ba-p/5172252),
[Microsoft MCP server listing](https://github.com/mcp/com.microsoft/microsoft-fabric).
