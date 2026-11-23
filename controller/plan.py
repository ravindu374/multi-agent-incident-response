"""The controller's output contract: a RemediationPlan out, an ExecutionReport back.

The controller never executes anything. It proposes platform-neutral *intents*
("stop_session", "rotate_credential", ...). An executor maps each intent onto its platform:
the lab maps intents to sandbox actions (oncall/intents.py); on Microsoft Fabric an MCP-based
executor would map them to Fabric REST operations. The executor reports back what happened,
and only that verified report feeds the learner (Block 6). See docs/HANDOVER.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from . import DOMAINS

SCHEMA_VERSION = "1.3"  # 1.1: staged actions, plan kind, options considered; 1.2: kind "approve";
#                        1.3: precedents (similar past incidents from the history agent)
Risk = Literal["low", "medium", "high"]
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# The platform's components: the only valid action targets. (A Fabric deployment would list its
# workspace items here: lakehouse, pipelines, notebooks, semantic model, ...)
COMPONENTS = ("orders-api", "ingest", "warehouse", "transform", "metrics-api", "postgres-exporter", "prometheus")

# Every action parameter an intent may take, as JSON Schema. This is part of the output contract: an
# executor (the lab, or an MCP server on Fabric) receives exactly these names and types.
PARAM_SCHEMAS: dict[str, dict] = {
    "pid": {"type": "integer"},
    "pids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
    "application_name": {"type": "string", "minLength": 1},
    "usename": {"type": "string", "minLength": 1},  # PostgreSQL's spelling of the session's user
    "client_addr": {"type": "string", "minLength": 1},
    "exclude_application_name": {"type": "string", "minLength": 1},
    "ip": {"type": "string", "minLength": 1},
    "role": {"type": "string", "minLength": 1},
    "mount": {"type": "string", "minLength": 1},
    # {field the source sends now: field the pipeline expects}, e.g. {"total_amount": "amount"}
    "mapping": {"type": "object", "additionalProperties": {"type": "string"}},
    "route": {"type": "string", "enum": ["source", "warehouse"]},  # which connection of the ingest job
    "endpoint": {"type": "string", "enum": ["primary", "secondary"]},
    "memory_mb": {"type": "integer", "minimum": 64},
    "tests": {"type": "array", "items": {"type": "string"}},
}


def param_shape(name: str) -> str:
    """How a param looks in JSON, for prompts and docs: "pid": <integer>, "route": "source"|"warehouse"."""
    schema = PARAM_SCHEMAS[name]
    if "enum" in schema:
        value = "|".join(f'"{v}"' for v in schema["enum"])
    elif schema["type"] == "integer":
        value = "<integer>"
    elif schema["type"] == "array":
        item = "<integer>" if schema["items"]["type"] == "integer" else '"<text>"'
        value = f"[{item}, ...]"
    elif schema["type"] == "object":
        # an example, not a placeholder: llama3 once copied "<field the source sends now>" verbatim as the
        # key. The example's fields exist in no incident, so a copied example fails the located check.
        value = ('{"cust_ref": "customer_id"} (from the name the source sends now to the name the pipeline '
                 'expects; this example is for a source that renamed customer_id to cust_ref)')
    else:
        value = '"<text>"'
    return f'"{name}": {value}'


class IntentSpec(BaseModel):
    name: str
    risk: Risk  # minimum risk; a plan may raise it, never lower it
    summary: str
    # What an executor must be told: each group lists alternative params, at least one per group must
    # be given. E.g. stop_session needs *which* session: a pid, an application name, a user, ...
    needs: tuple[tuple[str, ...], ...] = ()
    optional: tuple[str, ...] = ()  # extra params it may take
    targets: tuple[str, ...] = COMPONENTS  # components it can act on (sessions live in the warehouse, ...)
    reversible: bool = True  # False: cannot be undone (a stopped session's work is lost), costs more risk

    @model_validator(mode="after")
    def _check(self) -> IntentSpec:
        unknown = {k for g in self.needs for k in g} | set(self.optional)
        unknown -= set(PARAM_SCHEMAS)
        if unknown or not set(self.targets) <= set(COMPONENTS):
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)} or targets {self.targets}")
        return self

    def alternatives(self) -> list[tuple[str, ...]]:
        """Every complete way to say what the action acts on: one param from each group."""
        return [tuple(c) for c in product(*self.needs)]

    def needs_text(self) -> str:
        return " or ".join("{" + ", ".join(param_shape(k) for k in alt) + "}" for alt in self.alternatives()
                           if alt) or "{} (the target is enough)"


INTENTS: dict[str, IntentSpec] = {s.name: s for s in [
    IntentSpec(name="reroute_connection", risk="low",
               summary="Send a job's traffic to a source or sink through another network path or gateway.",
               needs=(("route",), ("endpoint",)), targets=("ingest",)),
    IntentSpec(name="restore_config", risk="medium",
               summary="Roll a component back to its last known-good configuration or deployment."),
    IntentSpec(name="map_schema_fields", risk="high",
               summary="Map renamed source fields onto the target schema (changes what the data means).",
               needs=(("mapping",),), targets=("ingest",)),
    IntentSpec(name="quarantine_records", risk="low",
               summary="Move records that fail data-quality rules aside so downstream jobs can continue.",
               optional=("tests",), targets=("warehouse",)),
    IntentSpec(name="stop_session", risk="medium",
               summary="End a database session that blocks, leaks connections or runs away with resources.",
               needs=(("pid", "pids", "application_name", "usename", "client_addr"),),
               optional=("exclude_application_name",), targets=("warehouse",), reversible=False),
    IntentSpec(name="stop_process", risk="medium",
               summary="Stop an unknown or runaway process or job that is consuming compute.",
               needs=(("pids", "pid"),), reversible=False),
    IntentSpec(name="resize_compute", risk="medium",
               summary="Change the memory or compute available to a component.", needs=(("memory_mb",),)),
    IntentSpec(name="free_storage", risk="medium",
               summary="Delete temporary or staging files that have filled a component's storage.",
               optional=("mount",), reversible=False),
    IntentSpec(name="restart_component", risk="medium",
               summary="Restart a component or re-run its job."),
    IntentSpec(name="rotate_credential", risk="high",
               summary="Replace a credential that is being misused and update its legitimate users.",
               needs=(("role",),), targets=("warehouse",)),
    IntentSpec(name="block_client", risk="medium",
               summary="Block a misbehaving client (address or identity) from a component.", needs=(("ip",),),
               targets=("metrics-api", "warehouse", "orders-api")),
]}


class PlannedAction(BaseModel):
    intent: str
    target: str  # the component acted on, e.g. "ingest", "warehouse", "metrics-api"
    params: dict[str, Any] = Field(default_factory=dict)
    risk: Risk
    rationale: str
    # Execution stage. The executor runs stage 1, re-checks the platform's health, and runs stage 2
    # only if the incident is still open, and so on. Actions of one stage run together.
    stage: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_intent(self) -> PlannedAction:
        spec = INTENTS.get(self.intent)
        if spec is None:
            raise ValueError(f"unknown intent {self.intent!r}; allowed: {sorted(INTENTS)}")
        if RISK_ORDER[self.risk] < RISK_ORDER[spec.risk]:
            raise ValueError(f"risk of {self.intent} cannot be below {spec.risk}")
        return self


class PolicyDecision(BaseModel):
    approved: bool
    needs_human: bool
    reasons: list[str] = Field(default_factory=list)


class PlanOption(BaseModel):
    """One plan the optimizer considered, so a human can see why the chosen one won."""

    label: str
    actions: list[str] = Field(default_factory=list)  # "stage 1: stop_session on warehouse {'pid': 6079}"
    p_fixed: float = Field(ge=0.0, le=1.0)  # estimated chance the incident is fixed by the end of the plan
    expected_loss: float
    chosen: bool = False


class Precedent(BaseModel):
    """A similar past incident behind the plan (from the history agent), so a human sees why it is trusted."""

    incident_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    owner: str | None = None  # verified owner of that incident
    fix: list[str] = Field(default_factory=list)  # the kind of fix that resolved it, e.g. "stop_session on warehouse (pid)"
    verified_by: str = ""  # "executor" (health checks) or "operator" (postmortem)


class RemediationPlan(BaseModel):
    """Controller -> executor. One plan per incident."""

    schema_version: str = SCHEMA_VERSION
    incident_id: str
    created_at: str = Field(default_factory=_now)
    # remediate: carry out `actions` stage by stage, automatically (after policy approval)
    # approve:   the same stages, but a human says yes to each stage before it runs; a rejected stage is
    #            skipped, and if no stage is approved (or none fixes it) the human takes over. For fixes
    #            the policy agent will not let run alone (high risk, irreversible + unsure, ...).
    # escalate:  do nothing; a human takes over from zero (`options` lists what agents suggested)
    kind: Literal["remediate", "approve", "escalate"] = "remediate"
    owner: str  # domain that owns the root cause (for escalate: the most likely owner)
    team: list[str]  # domains that worked on it, owner first
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    actions: list[PlannedAction] = Field(default_factory=list)
    expected_loss: float | None = None  # of the chosen option (see controller/optimizer.py)
    options: list[PlanOption] = Field(default_factory=list)
    precedents: list[Precedent] = Field(default_factory=list)  # similar past incidents (empty: none, or no history)
    policy: PolicyDecision | None = None  # set by the policy agent before hand-over

    @model_validator(mode="after")
    def _check_domains(self) -> RemediationPlan:
        if self.owner not in DOMAINS:
            raise ValueError(f"owner must be one of {DOMAINS}")
        if not self.team or self.team[0] != self.owner or not set(self.team) <= set(DOMAINS):
            raise ValueError("team must list known domains, owner first")
        if self.kind in ("remediate", "approve") and not self.actions:
            raise ValueError(f"a {self.kind} plan needs at least one action")
        if self.kind == "escalate" and self.actions:
            raise ValueError("an escalate plan carries no actions (they are listed in options)")
        stages = sorted({a.stage for a in self.actions})
        if stages != list(range(1, len(stages) + 1)):
            raise ValueError(f"stages must be numbered 1, 2, ... without gaps, got {stages}")
        return self

    def stages(self) -> list[list[PlannedAction]]:
        return [[a for a in self.actions if a.stage == n] for n in sorted({a.stage for a in self.actions})]


class ActionOutcome(BaseModel):
    intent: str
    status: Literal["done", "failed", "skipped", "rejected"]
    detail: str = ""


class ExecutionReport(BaseModel):
    """Executor -> controller. `resolved` must come from the platform's own health checks."""

    schema_version: str = SCHEMA_VERSION
    incident_id: str
    outcomes: list[ActionOutcome]
    resolved: bool | None = None  # None = not verified yet
    verified_at: str | None = None
    notes: str = ""
