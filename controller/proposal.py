"""What the orchestrator sends an expert agent, and what the agent sends back.

    InvestigationTask(role primary|consult) -> agent -> Proposal   "what is wrong in my domain, and the fix"
    InvestigationTask(role review, claim=P) -> agent -> Review     "does my evidence agree with P?"

Over A2A these travel as JSON data parts of a message (task in) and of an artifact (out): artifact
"proposal" or "review". Block 5b collects proposals and cross-checks them; Block 5c picks the plan
with minimum expected loss.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .contracts import IncidentReport
from .plan import PlannedAction


class Finding(BaseModel):
    tool: str
    fact: str  # one observed fact, in words (what an engineer would note)


class Proposal(BaseModel):
    incident_id: str
    agent_id: str
    domain: str
    # owner: "the root cause is in my domain, here is the fix"
    # not_mine: "I checked my domain; it is healthy, or the cause is elsewhere"
    # unsure: "I could not check properly" (missing evidence, tool failure)
    # When the task says must_propose, every verdict comes with the agent's best fix: the verdict then says
    # how strongly the agent's own evidence backs that fix, and the optimizer weighs it accordingly.
    verdict: Literal["owner", "not_mine", "unsure"]
    root_cause: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    findings: list[Finding] = Field(default_factory=list)
    actions: list[PlannedAction] = Field(default_factory=list)
    points_to: str | None = None  # a domain the evidence suggests instead, if any
    tools_used: list[str] = Field(default_factory=list)
    reasoner: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)  # e.g. tokens and seconds for LLM reasoners


class InvestigationTask(BaseModel):
    incident: IncidentReport  # shared evidence only: alerts, logs, job runs
    domain: str  # the domain this agent is asked to examine (for a review: the evidence it checks with)
    role: Literal["primary", "consult", "review"] = "primary"
    question: str | None = None  # free text, e.g. why the controller asks
    claim: Proposal | None = None  # role "review": another agent's ownership claim to check
    # The classifier chose this agent, so it must hand in its best fix for its domain even when its
    # evidence is weak (it still says how weak, in its verdict). Never an invented fix: the agent's
    # executable + grounded checks still drop those. False: "not mine" may come without a fix.
    must_propose: bool = False


class Review(BaseModel):
    """One agent's opinion of another agent's ownership claim, from its OWN domain's evidence."""

    incident_id: str
    reviewer: str  # agent id
    domain: str  # the domain whose evidence the reviewer used
    claim_agent: str
    claim_domain: str
    # supports:    my evidence agrees (e.g. "the CPU is used by database sessions" for a data claim)
    # contradicts: my evidence shows a different root cause, or shows the claim is wrong
    # cannot_tell: my evidence says nothing either way (my domain is healthy, or I could not check)
    stance: Literal["supports", "contradicts", "cannot_tell"]
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tools_used: list[str] = Field(default_factory=list)
    reasoner: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
