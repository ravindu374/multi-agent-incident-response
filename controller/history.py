"""Block 6c - What the controller and the history agent say to each other (over A2A or in-process).

    HistoryQuery (the new incident)   -> history agent -> artifact "history": HistoryReport
         similar past incidents (only those at least `min_similarity` alike: Block 6a showed that a less
         similar "nearest" incident usually belongs to another domain), each with its verified owner, the
         fix that worked and what every agent said then; plus every agent's verified record (TrustTable,
         Block 6b)
    RememberRequest (a closed incident, verified) -> history agent -> artifact "remembered": RememberAck

Past fixes are shown as KINDS of fix (intent on target, which params): a pid or an address from last week
means nothing today, so the agents take today's values from their own tools.

How the history is used (one lookup per incident):
  router        trusts each agent by its record (TrustTable)
  experts       see the similar past cases: an LLM agent in its prompt, a rule agent as "look here first"
  optimizer     counts precedent for each proposed fix (precedent(): within the fix's domain, how often
                this kind of fix resolved the similar past incidents, vs another kind); never for WHICH
                domain owns it, which the classifier already learned from the same kind of incidents
  plan          lists the precedents, for the human approver and the executor
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from .contracts import IncidentReport
from .plan import PlannedAction
from .trust import TrustTable

PRECEDENT_CAP = 3.0  # precedent moves a fix's odds by at most e^3 = 20x


class PastAnswer(BaseModel):
    agent_id: str
    domain: str
    verdict: str
    fix: list[str] = Field(default_factory=list)  # kinds of fix, e.g. "stop_session on warehouse (pid)"
    label: str | None = None  # right / wrong / unknown, from the verified feedback


class PastCase(BaseModel):
    incident_id: str
    similarity: float
    opened_at: str | None = None
    alerts: list[str] = Field(default_factory=list)
    owner: str | None = None  # verified owner
    fix_that_worked: list[PlannedAction] = Field(default_factory=list)  # that incident's own values
    outcome: str = ""  # who verified it: "executor" (health checks) or "operator" (postmortem)
    answers: list[PastAnswer] = Field(default_factory=list)


class HistoryQuery(BaseModel):
    incident: IncidentReport
    k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    before: str | None = None  # only incidents opened earlier (replaying the past without seeing the future)


class HistoryReport(BaseModel):
    incident_id: str
    memory_size: int = 0
    min_similarity: float = 0.85
    cases: list[PastCase] = Field(default_factory=list)
    trust: TrustTable | None = None


class RememberRequest(BaseModel):
    experience: dict[str, Any]  # a controller.memory.Experience as JSON (validated by the history agent)


class RememberAck(BaseModel):
    incident_id: str
    stored: bool
    memory_size: int


def fix_kind(a: PlannedAction) -> tuple[str, str]:
    return a.intent, a.target


def describe_fix(actions: list[PlannedAction]) -> list[str]:
    """Kinds of fix without the values: "stop_session on warehouse (pid)"."""
    return [f"{a.intent} on {a.target}" + (f" ({', '.join(sorted(a.params))})" if a.params else "") for a in actions]


def precedent(cases: list[PastCase], domain: str, actions: list[PlannedAction]) -> tuple[int, int, float]:
    """Among the similar past incidents owned by `domain` and fixed by a recorded fix: how many were fixed by
    this kind of fix (same intent on the same target), how many by another kind, and the log-odds that gives:
    log((same + 1) / (other + 1)), capped. (0, 0, 0.0) when the history says nothing."""
    kinds = {fix_kind(a) for a in actions}
    same = other = 0
    for c in cases:
        if c.owner != domain or not c.fix_that_worked:
            continue
        if kinds & {fix_kind(a) for a in c.fix_that_worked}:
            same += 1
        else:
            other += 1
    if not same and not other:
        return 0, 0, 0.0
    llr = math.log((same + 1) / (other + 1))
    return same, other, max(-PRECEDENT_CAP, min(PRECEDENT_CAP, llr))
