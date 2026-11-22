"""Block 6b - How much to trust each agent, from its verified record (counts + the arithmetic on them).

A TrustTable holds, per agent, counts of verified cases from the experience memory (controller/memory.py):
over the whole memory ("global") and over the past incidents most like this one ("near"). It travels with
the RoutingDecision, so the router, the orchestrator and the optimizer all read the same numbers.
controller/reliability.py fills it.

Every number starts from the planning guess it replaces and moves towards what was measured, the more
cases there are (Bayesian shrinkage: the guess counts as `prior_strength` imaginary cases):
    over the whole memory   global = (hits + prior_strength x guess) / (cases + prior_strength)
    on incidents like this  near   = (near hits + near_strength x global) / (near cases + near_strength)
So an agent with no record keeps the planning guess (a new agent, an empty memory), and an agent that is
good at one kind of incident and bad at another is trusted for the first and not for the second.

Evidence (whose domain owns it; does a review say the fix is right) is a log-likelihood ratio, measured over
the whole memory only (near incidents almost all have the same owner, so they cannot show how an agent
behaves when its domain is NOT the owner):
    llr = log P(this verdict | its domain owned it) / P(this verdict | it did not)   (+1 smoothing, capped)
blended with the planning value by min(cases owned, cases not owned) / (that + prior_strength).
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

VERDICTS = ("owner", "unsure", "not_mine")
STANCES = ("supports", "contradicts", "cannot_tell")
LLR_CAP = 4.0  # never let one agent's word move the odds by more than ~e^4 = 55x


class Counts(BaseModel):
    hits: int = 0
    n: int = 0


class AgentRecord(BaseModel):
    """Verified counts for one agent."""

    # in the owner's domain, did it hand in a right fix? (what the router means by "succeeds")
    success: Counts = Field(default_factory=Counts)
    # in the owner's domain, by verdict: was the fix right? (the optimizer's P(fix | domain))
    fix: dict[str, Counts] = Field(default_factory=dict)
    # how often it gave each verdict when its domain owned the incident, and when it did not
    verdict_owned: dict[str, int] = Field(default_factory=dict)
    verdict_not_owned: dict[str, int] = Field(default_factory=dict)
    # its reviews, keyed "same:supports" / "other:contradicts" ...: was the reviewed fix right (same
    # domain) or its domain the owner (other domain)?
    review_true: dict[str, int] = Field(default_factory=dict)
    review_false: dict[str, int] = Field(default_factory=dict)


def shrink(hits: float, n: float, prior: float, strength: float) -> float:
    return (hits + strength * prior) / (n + strength)


def smoothed_llr(when_true: dict[str, int], when_false: dict[str, int], key: str, keys: tuple[str, ...],
                 prior: float, strength: float) -> float:
    a, b = sum(when_true.get(k, 0) for k in keys), sum(when_false.get(k, 0) for k in keys)
    if not a or not b:  # never seen both sides: nothing to compare
        return prior
    measured = (math.log((when_true.get(key, 0) + 1) / (a + len(keys)))
                - math.log((when_false.get(key, 0) + 1) / (b + len(keys))))
    w = min(a, b) / (min(a, b) + strength)
    return max(-LLR_CAP, min(LLR_CAP, w * measured + (1 - w) * prior))


class TrustTable(BaseModel):
    mode: Literal["global", "local"] = "local"
    min_similarity: float = 0.85
    neighbours: int = 0  # past incidents this similar that were used ("near")
    prior_strength: float = 4.0
    near_strength: float = 2.0
    agents: dict[str, AgentRecord] = Field(default_factory=dict)  # over the whole memory
    near: dict[str, AgentRecord] = Field(default_factory=dict)  # over the similar past incidents

    def _estimate(self, agent_id: str, pick, prior: float) -> float:
        g = pick(self.agents.get(agent_id)) or Counts()
        value = shrink(g.hits, g.n, prior, self.prior_strength)
        if self.mode == "local":
            n = pick(self.near.get(agent_id)) or Counts()
            value = shrink(n.hits, n.n, value, self.near_strength)
        return value

    def success(self, agent_id: str, prior: float) -> float:
        return self._estimate(agent_id, lambda r: r.success if r else None, prior)

    def fix(self, agent_id: str, verdict: str, prior: float) -> float:
        return self._estimate(agent_id, lambda r: r.fix.get(verdict) if r else None, prior)

    def verdict_llr(self, agent_id: str, verdict: str, prior: float) -> float:
        r = self.agents.get(agent_id)
        if r is None:
            return prior
        return smoothed_llr(r.verdict_owned, r.verdict_not_owned, verdict, VERDICTS, prior, self.prior_strength)

    def review_llr(self, reviewer: str, same_domain: bool, stance: str, prior: float) -> float:
        r = self.agents.get(reviewer)
        if r is None:
            return prior
        side = "same" if same_domain else "other"
        return smoothed_llr(r.review_true, r.review_false, f"{side}:{stance}",
                            tuple(f"{side}:{s}" for s in STANCES), prior, self.prior_strength)

    def cases(self, agent_id: str) -> tuple[int, int]:
        """(verified cases over the whole memory, on similar incidents) behind this agent's numbers."""
        g, n = self.agents.get(agent_id), self.near.get(agent_id)
        return (g.success.n if g else 0), (n.success.n if n else 0)
