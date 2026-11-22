"""Block 5b - Orchestrator: ask the routed agents, follow their pointers, cross-check their fixes.

    round 1   investigate: every agent the router chose works on its own domain at the same time (one
              A2A task each). An agent that fails, or is no longer allowed (down, busy, breaker open), is
              replaced by its next fallback. By default every chosen agent MUST hand in its best fix
              (must_propose), saying in its verdict how strongly its evidence backs it; the optimizer
              (Block 5c) weighs and combines all of them. must_propose=False is the earlier design, where
              "not mine" comes without a fix (kept as a baseline).
    pointers  an agent that answers "not mine" may point to another domain. If nobody was asked there,
              the controller asks that domain's best agent (ownership transfer, at most
              max_extra_domains per incident).
    escalate  if still nobody claims ownership, the strongest agent not yet asked looks at the most
              likely domain (cheap agents first, a senior only when they find nothing).
    round 2   cross-check: every proposed fix goes to the other agents that took part. Each answers
              supports / contradicts / cannot_tell from its OWN evidence (A2A review tasks).

Agents never message each other directly: every task goes through the controller, so each delegation is
visible, budgeted and logged (docs/PLAN.md). The output is a Deliberation: all proposals, reviews and
calls, which Block 5c turns into one plan with minimum expected loss.

    AgentLink  how the controller reaches agents: A2ALink (real A2A over HTTP) or an in-process link
               (agents/local.py) for tests and evaluation. The rounds are the same either way.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from .cards import WELL_KNOWN_CARD_PATH, AgentCard
from .checks import CHECK
from .contracts import IncidentReport
from .plan import PolicyDecision
from .policy import PolicyRequest
from .proposal import InvestigationTask, Proposal, Review
from .router import AgentChoice, Router, RoutingDecision


class AgentLink(Protocol):
    async def investigate(self, card: AgentCard, task: InvestigationTask) -> Proposal: ...

    async def review(self, card: AgentCard, task: InvestigationTask) -> Review: ...

    async def review_plan(self, card: AgentCard, request: PolicyRequest) -> PolicyDecision: ...


class A2ALink:
    """Reach agents over A2A (JSON-RPC over HTTP) through one shared connection pool."""

    def __init__(self, http: httpx.AsyncClient | None = None, timeout: float = 180.0):
        self.http = http or httpx.AsyncClient(timeout=timeout)

    async def investigate(self, card: AgentCard, task: InvestigationTask) -> Proposal:
        from .wire import ask  # needs a2a-sdk; imported here so the controller core works without it

        return await ask(card, task, self.http)

    async def review(self, card: AgentCard, task: InvestigationTask) -> Review:
        from .wire import ask_review

        return await ask_review(card, task, self.http)

    async def review_plan(self, card: AgentCard, request: PolicyRequest) -> PolicyDecision:
        from .wire import ask_policy

        return await ask_policy(card, request, self.http)

    async def aclose(self) -> None:
        await self.http.aclose()


async def probe_agents(registry, http: httpx.AsyncClient, now: float | None = None) -> dict[str, bool]:
    """A2A discovery as a liveness check: fetch every agent's card; alive -> heartbeat, else mark down."""
    async def one(agent_id: str) -> bool:
        u = urlparse(registry.card(agent_id).supported_interfaces[0].url)
        try:
            r = await http.get(f"{u.scheme}://{u.netloc}{WELL_KNOWN_CARD_PATH}", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    ids = registry.agents()
    alive = dict(zip(ids, await asyncio.gather(*(one(a) for a in ids))))
    for agent_id, ok in alive.items():
        if ok:
            registry.heartbeat(agent_id, now=now)
        else:
            registry.mark_down(agent_id)
    return alive


@dataclass
class OrchestratorPolicy:
    # "always": review every claim; "contested": only when there are 2+ claims or a claim below 0.8
    # confidence; "never": no cross-check (a baseline)
    cross_check: Literal["always", "contested", "never"] = "always"
    # every agent the classifier chose must hand in its best fix (False: "not mine" may come without one)
    must_propose: bool = True
    follow_pointers: bool = True
    max_extra_domains: int = 1
    # nobody claimed after round 1 + pointers: ask the strongest agent not yet asked for the most likely
    # domain(s) before giving up ("the junior found nothing, ask a senior")
    escalate: bool = True
    # the optimizer's best plan is less sure than this: ask a stronger agent for a second look, re-plan
    second_look_below: float = 0.7
    max_escalations: int = 1  # extra senior calls per incident (escalation and second look share it)
    max_attempts: int = 2  # calls per domain: the chosen agent, then fallbacks
    transfer_belief: float = 0.5  # P(owner) assumed for a domain another agent's evidence points to


class AgentCall(BaseModel):
    agent_id: str
    domain: str
    kind: Literal["investigate", "review", "policy"]
    ok: bool
    seconds: float
    cost_units: float
    detail: str = ""


class Deliberation(BaseModel):
    """Everything that happened for one incident between routing and plan selection."""

    incident_id: str
    routing: RoutingDecision
    proposals: list[Proposal] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)
    calls: list[AgentCall] = Field(default_factory=list)
    transfers: list[str] = Field(default_factory=list)  # pointers followed to another domain
    escalations: list[str] = Field(default_factory=list)  # stronger agents asked after nobody claimed
    unanswered: list[str] = Field(default_factory=list)  # domains where no agent answered
    notes: list[str] = Field(default_factory=list)
    first_round: int | None = None  # proposals[:first_round] came from round 1; the rest were asked later

    def claims(self) -> list[Proposal]:
        """Ownership claims: "it's mine, and here is the fix" (an owner always has a fix that passed the checks)."""
        return [p for p in self.proposals if p.verdict == "owner"]

    def suggestions(self) -> list[Proposal]:
        """Every proposal that carries a usable fix, whatever its verdict: what the optimizer combines.
        Equal to claims() when agents may abstain (must_propose off)."""
        return [p for p in self.proposals if p.actions]

    def reviews_of(self, claim: Proposal) -> list[Review]:
        return [r for r in self.reviews if r.claim_agent == claim.agent_id and r.claim_domain == claim.domain]

    def tally(self, claim: Proposal) -> dict[str, int]:
        counts = Counter(r.stance for r in self.reviews_of(claim))
        return {s: counts.get(s, 0) for s in ("supports", "contradicts", "cannot_tell")}

    @property
    def cost_units(self) -> float:
        return round(sum(c.cost_units for c in self.calls), 2)  # failed calls cost too (time, tokens)


class Orchestrator:
    def __init__(self, router: Router, link: AgentLink, policy: OrchestratorPolicy | None = None):
        self.router, self.registry, self.link = router, router.registry, link
        self.policy = policy or OrchestratorPolicy()

    async def run(self, report: IncidentReport, decision: RoutingDecision | None = None,
                  now: float | None = None) -> Deliberation:
        decision = decision or self.router.route(report, now=now)
        d = Deliberation(incident_id=report.incident_id, routing=decision)
        d.unanswered += [f"{x} (no eligible agent at routing time)" for x in decision.unassigned]

        choices, generalists = [], set()
        for c in ([decision.primary] if decision.primary else []) + decision.consults:
            if self.registry.profile(c.agent_id).role == "generalist":
                if c.agent_id in generalists:  # one generalist call already covers every domain
                    d.notes.append(f"{c.domain}: covered by {c.agent_id}'s other task")
                    continue
                generalists.add(c.agent_id)
            choices.append(c)
        primary = decision.primary
        await asyncio.gather(*(self._investigate(report, c, d, now, "primary" if c is primary else "consult")
                               for c in choices))
        d.first_round = len(d.proposals)

        asked = {c.domain for c in choices}
        if self.policy.follow_pointers:
            await self._follow_pointers(report, decision, d, asked, now)
        if self.policy.escalate and not d.claims():
            await self._escalate(report, decision, d, now)
        await self._cross_check(report, d, now)
        return d

    async def _escalate(self, report: IncidentReport, decision: RoutingDecision, d: Deliberation,
                        now: float | None) -> None:
        domains = list(dict.fromkeys(decision.team + [p.domain for p in d.proposals]))
        for domain in domains:
            if len(d.escalations) >= self.policy.max_escalations:
                return
            await self._ask_stronger(report, d, domain, "nobody claimed ownership", now)
            if d.claims():
                return

    async def second_look(self, report: IncidentReport, d: Deliberation, domain: str, reason: str,
                          now: float | None = None) -> bool:
        """Ask the strongest agent not yet asked to look at `domain`, then cross-check what is new.
        Used by the pipeline when the best plan is unsure. False if the escalation budget is used up
        or nobody is available."""
        if len(d.escalations) >= self.policy.max_escalations:
            return False
        proposal = await self._ask_stronger(report, d, domain, reason, now)
        if proposal is None:
            return False
        await self._cross_check(report, d, now)
        return True

    async def _ask_stronger(self, report: IncidentReport, d: Deliberation, domain: str, reason: str,
                            now: float | None) -> Proposal | None:
        asked_agents = {c.agent_id for c in d.calls if c.kind == "investigate"}
        choice = self.router.pick_agent(domain, d.routing, exclude=asked_agents, by="success", now=now)
        if choice is None:
            d.notes.append(f"{reason}, and no other {domain} agent is available")
            return None
        d.escalations.append(f"{reason}: asked {choice.agent_id} to look at {domain}")
        choice.fallbacks = []  # an escalation is one deliberate extra call, not a retry chain
        return await self._investigate(report, choice, d, now, "consult",
                                       question=f"{reason}; please investigate {domain} in depth")

    # -- round 1 + pointers ----------------------------------------------------------------------
    async def _investigate(self, report: IncidentReport, choice: AgentChoice, d: Deliberation, now: float | None,
                           role: str, question: str | None = None) -> Proposal | None:
        task = InvestigationTask(incident=report, domain=choice.domain, role=role, question=question,
                                 must_propose=self.policy.must_propose)
        attempts = 0
        for agent_id in [choice.agent_id, *choice.fallbacks]:
            if attempts >= self.policy.max_attempts:
                break
            why = self.registry.check(agent_id, choice.domain, now=now)
            if why:
                d.notes.append(f"{choice.domain}: skipped {agent_id} ({why})")
                continue
            attempts += 1
            proposal = await self._call(agent_id, choice.domain, "investigate", task, d, now)
            if proposal is not None:
                d.proposals.append(proposal)
                return proposal
        d.unanswered.append(choice.domain)
        return None

    async def _follow_pointers(self, report: IncidentReport, decision: RoutingDecision, d: Deliberation,
                               asked: set[str], now: float | None) -> None:
        pointed = Counter(p.points_to for p in d.proposals if p.points_to and p.points_to not in asked)
        for domain, _ in pointed.most_common(self.policy.max_extra_domains):
            who = [p.agent_id for p in d.proposals if p.points_to == domain]
            belief = max(decision.owner_probs.get(domain, 0.0), self.policy.transfer_belief)
            choice = self.router.pick_agent(domain, decision, p_owner=belief, now=now)
            if choice is None:
                d.unanswered.append(domain)
                d.notes.append(f"{', '.join(who)} pointed to {domain}, but no {domain} agent is available")
                continue
            d.transfers.append(f"{', '.join(who)} pointed to {domain}: asked {choice.agent_id}")
            asked.add(domain)
            await self._investigate(report, choice, d, now, "consult",
                                    question=f"{', '.join(who)} found evidence pointing to {domain}")

    # -- round 2 ---------------------------------------------------------------------------------
    async def _cross_check(self, report: IncidentReport, d: Deliberation, now: float | None) -> None:
        claims = d.suggestions()  # every proposed fix, claimed or best guess
        mode = self.policy.cross_check
        if not claims or mode == "never":
            return
        if mode == "contested" and len(claims) < 2 and all(c.confidence >= 0.8 for c in claims):
            d.notes.append("cross-check skipped: one confident claim (policy 'contested')")
            return
        jobs = []
        done = {(r.claim_agent, r.claim_domain, r.reviewer, r.domain) for r in d.reviews}
        for claim in claims:
            reviewers = [p for p in d.proposals if (p.agent_id, p.domain) != (claim.agent_id, claim.domain)]
            note = f"{claim.agent_id}'s claim has no reviewer (no other agent took part)"
            if not reviewers and note not in d.notes:
                d.notes.append(note)
            elif reviewers and note in d.notes:  # a second look brought a reviewer after all
                d.notes.remove(note)
            for other in reviewers:
                if (claim.agent_id, claim.domain, other.agent_id, other.domain) in done:
                    continue  # already reviewed (cross-check runs again after a second look)
                task = InvestigationTask(incident=report, domain=other.domain, role="review", claim=claim,
                                         question=f"does your {other.domain} evidence agree with {claim.agent_id}?")
                jobs.append(self._review(other.agent_id, other.domain, task, d, now))
        await asyncio.gather(*jobs)

    async def _review(self, agent_id: str, domain: str, task: InvestigationTask, d: Deliberation,
                      now: float | None) -> None:
        why = self.registry.check(agent_id, domain, now=now)
        if why:
            d.notes.append(f"review by {agent_id} skipped ({why})")
            return
        review = await self._call(agent_id, domain, "review", task, d, now)
        if review is not None:
            d.reviews.append(review)

    # -- one call ----------------------------------------------------------------------------------
    async def _call(self, agent_id: str, domain: str, kind: str, task: InvestigationTask | PolicyRequest,
                    d: Deliberation, now: float | None):
        card, profile = self.registry.card(agent_id), self.registry.profile(agent_id)
        send = getattr(self.link, {"investigate": "investigate", "review": "review", "policy": "review_plan"}[kind])
        self.registry.begin_task(agent_id)
        t0 = time.monotonic()
        try:
            result = await send(card, task)
        except Exception as e:  # a failed agent is recorded and replaced, never fatal
            seconds = time.monotonic() - t0
            self.registry.end_task(agent_id, ok=False, latency_s=seconds, now=now)
            d.calls.append(AgentCall(agent_id=agent_id, domain=domain, kind=kind, ok=False, seconds=round(seconds, 2),
                                     cost_units=profile.cost_units, detail=f"{type(e).__name__}: {e}"[:300]))
            return None
        seconds = time.monotonic() - t0
        self.registry.end_task(agent_id, ok=True, latency_s=seconds, now=now)
        self.registry.heartbeat(agent_id, now=now)  # an answer proves the agent is alive
        d.calls.append(AgentCall(agent_id=agent_id, domain=domain, kind=kind, ok=True, seconds=round(seconds, 2),
                                 cost_units=profile.cost_units))
        return result


def format_deliberation(d: Deliberation) -> str:
    r = d.routing
    out = [f"{d.incident_id}  {r.priority}  team={r.team}  "
           + "  ".join(f"{k}={v:.2f}" for k, v in sorted(r.owner_probs.items(), key=lambda kv: -kv[1]))]
    def answers(proposals) -> list[str]:
        lines = []
        for p in proposals:
            line = f"    {p.agent_id:<16} {p.domain:<9} {p.verdict.upper():<8} {p.confidence:.2f}  {p.root_cause}"
            lines.append(line + (f"   -> points to {p.points_to}" if p.points_to else ""))
            lines += [f"{'':31}fix: {a.intent} on {a.target} {a.params}  [{a.risk} risk]" for a in p.actions]
            lines += [f"{'':31}check: {f.fact}" for f in p.findings if f.tool == CHECK]  # why a fix was dropped
        return lines

    split = len(d.proposals) if d.first_round is None else d.first_round
    out.append("  round 1 - investigate")
    out += answers(d.proposals[:split])
    out += [f"  pointer: {t}" for t in d.transfers]
    out += [f"  escalation: {e}" for e in d.escalations]
    if d.proposals[split:]:
        out.append("  asked after round 1 (pointer / escalation / second look)")
        out += answers(d.proposals[split:])
    fixes = d.suggestions()
    if fixes and d.reviews:
        out.append("  round 2 - cross-check")
        for c in fixes:
            t = d.tally(c)
            kind = "claim" if c.verdict == "owner" else f"best guess ({c.verdict})"
            out.append(f"    {kind} by {c.agent_id}: {t['supports']} supports, {t['contradicts']} contradicts, "
                       f"{t['cannot_tell']} cannot tell")
            out += [f"      {v.reviewer:<16} {v.stance:<11} {v.reason}" for v in d.reviews_of(c)]
    elif not fixes:
        out.append("  no agent proposed a usable fix")
    elif not d.claims():
        out.append("  no agent claimed ownership (only best guesses)")
    failed = [c for c in d.calls if not c.ok]
    out.append(f"  {len(d.calls)} agent calls ({len(failed)} failed), cost {d.cost_units:g} units")
    out += [f"    failed: {c.agent_id} {c.kind}: {c.detail}" for c in failed]
    out += [f"  unanswered: {u}" for u in d.unanswered]
    out += [f"  note: {n}" for n in d.notes]
    return "\n".join(out)
