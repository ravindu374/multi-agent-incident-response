"""The whole controller for one incident:

    route (Block 4) -> deliberate over A2A (5b) -> optimize: one staged plan (5c)
       -> unsure? second look by a stronger agent, cross-check, optimize again
    -> policy agent (5c) -> RemediationPlan for the executor (MCP / Fabric; docs/HANDOVER.md)

If no policy agent answers, the plan is never approved: a human has to look at it. If the policy agent
approves an automatic plan but wants an operator's yes first, the plan becomes kind "approve".

With a history agent (Block 6c, history="on") the incident is looked up FIRST, once:
    history agent -> similar past incidents + every agent's verified record
       -> route with that record (6b) -> deliberate, the experts seeing the past incidents
       -> optimize, counting precedent for each fix -> ... as above; the plan lists the precedents
After the executor (and, if a person was involved, the operator) reports back, `remember` turns the
verified outcome into an Experience (controller/memory.py) and sends it to the history agent, which stores
it: the learning loop. history="remember" stores outcomes without looking anything up (Block 6a).
"""
from __future__ import annotations

from pydantic import BaseModel

from .contracts import IncidentReport
from .history import HistoryQuery, HistoryReport, RememberRequest
from .memory import Experience, Feedback, experience_from
from .optimizer import Optimizer, OptimizerPolicy, tiers_from
from .orchestrator import AgentCall, AgentLink, Deliberation, Orchestrator, OrchestratorPolicy
from .plan import PolicyDecision, RemediationPlan
from .policy import PolicyRequest
from .router import Router

HISTORY_MODES = ("off", "remember", "on")


class Resolution(BaseModel):
    deliberation: Deliberation
    plan: RemediationPlan


class Controller:
    def __init__(self, router: Router, link: AgentLink, orchestrator_policy: OrchestratorPolicy | None = None,
                 optimizer_policy: OptimizerPolicy | None = None, history: str = "off"):
        """history: "off" (no history agent), "remember" (only store verified outcomes), "on" (look every
        incident up first, and store outcomes)."""
        if history not in HISTORY_MODES:
            raise ValueError(f"history must be one of {HISTORY_MODES}")
        self.registry = router.registry
        self.orchestrator = Orchestrator(router, link, orchestrator_policy)
        self.optimizer = Optimizer(tiers_from(router.registry), optimizer_policy)
        self.history = history

    def experience(self, report: IncidentReport, resolution: Resolution, feedback: Feedback) -> Experience:
        """A closed incident: what was decided, and what the verified feedback says about it."""
        return experience_from(report, resolution.deliberation, resolution.plan, feedback, self.optimizer.tiers)

    async def remember(self, report: IncidentReport, resolution: Resolution, feedback: Feedback,
                       now: float | None = None) -> Experience:
        """Close the loop: send the verified outcome to the history agent (if history is not "off")."""
        experience = self.experience(report, resolution, feedback)
        if self.history != "off":
            request = RememberRequest(experience=experience.model_dump(mode="json"))
            for agent_id in self.registry.history_agents(now=now):
                ack, _ = await self.orchestrator.call(agent_id, "-", "remember", request, now)
                if ack is not None:
                    break
        return experience

    async def look_up(self, report: IncidentReport, now: float | None = None
                      ) -> tuple[HistoryReport | None, list[AgentCall]]:
        """Ask the history agent about this incident (the first live one that answers)."""
        calls = []
        for agent_id in self.registry.history_agents(now=now):
            found, call = await self.orchestrator.call(agent_id, "-", "history", HistoryQuery(incident=report), now)
            calls.append(call)
            if found is not None:
                return found, calls
        return None, calls

    async def handle(self, report: IncidentReport, now: float | None = None) -> Resolution:
        found, calls = await self.look_up(report, now) if self.history == "on" else (None, [])
        decision = self.orchestrator.router.route(report, now=now, trust=found.trust if found else None)
        d = await self.orchestrator.run(report, decision, now=now, history=found, calls=calls)
        if self.history == "on" and found is None:
            d.notes.append("no history agent answered: agents trusted by planning numbers, no past incidents")
        plan = self.optimizer.plan(d)
        threshold = self.orchestrator.policy.second_look_below
        if plan.confidence < threshold:
            reason = (f"best plan only {plan.confidence:.0%} sure" if plan.kind == "remediate"
                      else "no plan worth acting on")
            if await self.orchestrator.second_look(report, d, plan.owner, reason, now):
                plan = self.optimizer.plan(d)
        plan.policy = await self._policy(plan, d, now)
        if (self.optimizer.policy.approval and plan.kind == "remediate" and plan.policy.approved
                and plan.policy.needs_human):  # the gate wants an operator's yes: the plan waits for one
            plan.kind = "approve"
        return Resolution(deliberation=d, plan=plan)

    async def _policy(self, plan: RemediationPlan, d: Deliberation, now: float | None) -> PolicyDecision:
        request = PolicyRequest(plan=plan, priority=d.routing.priority)
        for agent_id in self.registry.policy_agents(now=now):
            decision = await self.orchestrator._call(agent_id, "-", "policy", request, d, now)
            if decision is not None:
                return decision
        d.notes.append("no policy agent answered")
        return PolicyDecision(approved=False, needs_human=True,
                              reasons=["no policy agent answered: a human must check this plan"])


def format_plan(plan: RemediationPlan) -> str:
    out = [f"  PLAN ({plan.kind}) owner={plan.owner}  confidence {plan.confidence:.2f}  "
           f"expected loss {plan.expected_loss}"]
    out.append(f"    root cause: {plan.root_cause}")
    for n, stage in enumerate(plan.stages(), 1):
        then = "" if n == 1 else "  (only if still broken)"
        if plan.kind == "approve":
            then += "  (after a human's yes)"
        for a in stage:
            out.append(f"    stage {n}: {a.intent} on {a.target} {a.params}  [{a.risk} risk]{then}")
    out.append("    options considered:")
    for o in plan.options:
        mark = ">" if o.chosen else " "
        out.append(f"     {mark} {o.label:<55} P(fixed)={o.p_fixed:.2f}  loss={o.expected_loss:.2f}")
    if plan.precedents:
        out.append(f"    similar past incidents ({len(plan.precedents)}):")
        out += [f"      {p.similarity:.2f}  {p.incident_id}  owner={p.owner or '?'}  fixed by "
                f"{'; '.join(p.fix) or 'a person'}  ({p.verified_by})" for p in plan.precedents]
    if plan.policy:
        p = plan.policy
        state = ("APPROVED" if p.approved else "NOT APPROVED") + (", needs a human" if p.needs_human else "")
        out.append(f"    policy: {state}")
        out += [f"      - {r}" for r in p.reasons]
    return "\n".join(out)
