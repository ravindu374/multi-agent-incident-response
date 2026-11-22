"""The whole controller for one incident:

    route (Block 4) -> deliberate over A2A (5b) -> optimize: one staged plan (5c)
       -> unsure? second look by a stronger agent, cross-check, optimize again
    -> policy agent (5c) -> RemediationPlan for the executor (MCP / Fabric; docs/HANDOVER.md)

If no policy agent answers, the plan is never approved: a human has to look at it. If the policy agent
approves an automatic plan but wants an operator's yes first, the plan becomes kind "approve".

After the executor (and, if a person was involved, the operator) reports back, `learn` stores the verified
outcome in the experience memory (Block 6a, controller/memory.py). With trust "local" or "global" the
memory is also READ (Block 6b, controller/reliability.py): the router and the optimizer use every agent's
verified record instead of the planning numbers. trust "off" = remember only (Block 6a behaviour).
"""
from __future__ import annotations

from pydantic import BaseModel

from .contracts import IncidentReport
from .memory import Experience, ExperienceStore, Feedback, experience_from
from .optimizer import Optimizer, OptimizerPolicy, tiers_from
from .orchestrator import AgentLink, Deliberation, Orchestrator, OrchestratorPolicy
from .plan import PolicyDecision, RemediationPlan
from .policy import PolicyRequest
from .reliability import ReliabilityModel
from .router import Router


class Resolution(BaseModel):
    deliberation: Deliberation
    plan: RemediationPlan


class Controller:
    def __init__(self, router: Router, link: AgentLink, orchestrator_policy: OrchestratorPolicy | None = None,
                 optimizer_policy: OptimizerPolicy | None = None, memory: ExperienceStore | None = None,
                 trust: str = "local"):
        if trust not in ("off", "local", "global"):
            raise ValueError("trust must be 'off', 'local' or 'global'")
        self.registry = router.registry
        self.orchestrator = Orchestrator(router, link, orchestrator_policy)
        self.optimizer = Optimizer(tiers_from(router.registry), optimizer_policy)
        self.memory = memory  # Block 6a: written by learn()
        if memory is not None and trust != "off":  # Block 6b: and read, to trust each agent by its record
            router.trust_model = ReliabilityModel(memory, mode=trust)

    def learn(self, report: IncidentReport, resolution: Resolution, feedback: Feedback) -> Experience:
        """Remember a closed incident: what was decided, and what the verified feedback says about it."""
        experience = experience_from(report, resolution.deliberation, resolution.plan, feedback, self.optimizer.tiers)
        if self.memory is not None:
            self.memory.add(experience)
        return experience

    async def handle(self, report: IncidentReport, now: float | None = None) -> Resolution:
        d = await self.orchestrator.run(report, now=now)
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
    if plan.policy:
        p = plan.policy
        state = ("APPROVED" if p.approved else "NOT APPROVED") + (", needs a human" if p.needs_human else "")
        out.append(f"    policy: {state}")
        out += [f"      - {r}" for r in p.reasons]
    return "\n".join(out)
