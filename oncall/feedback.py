"""Simulated verified feedback for the experience memory (lab code: it reads ground truth).

What a real platform would send back after a plan, and nothing more:
  executor   which stages ran, and whether the health checks passed after each (the oracle; never wrong)
  operator   only when a person was involved (an approval, an escalation, or a plan that did not fix it):
             the postmortem's owner domain and, for every agent's fix, whether it would have fixed it
mode "bandit" (realistic) gives exactly that, so fixes the controller did not try while it fixed an incident
on its own stay unchecked. mode "full" pretends an operator reviews every incident (an upper bound).
noise: the share of operator answers that are wrong (C3: robustness to wrong feedback).
"""
from __future__ import annotations

import random

from controller import DOMAINS
from controller.memory import Feedback
from controller.orchestrator import Deliberation
from controller.plan import RemediationPlan

from .agent_eval import correct_fix


def is_right(p, truth: dict, seen: str) -> bool:
    """Ground truth: the proposal is in the owner's domain and one of its actions is a correct fix."""
    return p.domain == truth["owner"] and any(correct_fix(a, truth, seen) for a in p.actions)


def simulated_feedback(d: Deliberation, plan: RemediationPlan, truth: dict, seen: str, mode: str = "bandit",
                       noise: float = 0.0, rng: random.Random | None = None, approval: bool = True) -> Feedback:
    """Run the plan as the executor would (team_eval.plan_loss uses the same rules) and report what is verified."""
    if mode not in ("bandit", "full"):
        raise ValueError("mode must be 'bandit' or 'full'")
    rng = rng or random.Random(0)
    stages = plan.stages()
    first_right = next((i for i, st in enumerate(stages) if any(correct_fix(a, truth, seen) for a in st)), None)
    fixed_by, failed, person = [], [], True
    if plan.kind == "remediate" and (plan.policy is None or plan.policy.approved or not approval):
        if first_right is None:  # every stage ran, still broken: a person takes over
            failed = [a for st in stages for a in st]
        else:
            fixed_by = stages[first_right]
            failed = [a for st in stages[:first_right] for a in st]
            person = False
    elif plan.kind == "approve" and first_right is not None:
        fixed_by = stages[first_right]  # the person rejected the wrong stages: they never ran
    # escalate, or a plan the policy agent refused: nothing ran, a person fixed it
    feedback = Feedback(incident_id=d.incident_id, resolved=True, fixed_by=fixed_by, failed=failed)
    if person or mode == "full":
        owner = truth["owner"]
        credits = {p.agent_id: is_right(p, truth, seen) for p in d.proposals if p.actions}
        if noise:
            credits = {a: (not ok) if rng.random() < noise else ok for a, ok in credits.items()}
            if rng.random() < noise:
                owner = rng.choice([x for x in DOMAINS if x != owner])
        feedback.owner, feedback.credits, feedback.source = owner, credits, "operator"
    return feedback
