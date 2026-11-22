"""Block 5c - Policy agent rules: the last gate before a plan leaves the controller.

    approved = False   the plan breaks a hard rule (non-executable action, too many actions), or it is an
                       escalation: nothing may run automatically
    needs_human = True an operator must say yes first: high-risk actions, actions that cannot be undone
                       while the plan is unsure, P1 incidents with an unsure plan (approval_reasons), any
                       "approve" plan, any escalation

The optimizer knows these rules too (approval_reasons): it offers a fix to run automatically only if
this gate would let it, and otherwise offers it for a human's approval (plan kind "approve").

Deterministic on purpose: a safety gate must give the same answer every time, and it must not be talked
out of a rule. (The card's model field leaves room for an LLM to *explain* decisions later, never to make
them.) Served over A2A as the "policy" agent (agents/server.py); runs the same rules in-process.
"""
from __future__ import annotations

from pydantic import BaseModel

from .checks import executable_problems
from .plan import INTENTS, PlannedAction, PolicyDecision, RemediationPlan

MAX_ACTIONS = 6
SURE = 0.8  # below this confidence, irreversible actions need an operator
P1_SURE = 0.6


class PolicyRequest(BaseModel):
    plan: RemediationPlan
    priority: str = "P3"


def approval_reasons(actions: list[PlannedAction], confidence: float, priority: str = "P3") -> list[str]:
    """Why these actions may not run without an operator's yes (empty: they may run automatically)."""
    soft = []
    for a in actions:
        if a.risk == "high":
            soft.append(f"stage {a.stage} {a.intent} is high risk")
        if not INTENTS[a.intent].reversible and confidence < SURE:
            soft.append(f"stage {a.stage} {a.intent} cannot be undone and the plan is only {confidence:.0%} sure")
    if priority == "P1" and confidence < P1_SURE:
        soft.append(f"P1 incident and the plan is only {confidence:.0%} sure")
    return soft


def check_plan(plan: RemediationPlan, priority: str = "P3") -> PolicyDecision:
    if plan.kind == "escalate":
        return PolicyDecision(approved=False, needs_human=True,
                              reasons=["escalation: a human decides; nothing runs automatically"])
    hard = []
    if len(plan.actions) > MAX_ACTIONS:
        hard.append(f"{len(plan.actions)} actions is more than the {MAX_ACTIONS} allowed in one plan")
    for a in plan.actions:
        for problem in executable_problems(a):
            hard.append(f"stage {a.stage} {a.intent}: {problem}")
    soft = approval_reasons(plan.actions, plan.confidence, priority)
    if hard:
        return PolicyDecision(approved=False, needs_human=True, reasons=hard + soft)
    if plan.kind == "approve":
        return PolicyDecision(approved=True, needs_human=True,
                              reasons=soft + ["a human approves each stage before it runs"])
    return PolicyDecision(approved=True, needs_human=bool(soft),
                          reasons=soft or ["low/medium risk, reversible or confident: may run automatically"])
