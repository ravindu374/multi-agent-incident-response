"""The policy agent: approves, rejects or escalates a RemediationPlan (rules in controller/policy.py).

It sits behind its own Agent Card ("policy") and A2A server like the experts, so the controller asks it
over A2A before any plan is handed to the executor.
"""
from __future__ import annotations

from controller.cards import AgentCard
from controller.plan import PolicyDecision
from controller.policy import PolicyRequest, check_plan


class PolicyAgent:
    def __init__(self, card: AgentCard):
        self.card, self.profile = card, card.routing_profile()
        if self.profile.role != "policy":
            raise ValueError(f"{card.name} is not a policy agent")

    @property
    def agent_id(self) -> str:
        return self.card.name

    def decide(self, request: PolicyRequest) -> PolicyDecision:
        return check_plan(request.plan, request.priority)
