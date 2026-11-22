"""In-process link to the expert agents: the orchestrator's rounds without HTTP.

Same agents, same checks, same Proposal/Review objects as over A2A (tests/test_a2a_server.py shows the
two paths give identical proposals); only the transport differs. Used by tests, by `python -m oncall
eval-team`, and for quick runs when no agent servers are running.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from controller.cards import AgentCard
from controller.plan import PolicyDecision
from controller.policy import PolicyRequest
from controller.pool import default_pool
from controller.proposal import InvestigationTask, Proposal, Review

from .policy_agent import PolicyAgent
from .reasoners.ollama import OllamaReasoner
from .reasoners.rules import RuleReasoner
from .specialist import SpecialistAgent
from .tools import EvidenceSource, RecordedEvidence


class LocalLink:
    def __init__(self, agents: dict[str, SpecialistAgent]):
        self.agents = agents

    async def investigate(self, card: AgentCard, task: InvestigationTask) -> Proposal:
        return await asyncio.to_thread(self.agents[card.name].handle, task)

    async def review(self, card: AgentCard, task: InvestigationTask) -> Review:
        return await asyncio.to_thread(self.agents[card.name].review, task)

    async def review_plan(self, card: AgentCard, request: PolicyRequest) -> PolicyDecision:
        return PolicyAgent(card).decide(request)


def local_agents(evidence: EvidenceSource, reasoner: str = "rules", model: str | None = None,
                 cards: list[AgentCard] | None = None) -> dict[str, SpecialistAgent]:
    """Every investigating agent of the pool (all but the policy agent), sharing one evidence source."""
    brain = OllamaReasoner(model=model) if reasoner == "ollama" else RuleReasoner()
    return {c.name: SpecialistAgent(c, brain, evidence) for c in (cards or default_pool())
            if c.routing_profile().role != "policy"}


def local_link(incidents: Path, reasoner: str = "rules", model: str | None = None) -> LocalLink:
    return LocalLink(local_agents(RecordedEvidence.from_incidents_file(incidents), reasoner, model))
