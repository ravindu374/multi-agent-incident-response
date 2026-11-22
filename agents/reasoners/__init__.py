"""Reasoners turn an agent's tool results into a Proposal. Same interface, swappable:

  RuleReasoner    deterministic checks an engineer would make; needs no LLM; the stand-in that lets
                  the whole pipeline run today. Written from the same knowledge as the synthetic
                  generator, so its scores on synthetic data are NOT a research result.
  OllamaReasoner  a local LLM reads the same tool results and answers in the same JSON shape.
"""
from typing import Protocol

from controller.cards import RoutingProfile
from controller.proposal import InvestigationTask, Proposal, Review

from ..tools import Toolbox


class ReasonerFailed(RuntimeError):
    """No answer: the LLM server failed, timed out, or wrote something unparsable. The agent then fails
    its task instead of answering "unsure" (an A2A task in state failed), so the controller counts a
    failed call and asks the domain's fallback agent, the way it handles a crashed agent."""


class Reasoner(Protocol):
    name: str

    def investigate(self, task: InvestigationTask, agent_id: str, profile: RoutingProfile,
                    toolbox: Toolbox) -> Proposal: ...

    def review(self, task: InvestigationTask, agent_id: str, profile: RoutingProfile,
               toolbox: Toolbox) -> Review: ...  # task.claim is the claim to check
